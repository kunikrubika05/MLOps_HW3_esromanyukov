import html
import json
from pathlib import Path

from fraud_detection.database import connect
from pipeline.common import COLUMNS, REPORTS, TRAINING, WINDOW_MONTHS, write_json

RULES = [
    ('Контракт витрины', 'Нужные столбцы, типы и заполненные обязательные поля'),
    (
        'Допустимость значений',
        'Конечные неотрицательные суммы, целое население и допустимые координаты',
    ),
    (
        'Целостность месяцев',
        'Время соответствует месяцу строки, календарные месяцы не пропущены',
    ),
    ('Корректность разметки', 'Метки принимают только 0 или 1'),
    (
        'Целостность загрузки',
        'Ключи уникальны, размеры добавленных месяцев совпадают с размером батчей',
    ),
    ('Разбиение по времени', 'Обучение раньше проверки, строки частей не пересекаются'),
]


def check_dataset() -> str:
    """
    Функция проверяет витрину по шести правилам качества данных и сохраняет результаты в
    JSON
    При повторном исполнении таски заново проверяет витрину и заменяет JSON своего
    запуска
    Данные витрины функция не изменяет
    """
    from airflow.operators.python import get_current_context

    identifier = get_current_context()['run_id'].replace('/', '_').replace(':', '_')
    results = [
        {'rule': name, 'description': description, 'violations': None, 'passed': None}
        for name, description in RULES
    ]
    report = {'run_id': identifier, 'rows': 0, 'rules': results, 'readiness': None}
    expected = dict.fromkeys(COLUMNS, 'text')
    expected.update(
        dict.fromkeys(
            ['amount', 'population_city', 'lat', 'lon', 'merchant_lat', 'merchant_lon'],
            'double precision',
        )
    )
    expected.update(
        data_month='date',
        source_row='bigint',
        transaction_time='timestamp without time zone',
        target='smallint',
    )
    with connect() as connection, connection.transaction():
        connection.execute('set transaction isolation level repeatable read')
        columns = connection.execute(
            'select column_name, data_type from information_schema.columns '
            "where table_name = 'labeled_transactions' and table_schema = 'public'"
        ).fetchall()
        actual = {row['column_name']: row['data_type'] for row in columns}
        mismatches = [
            name for name, dtype in expected.items() if actual.get(name) != dtype
        ]
        report['invalid_columns'] = mismatches
        if mismatches:
            results[0].update(violations=len(mismatches), passed=False)
            results[0]['description'] += ': ' + ', '.join(mismatches)
        else:
            counts = connection.execute(
                'select count(*) as rows, '
                'count(*) filter (where transaction_time is null or target is null '
                'or data_month is null or source_row is null) as required, '
                'count(*) filter (where amount is null or population_city is null '
                "or not (amount >= 0 and amount < 'Infinity'::float8) "
                'or not (population_city >= 0 '
                "and population_city < 'Infinity'::float8) "
                'or population_city <> trunc(population_city) '
                'or lat is null or lon is null '
                'or merchant_lat is null or merchant_lon is null '
                'or not (abs(lat) <= 90 and abs(lon) <= 180 '
                'and abs(merchant_lat) <= 90 and abs(merchant_lon) <= 180)) as values, '
                'count(*) filter (where transaction_time is null '
                'or not isfinite(transaction_time) '
                'or data_month is null or not isfinite(data_month) '
                "or date_trunc('month', transaction_time)::date <> data_month) "
                'as time, '
                'count(*) filter (where target is null '
                'or target not in (0, 1)) as target, '
                'count(*) - count(distinct (data_month, source_row)) as duplicates '
                'from labeled_transactions'
            ).fetchone()
            months = connection.execute(
                'select data_month, count(*) as rows from labeled_transactions '
                'where isfinite(data_month) group by data_month order by data_month'
            ).fetchall()
            gaps = connection.execute(
                'select count(*) as n from generate_series('
                '(select min(data_month) from labeled_transactions '
                'where isfinite(data_month)), '
                '(select max(data_month) from labeled_transactions '
                'where isfinite(data_month)), '
                "interval '1 month') month where not exists ("
                'select 1 from labeled_transactions where data_month = month::date)'
            ).fetchone()['n']
            sizes = {row['data_month']: row['rows'] for row in months}
            supplied = connection.execute(
                'select month, rows from supplied_months '
                'where month >= (select max(data_month) from labeled_transactions) '
                "- interval '35 months'"
            ).fetchall()
            batch_errors = sum(
                sizes.get(row['month'], 0) != row['rows'] for row in supplied
            )
            bad = [
                counts['required'],
                counts['values'],
                counts['time'] + gaps,
                counts['target'],
                counts['duplicates'] + batch_errors,
            ]
            for result, count in zip(results, bad):
                result.update(
                    violations=count, passed=count == 0 and counts['rows'] > 0
                )
            report['rows'] = counts['rows']
            report['missing_months'] = gaps
            report['batch_size_mismatches'] = batch_errors
            latest = connection.execute(
                'select month from training_cycles order by month desc limit 1'
            ).fetchone()
            if latest:
                try:
                    split = inspect_split(latest['month'].isoformat())
                except (OSError, ValueError, TypeError) as error:
                    split = {'violations': 1, 'error': str(error)}
                results[5].update(
                    violations=split['violations'],
                    passed=split['violations'] == 0,
                )
                if split.get('error'):
                    results[5]['description'] += ': ' + split['error']
                report['split_month'] = latest['month'].isoformat()
                results[5]['description'] += ': ' + report['split_month'][:7]
                report['readiness'] = split.get('readiness')
    path = REPORTS / f'dq_{identifier}.json'
    write_json(path, report)
    return str(path)


def inspect_split(
    month: str,
) -> dict:
    import pandas as pd

    folder = TRAINING / 'retrain' / month
    columns = ['data_month', 'source_row', 'transaction_time', 'target']
    try:
        train = pd.read_parquet(folder / 'train.parquet', columns=columns)
        test = pd.read_parquet(folder / 'test.parquet', columns=columns)
    except (OSError, ValueError) as error:
        return {'violations': 1, 'error': str(error)}
    end = pd.Period(month, freq='M')
    start = end - (WINDOW_MONTHS - 1)
    train_dates = pd.to_datetime(
        train['transaction_time'], errors='coerce', utc=True
    ).dt.tz_localize(None)
    test_dates = pd.to_datetime(
        test['transaction_time'], errors='coerce', utc=True
    ).dt.tz_localize(None)
    keys = ['data_month', 'source_row']
    overlap = len(train[keys].merge(test[keys], on=keys))
    chronological = (
        not train.empty and not test.empty and train_dates.max() < test_dates.min()
    )
    correct_months = set(train_dates.dt.to_period('M')) == set(
        pd.period_range(start, end - 1, freq='M')
    ) and set(test_dates.dt.to_period('M')) == {end}
    nulls = int(
        train[columns].isna().any(axis=1).sum()
        + test[columns].isna().any(axis=1).sum()
        + train_dates.isna().sum()
        + test_dates.isna().sum()
    )
    duplicates = int(train.duplicated(keys).sum() + test.duplicated(keys).sum())
    train_classes = sorted(train['target'].dropna().unique().tolist())
    test_classes = sorted(test['target'].dropna().unique().tolist())
    return {
        'violations': overlap
        + nulls
        + duplicates
        + int(not chronological)
        + int(not correct_months),
        'readiness': {
            'train_classes': train_classes,
            'test_classes': test_classes,
            'ready': train_classes == [0, 1] and test_classes == [0, 1],
        },
    }


def publish_report(
    report_path: str,
) -> str:
    """
    Функция читает JSON проверки витрины и сохраняет HTML-отчёт в каталог отчётов
    Если правила качества нарушены, таска завершается ошибкой после сохранения отчёта
    При повторном исполнении таски заменяет тот же HTML-файл
    """
    report = json.loads(Path(report_path).read_text())
    rows = ''.join(
        '<tr><td>'
        + html.escape(rule['rule'])
        + '</td><td>'
        + html.escape(rule['description'])
        + '</td><td>'
        + (str(rule['violations']) if rule['violations'] is not None else '—')
        + '</td><td>'
        + (
            'Пройдена'
            if rule['passed'] is True
            else 'Ошибка'
            if rule['passed'] is False
            else 'Ожидание данных'
        )
        + '</td></tr>'
        for rule in report['rules']
    )
    readiness = report['readiness']
    ready_text = 'Ожидание файлов обучающей и проверочной частей'
    if readiness is not None:
        ready_text = (
            'Оба класса присутствуют'
            if readiness['ready']
            else 'Недостаточно классов для обучения или расчёта метрик'
        )
    document = (
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<title>Качество данных</title><style>body{font:16px sans-serif;margin:32px}'
        'table{border-collapse:collapse}'
        'td,th{border:1px solid #aaa;padding:10px}</style>'
        '<h1>Качество данных</h1><p>Запуск: '
        + html.escape(report['run_id'])
        + '. Строк: '
        + str(report['rows'])
        + '</p><table><tr><th>Правило</th>'
        '<th>Условие</th><th>Нарушений</th><th>Результат</th></tr>'
        + rows
        + '</table><h2>Готовность к обучению и расчёту метрик</h2><p>'
        + html.escape(ready_text)
        + '</p></html>'
    )
    path = Path(report_path).with_suffix('.html')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(document, encoding='utf-8')
    temporary.replace(path)
    if any(rule['passed'] is False for rule in report['rules']):
        raise ValueError(f'Data quality checks failed: {path}')
    return str(path)
