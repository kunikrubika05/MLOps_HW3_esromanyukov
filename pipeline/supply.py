import calendar

from fraud_detection.database import connect
from pipeline.common import (
    BATCH_SIZE,
    COLUMNS,
    HISTORY_MONTHS,
    TRAINING,
    read_frame,
    reserve_month,
)


def sample_next_month() -> dict:
    """
    Функция выбирает до 100000 строк витрины, переносит даты транзакций в следующий
    календарный месяц и сохраняет батч в файл
    При повторном исполнении таски после ошибки использует тот же месяц и готовый файл,
    если он есть
    Если файла нет, заново выбирает строки витрины для того же месяца
    """
    import pandas as pd
    from airflow.operators.python import get_current_context

    month = reserve_month('supply', get_current_context()['run_id'])
    folder = TRAINING / 'supply' / month
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'batch.parquet'
    if not path.exists():
        frame = read_frame(
            'select * from (select distinct on (source_row) * '
            'from labeled_transactions '
            'order by source_row, transaction_time desc) source '
            'order by md5(source_row::text || %s) limit %s',
            (month, BATCH_SIZE),
        )
        if frame.empty:
            raise ValueError('Labeled dataset is empty')
        target = pd.Timestamp(month)
        dates = pd.to_datetime(frame['transaction_time'])
        frame['transaction_time'] = pd.to_datetime(
            {
                'year': target.year,
                'month': target.month,
                'day': dates.dt.day.clip(
                    upper=calendar.monthrange(target.year, target.month)[1]
                ),
                'hour': dates.dt.hour,
                'minute': dates.dt.minute,
                'second': dates.dt.second,
            }
        )
        frame['data_month'] = target.date()
        temporary = path.with_suffix('.tmp')
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    return {'month': month, 'path': str(path)}


def append_labeled_batch(
    info: dict,
) -> dict:
    """
    Функция читает файл батча и одной транзакцией сохраняет его строки в витрину и
    отметку о загруженном месяце
    При повторном исполнении таски пропускает уже загруженный месяц, чтобы не добавить
    строки повторно
    """
    import pandas as pd

    with connect() as connection, connection.transaction():
        connection.execute('select pg_advisory_xact_lock(301)')
        done = connection.execute(
            'select rows from supplied_months where month = %s', (info['month'],)
        ).fetchone()
        if done:
            return {**info, 'rows': done['rows']}
        frame = pd.read_parquet(info['path'])
        columns = ['data_month', 'source_row', *COLUMNS]
        clean = frame[columns].astype(object).where(pd.notna(frame[columns]), None)
        with connection.cursor() as cursor:
            with cursor.copy(
                'copy labeled_transactions (' + ', '.join(columns) + ') from stdin'
            ) as copy:
                for row in clean.itertuples(index=False, name=None):
                    copy.write_row(row)
        connection.execute(
            'insert into supplied_months values (%s, %s)', (info['month'], len(frame))
        )
    return {**info, 'rows': len(frame)}


def prune_old_months(
    info: dict,
) -> dict:
    """
    Функция удаляет из витрины строки старше последних 36 месяцев относительно месяца
    добавленного батча
    При повторном исполнении таски использует ту же границу удаления, не сокращая
    историю ещё раз
    """
    with connect() as connection:
        removed = connection.execute(
            'delete from labeled_transactions '
            "where data_month < (%s::date - (%s - 1) * interval '1 month')::date",
            (info['month'], HISTORY_MONTHS),
        ).rowcount
    return {**info, 'deleted': removed}
