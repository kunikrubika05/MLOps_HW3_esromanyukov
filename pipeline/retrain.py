from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
from psycopg.types.json import Jsonb

from fraud_detection.database import connect
from pipeline.common import (
    CANDIDATES,
    MODELS,
    TRAINING,
    WINDOW_MONTHS,
    read_frame,
    reserve_month,
    write_frame,
    write_json,
)


def extract_window() -> dict:
    """
    Функция сохраняет данные из витрины за три месяца в файл для обучения
    При повторном исполнении таски после ошибки использует те же даты окна
    Если файл сохранился, использует его, иначе заново читает витрину за те же даты
    """
    import pandas as pd
    from airflow.exceptions import AirflowSkipException
    from airflow.operators.python import get_current_context

    month = reserve_month('retrain', get_current_context()['run_id'])
    folder = TRAINING / 'retrain' / month
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'window.parquet'
    if not path.exists():
        end = pd.Timestamp(month)
        start = (end.to_period('M') - (WINDOW_MONTHS - 1)).to_timestamp()
        frame = read_frame(
            'select * from labeled_transactions '
            'where transaction_time >= %s and transaction_time < %s '
            'order by transaction_time, source_row',
            (start.to_pydatetime(), (end + pd.DateOffset(months=1)).to_pydatetime()),
        )
        if (
            frame.empty
            or end.to_period('M')
            not in pd.to_datetime(frame['transaction_time']).dt.to_period('M').unique()
        ):
            raise AirflowSkipException(f'Month {month} is not available')
        months = pd.to_datetime(frame['transaction_time']).dt.to_period('M').unique()
        if len(months) != WINDOW_MONTHS:
            raise ValueError('Training window must contain three consecutive months')
        temporary = path.with_suffix('.tmp')
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    return {
        'month': month,
        'folder': str(folder),
        'version': month[:7].replace('-', ''),
    }


def time_train_test_split(
    info: dict,
) -> dict:
    """
    Функция читает файл трёхмесячного окна и сохраняет два первых месяца в train,
    последний — в test
    При повторном исполнении таски после ошибки заново читает тот же файл окна и
    перезаписывает файлы train и test
    """
    import pandas as pd

    folder = Path(info['folder'])
    frame = pd.read_parquet(folder / 'window.parquet')
    periods = pd.to_datetime(frame['transaction_time']).dt.to_period('M')
    train = frame.loc[periods < pd.Period(info['month'], freq='M')]
    test = frame.loc[periods == pd.Period(info['month'], freq='M')]
    if train.empty or test.empty or train['target'].nunique() < 2:
        raise ValueError('Training split must contain rows of both classes')
    write_frame(train, folder / 'train.parquet')
    write_frame(test, folder / 'test.parquet')
    return {**info, 'train_rows': len(train), 'test_rows': len(test)}


def preprocess_dataset(
    info: dict,
) -> dict:
    """
    Функция читает train и test, рассчитывает параметры препроцессинга только по train
    Сохраняет обработанные признаки и параметры препроцессинга для новой модели
    При повторном исполнении таски после ошибки перезаписывает эти файлы из тех же train
    и test
    """
    import pandas as pd

    from fraud_detection.preprocessing import fit_state, preprocess

    folder = Path(info['folder'])
    train = pd.read_parquet(folder / 'train.parquet')
    test = pd.read_parquet(folder / 'test.parquet')
    state = fit_state(train)
    candidate = CANDIDATES / info['version']
    candidate.mkdir(parents=True, exist_ok=True)
    write_json(candidate / 'preprocessing.json', state)
    for name, frame in (('train', train), ('test', test)):
        features = preprocess(frame, state)
        features['target'] = frame['target'].to_numpy()
        features.to_parquet(folder / f'{name}_features.parquet', index=False)
    return info


def fit_catboost(
    info: dict,
) -> dict:
    """
    Функция обучает модель на обработанных признаках train и сохраняет её в файл
    кандидата
    При повторном исполнении таски после ошибки переобучает и заменяет только кандидата,
    не меняя модель в сервисе скоринга
    """
    import pandas as pd

    from fraud_detection.training import train_model

    folder = Path(info['folder'])
    frame = pd.read_parquet(folder / 'train_features.parquet')
    labels = frame.pop('target')
    model = train_model(frame, labels)
    candidate = CANDIDATES / info['version']
    temporary = candidate / '.model.tmp'
    model.save_model(temporary)
    os.replace(temporary, candidate / 'model.cbm')
    return info


def model_metrics(
    directory: Path,
    frame: pd.DataFrame,
) -> dict:
    from catboost import CatBoostClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    from fraud_detection.preprocessing import preprocess

    model = CatBoostClassifier()
    model.load_model(directory / 'model.cbm')
    state = json.loads((directory / 'preprocessing.json').read_text())
    scores = model.predict_proba(preprocess(frame, state), thread_count=2)[:, 1]
    return {
        'pr_auc': float(average_precision_score(frame['target'], scores)),
        'roc_auc': float(roc_auc_score(frame['target'], scores)),
    }


def compare_models(
    info: dict,
) -> dict:
    """
    Функция сравнивает новую модель с моделью сервиса скоринга на одном test по PR-AUC и
    ROC-AUC
    Сохраняет решение о публикации новой модели: обе метрики должны быть не ниже прежних
    При повторном исполнении таски после ошибки использует сохранённое решение, если оно
    есть
    """
    import pandas as pd

    decision_path = CANDIDATES / info['version'] / 'decision.json'
    if decision_path.exists():
        return info
    frame = pd.read_parquet(Path(info['folder']) / 'test.parquet')
    current = (MODELS / 'latest.txt').read_text().strip()
    decision = {**info, 'incumbent_version': current, 'published': False}
    if frame['target'].nunique() < 2:
        decision['reason'] = 'Test month must contain both target classes'
    else:
        candidate = model_metrics(CANDIDATES / info['version'], frame)
        incumbent = model_metrics(MODELS / current, frame)
        decision.update(candidate_metrics=candidate, incumbent_metrics=incumbent)
        decision['published'] = all(
            candidate[key] >= incumbent[key] for key in candidate
        )
        decision['reason'] = (
            'Both metrics are not worse'
            if decision['published']
            else 'Candidate metrics decreased'
        )
    write_json(decision_path, decision)
    return info


def publish_model(
    info: dict,
) -> dict:
    """
    Функция читает решение о публикации и при отсутствии ухудшения передаёт новую модель
    и её препроцессинг сервису скоринга
    Отмечает месяц как обработанный, даже если новая модель отклонена
    При повторном исполнении таски для уже обработанного месяца не меняет активную
    модель и не сдвигает месяц обучения снова
    """
    candidate = CANDIDATES / info['version']
    decision = json.loads((candidate / 'decision.json').read_text())
    with connect() as connection, connection.transaction():
        connection.execute('select pg_advisory_xact_lock(301)')
        done = connection.execute(
            'select decision from training_cycles where month = %s', (info['month'],)
        ).fetchone()
        if done:
            return done['decision']
        for filename in ('model.cbm', 'preprocessing.json'):
            if not (candidate / filename).is_file():
                raise FileNotFoundError(candidate / filename)
        if decision['published']:
            final = MODELS / info['version']
            if not final.exists():
                staging = MODELS / ('.' + info['version'] + '.tmp')
                staging.mkdir(parents=True, exist_ok=True)
                for filename in ('model.cbm', 'preprocessing.json', 'decision.json'):
                    shutil.copyfile(candidate / filename, staging / filename)
                os.replace(staging, final)
            temporary = MODELS / '.latest.tmp'
            temporary.write_text(info['version'] + '\n')
            os.replace(temporary, MODELS / 'latest.txt')
        connection.execute(
            'insert into training_cycles values (%s, %s, %s, %s)',
            (info['month'], info['version'], decision['published'], Jsonb(decision)),
        )
        connection.execute(
            'update training_state set next_month = '
            "greatest(next_month, (%s::date + interval '1 month')::date) where id = 1",
            (info['month'],),
        )
    return decision
