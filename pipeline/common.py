from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from fraud_detection.database import connect

ROOT = Path(os.getenv('RUNTIME_DIR', '/runtime'))
MODELS = ROOT / 'models'
CANDIDATES = ROOT / 'candidates'
TRAINING = ROOT / 'training'
REPORTS = ROOT / 'reports'
WINDOW_MONTHS = 3
HISTORY_MONTHS = 36
BATCH_SIZE = 100_000
COLUMNS = [
    'transaction_time',
    'merch',
    'cat_id',
    'amount',
    'name_1',
    'name_2',
    'gender',
    'street',
    'one_city',
    'us_state',
    'post_code',
    'lat',
    'lon',
    'population_city',
    'jobs',
    'merchant_lat',
    'merchant_lon',
    'target',
]


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    os.replace(temporary, path)


def read_frame(
    sql: str,
    parameters: tuple = (),
) -> pd.DataFrame:
    import pandas as pd

    with connect() as connection:
        return pd.DataFrame(connection.execute(sql, parameters).fetchall())


def write_frame(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    temporary = path.with_suffix('.tmp')
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def reserve_month(
    flow: str,
    run_id: str,
) -> str:
    with connect() as connection, connection.transaction():
        connection.execute('select pg_advisory_xact_lock(301)')
        stored = connection.execute(
            'select month from run_months where flow = %s and run_id = %s',
            (flow, run_id),
        ).fetchone()
        if stored:
            return stored['month'].isoformat()
        if flow == 'retrain':
            month = connection.execute(
                'select next_month as month from training_state where id = 1'
            ).fetchone()['month']
        else:
            month = connection.execute(
                "select (max(data_month) + interval '1 month')::date as month "
                'from labeled_transactions'
            ).fetchone()['month']
        if month is None:
            raise ValueError('Labeled dataset is empty')
        connection.execute(
            'insert into run_months (flow, run_id, month) values (%s, %s, %s)',
            (flow, run_id, month),
        )
        return month.isoformat()
