import os

import pandas as pd

from fraud_detection.database import connect
from fraud_detection.preprocessing import fit_state, preprocess
from fraud_detection.training import train_model
from pipeline.common import COLUMNS, MODELS, ROOT, write_json


def main() -> None:
    for directory in ('models', 'candidates', 'training', 'reports', 'airflow', 'logs'):
        path = ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o777)
    with connect() as connection, connection.transaction():
        connection.execute('select pg_advisory_xact_lock(301)')
        initialized = connection.execute('select 1 from training_state').fetchone()
        if not initialized:
            row_number = 0
            for chunk in pd.read_csv(
                os.getenv('TRAIN_DATA_CSV', '/data/train.csv'), chunksize=50000
            ):
                chunk['transaction_time'] = pd.to_datetime(chunk['transaction_time'])
                months = (
                    chunk['transaction_time']
                    .dt.to_period('M')
                    .dt.to_timestamp()
                    .dt.date
                )
                clean = (
                    chunk[COLUMNS].astype(object).where(pd.notna(chunk[COLUMNS]), None)
                )
                with connection.cursor() as cursor:
                    with cursor.copy(
                        'copy labeled_transactions (data_month, source_row, '
                        + ', '.join(COLUMNS)
                        + ') from stdin'
                    ) as copy:
                        for month, row in zip(
                            months,
                            clean.itertuples(index=False, name=None),
                            strict=True,
                        ):
                            row_number += 1
                            copy.write_row((month, row_number, *row))
            first = connection.execute(
                'select min(data_month) as month from labeled_transactions'
            ).fetchone()['month']
            if first is None:
                raise ValueError('Training CSV is empty')
            connection.execute(
                'insert into training_state values '
                "(1, (%s::date + interval '3 months')::date)",
                (first,),
            )
        if not (MODELS / 'latest.txt').exists():
            initial = MODELS / 'initial'
            initial.mkdir(exist_ok=True)
            first = connection.execute(
                'select min(data_month) as month from labeled_transactions'
            ).fetchone()['month']
            rows = connection.execute(
                'select * from labeled_transactions '
                'where transaction_time >= %s '
                "and transaction_time < %s::date + interval '2 months' "
                'order by transaction_time, source_row',
                (first, first),
            ).fetchall()
            train = pd.DataFrame(rows)
            state = fit_state(train)
            model = train_model(preprocess(train, state), train['target'])
            model.save_model(initial / 'model.cbm')
            write_json(initial / 'preprocessing.json', state)
            temporary = MODELS / '.latest.tmp'
            temporary.write_text('initial\n')
            os.replace(temporary, MODELS / 'latest.txt')
    print('Dataset and initial model ready')


if __name__ == '__main__':
    main()
