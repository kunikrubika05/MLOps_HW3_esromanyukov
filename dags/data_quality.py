from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.quality import check_dataset, publish_report


@dag(
    schedule='*/5 * * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={'retries': 1, 'retry_delay': timedelta(seconds=15)},
    tags=['mlops'],
)
def data_quality() -> None:
    checked = task(check_dataset)()
    task(publish_report)(checked)


data_quality()
