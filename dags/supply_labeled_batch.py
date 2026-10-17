from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.supply import append_labeled_batch, prune_old_months, sample_next_month


@dag(
    schedule='*/5 * * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={'retries': 1, 'retry_delay': timedelta(seconds=15)},
    tags=['mlops'],
)
def supply_labeled_batch() -> None:
    sampled = task(sample_next_month)()
    appended = task(append_labeled_batch)(sampled)
    task(prune_old_months)(appended)


supply_labeled_batch()
