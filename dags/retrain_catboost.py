from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.retrain import (
    compare_models,
    extract_window,
    fit_catboost,
    preprocess_dataset,
    publish_model,
    time_train_test_split,
)


@dag(
    schedule='*/5 * * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={'retries': 1, 'retry_delay': timedelta(seconds=15)},
    tags=['mlops'],
)
def retrain_catboost() -> None:
    extracted = task(extract_window)()
    split = task(time_train_test_split)(extracted)
    processed = task(preprocess_dataset)(split)
    trained = task(fit_catboost)(processed)
    compared = task(compare_models)(trained)
    task(publish_model)(compared)


retrain_catboost()
