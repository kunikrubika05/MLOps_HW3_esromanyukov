#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/opt/airflow:/opt/airflow/project
export AIRFLOW__CORE__EXECUTOR=SequentialExecutor
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////runtime/airflow/airflow.db
unset _AIRFLOW_DB_MIGRATE _AIRFLOW_WWW_USER_CREATE
run_airflow() {
  /usr/sbin/runuser --preserve-environment --user airflow -- "$@"
}
run_airflow airflow db migrate
if ! run_airflow airflow users list --output json | grep -q '"username": "admin"'; then
  run_airflow airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
fi
exec /usr/sbin/runuser --preserve-environment --user airflow -- /entrypoint standalone
