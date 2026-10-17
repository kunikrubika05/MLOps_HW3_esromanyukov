FROM apache/airflow:2.10.5-python3.11
USER airflow
RUN pip install --no-cache-dir \
    'apache-airflow==2.10.5' \
    'numpy==1.26.4' \
    'pandas==2.1.4' \
    'scipy==1.14.1' \
    'scikit-learn==1.5.2' \
    'catboost==1.2.8' \
    'cloudpickle==2.2.1' \
    'psycopg[binary]==3.2.9' \
    'ruff==0.17.0' \
    && python -c 'import airflow, catboost, sklearn, pandas, pyarrow, psycopg; print(airflow.__version__)'
USER root
COPY entrypoint.sh /custom-entrypoint.sh
RUN chmod +x /custom-entrypoint.sh
ENTRYPOINT ["/custom-entrypoint.sh"]
