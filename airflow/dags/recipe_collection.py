import os
import requests

from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

from lib.page_crawling import crawl_incremental

default_args = {
    "owner": "airflow",
    "start_date": datetime(2026, 1, 18),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="recipe_collection",
    default_args=default_args,
    schedule_interval=timedelta(days=7),
    max_active_runs=1,
    catchup=True,
    tags=["ai-tech"],
) as dag:
    save_mongo_db = PythonOperator(
        task_id="recipe_crawling",
        python_callable=crawl_incremental,
    )

    save_mongo_db
