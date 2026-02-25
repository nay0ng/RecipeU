import os
import requests

from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

from lib.ranking_crawling import process_ranking_to_recipes


default_args = {
    "owner": "airflow",
    "start_date": datetime(2026, 2, 2),
}

with DAG(
    dag_id="recipe_ranking",
    default_args=default_args,
    schedule_interval="0 22 * * *",
    max_active_runs=1,
    catchup=False,
    tags=["ai-tech"],
) as dag:

    ranking_recipes = PythonOperator(
        task_id="recipe_ranking",
        python_callable=process_ranking_to_recipes,
    )

    ranking_recipes
