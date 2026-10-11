
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/opt/airflow/project")


def preprocess():
    subprocess.run(
        [sys.executable, str(PROJECT / "src" / "preprocessing.py")],
        cwd=PROJECT,
        check=True,
    )


def train_model():
    subprocess.run(
        [sys.executable, str(PROJECT / "src" / "clusteringclassification.py")],
        cwd=PROJECT,
        check=True,
    )


with DAG(
    dag_id="retrain_pipeline",
    start_date=datetime(2026, 10, 10),
    schedule="@weekly",
    catchup=False,
) as dag:

    task1 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess,
    )

    task2 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    task1 >> task2
