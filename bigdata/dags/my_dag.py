# my_dag.py

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

default_args = {
    'owner': 'bigdata',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'my_dag',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
)

def my_python_function(**kwargs):
    # Replace this with your Python code for the task
    print("Hello from my Python function!")

task1 = PythonOperator(
    task_id='task_1',
    python_callable=my_python_function,
    provide_context=True,
    dag=dag,
)

