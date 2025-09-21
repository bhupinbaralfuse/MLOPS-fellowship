
from datetime import datetime
from utils.preprocess import preprocess_and_infer
from utils.validation import validate_and_report

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 7, 11),
    'retries': 2,
}

dag = DAG(
    'batch_labeling_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)

# Task 1: Deploy infrastructure (MLflow and inference)
deploy_infra = BashOperator(
    task_id='deploy_infrastructure',
    bash_command='kubectl apply -f kubernetes/mlflow/ && kubectl apply -f kubernetes/inference/',
    dag=dag,
)

# Task 2: Preprocess and infer
process_infer = PythonOperator(
    task_id='preprocess_and_infer',
    python_callable=preprocess_and_infer,
    op_kwargs={
        'input_dir': 'data/images',
        'output_file': 'data/output/predictions.json',
        'mlflow_tracking_uri': 'http://mlflow-service:5000',
    },
    dag=dag,
)

# Task 3: Validate and report
validate = PythonOperator(
    task_id='validate_and_report',
    python_callable=validate_and_report,
    op_kwargs={
        'predictions_file': 'data/output/predictions.json',
        'sample_size': 0.1,
        'mlflow_tracking_uri': 'http://mlflow-service:5000',
    },
    dag=dag,
)

# Task 4: Teardown infrastructure
teardown_infra = BashOperator(
    task_id='teardown_infrastructure',
    bash_command='kubectl delete -f kubernetes/mlflow/ && kubectl delete -f kubernetes/inference/',
    dag=dag,
)

# Dependencies
deploy_infra >> process_infer >> validate >> teardown_infra