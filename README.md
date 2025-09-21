Scalable Batch Labeling MLOps Pipeline with MLflow
Overview
A scalable MLOps pipeline for batch image labeling, integrating Airflow, MLflow (in Docker/Kubernetes), Docker, Kubernetes (Minikube), and GitHub Actions. Features model drift detection and automated rollback.
Prerequisites

Docker
Minikube
kubectl
Python 3.9
Git

Setup

Clone the repository:git clone <repository_url>
cd project


Start Minikube:minikube start


Build Docker images:docker build -t batch-labeling:latest -f docker/inference/Dockerfile .
docker build -t mlflow-server:latest -f docker/mlflow/Dockerfile .


Run Airflow:docker-compose -f docker-compose.yml up



Running the Pipeline

Place images in data/images/.
Log model to MLflow (if not done):python -c "import mlflow; with mlflow.start_run(): mlflow.log_artifact('data/model/model.pt'); mlflow.log_artifact('data/model/class_mapping.json'); mlflow.log_artifact('data/model/transforms.json')"


Trigger the DAG via Airflow UI (http://localhost:8080, credentials: airflow/airflow) or CLI:airflow dags trigger batch_labeling_pipeline


Check data/output/predictions.json for results.
Access MLflow UI at http://<minikube-ip>:5000.

CI/CD

Automated via GitHub Actions on push to main.

Testing
Run unit tests:
pytest tests/

Monitoring

Prometheus can be integrated for Kubernetes monitoring (not included).

Cleanup
Stop Minikube:
minikube stop
