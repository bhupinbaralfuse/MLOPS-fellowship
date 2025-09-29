FROM python:3.9-slim

RUN pip install mlflow boto3

EXPOSE 5000

WORKDIR /mlflow

ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=minio123

CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "s3://mlflow/", "--host", "0.0.0.0", "--port", "5000"]