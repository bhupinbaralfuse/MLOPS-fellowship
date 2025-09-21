FROM python:3.10-slim

LABEL MAINTAINER bhupin.baral.729@gmail.com

WORKDIR /app

RUN apt update && apt upgrade -y && apt install -y sqlite3

RUN pip install --upgrade pip && pip install --no-cache-dir "mlflow==2.9.2"

ENV MLFLOW_PORT=9000
ENV MLRUNS_DIR=/mlruns

RUN mkdir -p $MLRUNS_DIR && chmod -R 777 $MLRUNS_DIR

EXPOSE $MLFLOW_PORT

CMD mlflow server \
    --backend-store-uri sqlite:///$MLRUNS_DIR/mlflow.db \
    --default-artifact-root file://$MLRUNS_DIR \
    -h 0.0.0.0 -p $MLFLOW_PORT
