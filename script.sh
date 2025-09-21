#!/bin/bash


BASE_DIR="."

git init 


mkdir -p $BASE_DIR/dags/utils
mkdir -p $BASE_DIR/docker
mkdir -p $BASE_DIR/tests
mkdir -p $BASE_DIR/.github/workflows
mkdir -p $BASE_DIR/kubernetes
mkdir -p $BASE_DIR/data/images
mkdir -p $BASE_DIR/data/model
mkdir -p $BASE_DIR/data/output

touch $BASE_DIR/dags/ml_pipeline.py
touch $BASE_DIR/dags/utils/__init__.py
touch $BASE_DIR/dags/utils/preprocess.py
touch $BASE_DIR/dags/utils/inference.py
touch $BASE_DIR/dags/utils/validation.py

touch $BASE_DIR/docker/Dockerfile
touch $BASE_DIR/docker/requirements.txt

touch $BASE_DIR/tests/test_preprocess.py

touch $BASE_DIR/.github/workflows/ci_cd.yml

touch $BASE_DIR/kubernetes/deployment.yaml
touch $BASE_DIR/kubernetes/service.yaml
touch $BASE_DIR/kubernetes/hpa.yaml

mv model.pt $BASE_DIR/data/model/model.pt
mv class_mapping.json $BASE_DIR/data/model/class_mapping.json
mv  transforms.json $BASE_DIR/data/model/transforms.json 


touch $BASE_DIR/data/output/predictions.json

touch $BASE_DIR/README.md
touch $BASE_DIR/docker-compose.yml

echo "Project structure created under '$BASE_DIR'."

