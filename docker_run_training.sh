#!/bin/bash
# exit when any command fails
set -e
dvc pull
python3.9 -u src/models/train_model.py
gsutil cp -r outputs gs://dtu-mlops-project-training-output
