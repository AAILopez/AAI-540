# evaluate.py

import argparse
import os
import tarfile
import json

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        dest="model_dir",
        required=True,
        help="Path to the directory containing the trained XGBoost model.tar.gz"
    )
    parser.add_argument(
        "--batch-data",
        type=str,
        dest="batch_dir",
        required=True,
        help="Path to the directory containing batch.csv (test split)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        dest="output_dir",
        required=True,
        help="Directory where metrics.json will be written"
    )
    args = parser.parse_args()

    # 1) Extract the XGBoost model
    model_tar_path = os.path.join(args.model_dir, "model.tar.gz")
    with tarfile.open(model_tar_path) as tar:
        tar.extractall(path=args.model_dir)

    # The extracted file will be named 'xgboost-model'
    model_path = os.path.join(args.model_dir, "xgboost-model")
    booster = xgb.Booster()
    booster.load_model(model_path)

    # 2) Load the batch/test split
    batch_df = pd.read_csv(os.path.join(args.batch_dir, "batch.csv"))
    X_test = batch_df.drop("need_maintenance", axis=1)
    y_true = batch_df["need_maintenance"]

    # 3) Make predictions
    dmatrix = xgb.DMatrix(X_test)
    y_proba = booster.predict(dmatrix)
    y_pred = (y_proba >= 0.5).astype(int)

    # 4) Compute evaluation metrics
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_proba)
    }

    # 5) Write metrics.json
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics_cicd.json"), "w") as f:
        json.dump(metrics, f)

    print("Evaluation metrics:", metrics)
