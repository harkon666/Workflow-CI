"""
MLProject Retraining Script
Nama Siswa: Bryan Dewa Wicaksana
"""

import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def find_dataset(filename="heart_processed.csv"):
    possible_paths = [
        filename,
        os.path.join("dataset_preprocessing", filename),
        os.path.join("..", "dataset_preprocessing", filename),
        os.path.join("..", filename)
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Processed dataset '{filename}' tidak ditemukan di lokasi: {possible_paths}")


def main():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    # Jika dipanggil via `mlflow run`, mlruns berada di parent directory
    if "MLFLOW_RUN_ID" in os.environ and os.path.exists("../mlruns"):
        mlflow.set_tracking_uri("file:../mlruns")
    else:
        mlflow.set_tracking_uri("file:./mlruns")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()

    dataset_path = find_dataset("heart_processed.csv")
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"[CI RETRAIN SUCCESS] Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")


if __name__ == "__main__":
    main()
