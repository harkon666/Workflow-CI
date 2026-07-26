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


def main():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Heart_Disease_CI_Bryan")

    dataset_path = "heart_processed.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "../heart_processed.csv"

    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="CI_Retrain_RandomForest"):
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
