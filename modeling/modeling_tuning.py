import os
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
 
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from dotenv import load_dotenv 
load_dotenv()


# Configuration
DATA_PATH = "modeling/bank_transactions_preprocessing.csv"
EXPERIMENT_NAME = "fraud-detection-tuning"
RANDOM_STATE = 42
PARAM_GRID = {
    "n_estimators": [100, 200],
    "contamination": [0.03, 0.05, 0.08],
    "max_samples": ["auto", 0.8],
}

# MLflow setup
def mlflow_setup():
    """
    Setup MLflow tracking URI
    """
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")
    dagshub_username = os.environ.get("DAGSHUB_USERNAME")
    dagshub_repo = os.environ.get("DAGSHUB_REPO")

    if dagshub_token and dagshub_username and dagshub_repo:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        tracking_uri = f"https://dagshub.com/{dagshub_username}/{dagshub_repo}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI set to: {tracking_uri}")
    else:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        print("MLflow setup locally completed")

    mlflow.set_experiment(EXPERIMENT_NAME)


# Load data
def load_data(data_path=DATA_PATH):
    """
    Load preprocessed CSV and scale features
    """
    df = pd.read_csv(data_path)
    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(df)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=df.columns)

    print(f"Data loaded and scaled. Shape: {df.shape}")
    return df, scaler, X_scaled


# Artifacts
def plot_anomaly_score(anomaly_scores, labels, run_tag, save_dir="models"):
    """
    Plot and save anomaly score distribution
    """
    os.makedirs(f"{save_dir}/isolation_forest_tuned", exist_ok=True)
    save_path = f"{save_dir}/isolation_forest_tuned/anomaly_score_{run_tag}.png"
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(anomaly_scores[labels == 0], bins=60, alpha=0.6, color="steelblue", label="Normal")
    ax.hist(anomaly_scores[labels == 1], bins=60, alpha=0.6, color="crimson", label="Fraud")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Decision Boundary")
    ax.set_title(f"Anomaly Score Distribution [{run_tag}]")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def plot_fraud_proportion(labels, run_tag, save_dir="models"):
    """
    Plot fraud and normal proportion
    """
    os.makedirs(f"{save_dir}/isolation_forest_tuned", exist_ok=True)
    save_path = f"{save_dir}/isolation_forest_tuned/fraud_proportion_{run_tag}.png"
    counts = pd.Series(labels).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Normal", "Fraud"], counts.values, color=["blue", "red"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200, f"{val:,}\n({val/len(labels):.1%})", ha="center", fontsize=10)
    ax.set_title(f"Fraud vs Normal [{run_tag}]")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def plot_feature_importance(df, anomaly_scores, run_tag, save_dir="models"):
    """
    Plot feature importance via correlation with anomaly score
    """
    os.makedirs(f"{save_dir}/isolation_forest_tuned", exist_ok=True)
    save_path = f"{save_dir}/isolation_forest_tuned/feature_importance_{run_tag}.png"

    correlations = [
        abs(np.corrcoef(df[col].values, anomaly_scores)[0, 1])
        for col in df.columns
    ]
    importance_df = pd.DataFrame({
        "Feature": df.columns.tolist(),
        "Importance": correlations
    }).sort_values("Importance", ascending=True)
 
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="blue", edgecolor="white")
    ax.set_title(f"Feature Importance [{run_tag}]")
    ax.set_xlabel("Absolute Correlation")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def plot_pca_visualization(X_scaled, labels, run_tag, save_dir="models"):
    """
    Plot PCA 2D scatter of fraud and normal
    """
    os.makedirs(f"{save_dir}/isolation_forest_tuned", exist_ok=True)
    save_path = f"{save_dir}/isolation_forest_tuned/pca_visualization_{run_tag}.png"

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="coolwarm", alpha=0.4, s=5)
    legend = ax.legend(*scatter.legend_elements(), labels=["Normal", "Fraud"], title="Label")
    ax.add_artist(legend)
    ax.set_title(f"Fraud vs Normal [{run_tag}]")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def model_train_tuning(df, X_scaled, params, run_tag):
    """
    Train Isolation Forest with manual MLflow tracking
    """
    with mlflow.start_run(run_name=f"{run_tag}") as run:
        # Train Model
        model = IsolationForest(
            n_estimators = params["n_estimators"],
            contamination = params["contamination"],
            max_samples = params["max_samples"],
            random_state = RANDOM_STATE,
        )
        model.fit(X_scaled)

        # Predict
        predictions = model.predict(X_scaled)
        anomaly_scores = model.decision_function(X_scaled)
        labels = (predictions == -1).astype(int)
        n_fraud = int(labels.sum())

        sample_idx = np.random.choice(len(X_scaled), size=min(5000, len(X_scaled)), replace=False)
        sil_score = silhouette_score(X_scaled.iloc[sample_idx], labels[sample_idx])

        # Log paramaters
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("contamination", params["contamination"])
        mlflow.log_param("max_samples", params["max_samples"])
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features", df.shape[1])

        # Log metrics
        mlflow.log_metric("n_anomalies", n_fraud)
        mlflow.log_metric("n_normal", int((labels == 0).sum()))
        mlflow.log_metric("anomaly_rate", round(n_fraud / len(labels), 4))
        mlflow.log_metric("silhouette_score", round(float(sil_score), 4))
        mlflow.log_metric("mean_anomaly_score", round(float(np.mean(anomaly_scores)), 4))
        mlflow.log_metric("std_anomaly_score", round(float(np.std(anomaly_scores)), 4))
        mlflow.log_metric("min_anomaly_score", round(float(np.min(anomaly_scores)), 4))
        mlflow.log_metric("max_anomaly_score", round(float(np.max(anomaly_scores)), 4))

        # Artifacts
        mlflow.log_artifact(plot_anomaly_score(anomaly_scores, labels, run_tag))
        mlflow.log_artifact(plot_fraud_proportion(labels, run_tag))
        mlflow.log_artifact(plot_feature_importance(df, anomaly_scores, run_tag))
        mlflow.log_artifact(plot_pca_visualization(X_scaled, labels, run_tag))
 
        input_example = X_scaled.head()
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model", 
            input_example=input_example,
            signature=infer_signature(X_scaled, model.predict(X_scaled)),
        )

        metrics = {
            "n_anomalies": n_fraud,
            "n_normal": int((labels == 0).sum()),
            "anomaly_rate": round(n_fraud / len(labels), 4),
            "silhouette_score": round(float(sil_score), 4),
            "mean_anomaly_score": round(float(np.mean(anomaly_scores)), 4),
            "std_anomaly_score": round(float(np.std(anomaly_scores)), 4),
            "min_anomaly_score": round(float(np.min(anomaly_scores)), 4),
            "max_anomaly_score": round(float(np.max(anomaly_scores)), 4),
        }

        return run.info.run_id, metrics, float(sil_score)
    

def main():
    try:
        mlflow_setup()
        
        df, scaler, X_scaled = load_data(DATA_PATH)
        keys = list(PARAM_GRID.keys())
        combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
        params_list = [dict(zip(keys, combo)) for combo in combos]
 
        results = {}
        best_score = -999
        best_tag = None
 
        for params in params_list:
            tag = f"est{params['n_estimators']}_cont{params['contamination']}_samp{params['max_samples']}"
            run_id, metrics, sil_score = model_train_tuning(df, X_scaled, params, run_tag=tag)
            results[tag] = {"run_id": run_id, "metrics": metrics, "silhouette": sil_score}
            if sil_score > best_score:
                best_score = sil_score
                best_tag = tag
 
        print(f"Best run: {best_tag}")
        print(f"Best silhouette: {best_score:.4f}")
        print(f"Best run ID: {results[best_tag]['run_id']}")
 
        os.makedirs("models", exist_ok=True)
        with open("models/model_run_id.txt", "w") as f:
            f.write(results[best_tag]["run_id"])
        print("Best run ID saved to models/model_run_id.txt")
 
        # Register Best Model to Model Registry
        best_run_id = results[best_tag]["run_id"]
        model_uri = f"runs:/{best_run_id}/model"
        registered = mlflow.register_model(
            model_uri=model_uri,
            name="fraud-detection-tuned",
        )
 
        # Set alias for best tuned model
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name="fraud-detection-tuned",
            alias="champion",
            version=registered.version
        )
 
        # Load model via alias to verify registration
        loaded_model = mlflow.pyfunc.load_model(
            f"models:/fraud-detection-tuned@champion"
        )
        sample_pred = loaded_model.predict(X_scaled.head())

        print(f"Sample predictions: {sample_pred}")
        return results
    except Exception as e:
        print(f"An error occurred during tuning: {e}")
        raise
 
 
if __name__ == "__main__":
    main()