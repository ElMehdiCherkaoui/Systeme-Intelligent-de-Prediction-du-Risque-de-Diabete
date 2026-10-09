
import importlib.metadata
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from preprocessing import load_clean_and_scale
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import mlflow
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import shutil

scaler = StandardScaler()

df_scaled = load_clean_and_scale(
    "data/raw/dataset-diabete.csv",
    "models/scaler.joblib"
)

df_cleaned = pd.read_csv("data/processed/df_cleaned.csv")

silhouette_scores = []
inertia = []

k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    labels = kmeans.labels_
    score = silhouette_score(df_scaled, labels)
    silhouette_scores.append(score)
    inertia.append(kmeans.inertia_)

best_index = silhouette_scores.index(max(silhouette_scores))
best_k = list(k_range)[best_index]

kmeans_final = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans_final.fit_predict(df_scaled)

df_scaled = df_scaled.copy()

df_scaled["Cluster"] = cluster_labels

df_cleaned["Cluster"] = cluster_labels

cluster_summary = df_cleaned.copy()
cluster_summary["Cluster"] = cluster_labels
cluster_summary1 = cluster_summary.groupby("Cluster").mean(numeric_only=True)

cluster_counts = df_scaled["Cluster"].value_counts()

cluster_haut_risk = cluster_summary1[
    (cluster_summary1["Glucose"] > 126) &
    (cluster_summary1["BMI"] > 30) &
    (cluster_summary1["DiabetesPedigreeFunction"] > 0.5)
]

list_clusters_haut_risk = cluster_haut_risk.index.tolist()

cluster_summary["risk_category"] = np.where(
    cluster_summary["Cluster"].isin(list_clusters_haut_risk),
    1,
    0
)

y = cluster_summary["risk_category"]

X = df_scaled.copy()
X = X.drop("Cluster", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.2
)

cluster_total = cluster_summary["risk_category"].value_counts()
high_risk_percent = (cluster_total.get(1, 0) / cluster_total.sum()) * 100
low_risk_percent = (cluster_total.get(0, 0) / cluster_total.sum()) * 100

check_balance = (
    "balanced"
    if abs(high_risk_percent - low_risk_percent) <= 10
    else "imbalanced"
)

trained_models = {}

lg = LogisticRegression(random_state=42)
rf = RandomForestClassifier(random_state=42)
svc = SVC(random_state=42)

lg.fit(X_train, y_train)
rf.fit(X_train, y_train)
svc.fit(X_train, y_train)

trained_models["Logistic Regression"] = lg
trained_models["Random Forest"] = rf
trained_models["Support Vector Classifier"] = svc

print("Models trained successfully!")

final_results = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    prec = (
        cm[1, 1] / (cm[1, 1] + cm[0, 1])
        if (cm[1, 1] + cm[0, 1]) != 0
        else 0.0
    )

    rec = (
        cm[1, 1] / (cm[1, 1] + cm[1, 0])
        if (cm[1, 1] + cm[1, 0]) != 0
        else 0.0
    )

    f1 = (
        2 * (prec * rec) / (prec + rec)
        if (prec + rec) != 0
        else 0.0
    )

    final_results[name] = {
        "confusion_matrix": cm,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }


param_grid_lg = {
    "C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
    "penalty": ["l1", "l2"],
    "max_iter": [100, 200, 300]
}

param_grid_rf = {
    "n_estimators": [25, 50, 75, 100, 150, 200, 250, 300],
    "max_depth": [None, 10, 15, 20, 25, 30],
    "min_samples_split": [2, 3, 5, 7, 10],
    "min_samples_leaf": [1, 2, 4, 6, 8],
}

param_grid_svc = {
    "C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": [2, 3, 4]
}

random_search_lg = RandomizedSearchCV(
    estimator=lg,
    n_iter=20,
    param_distributions=param_grid_lg,
    cv=5,
    random_state=42
)

random_search_rf = RandomizedSearchCV(
    estimator=rf,
    n_iter=20,
    param_distributions=param_grid_rf,
    cv=5,
    random_state=42
)

random_search_svc = RandomizedSearchCV(
    estimator=svc,
    n_iter=20,
    param_distributions=param_grid_svc,
    cv=5,
    random_state=42
)

random_search_lg.fit(X_train, y_train)
random_search_rf.fit(X_train, y_train)
random_search_svc.fit(X_train, y_train)

y_pred_optimized_lg = random_search_lg.predict(X_test)
y_pred_optimized_rf = random_search_rf.predict(X_test)
y_pred_optimized_svc = random_search_svc.predict(X_test)


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    prec = (
        cm[1, 1] / (cm[1, 1] + cm[0, 1])
        if (cm[1, 1] + cm[0, 1]) != 0
        else 0.0
    )

    rec = (
        cm[1, 1] / (cm[1, 1] + cm[1, 0])
        if (cm[1, 1] + cm[1, 0]) != 0
        else 0.0
    )

    f1 = (
        2 * (prec * rec) / (prec + rec)
        if (prec + rec) != 0
        else 0.0
    )

    return cm, prec, rec, f1


cm, prec, rec, f1 = calculate_metrics(
    y_test, y_pred_optimized_lg
)

cm_rf, prec_rf, rec_rf, f1_rf = calculate_metrics(
    y_test, y_pred_optimized_rf
)

cm_svc, prec_svc, rec_svc, f1_svc = calculate_metrics(
    y_test, y_pred_optimized_svc
)

final_results["logistic_regression best parameters"] = {
    "confusion_matrix": cm,
    "Precision": prec,
    "Recall": rec,
    "F1 Score": f1
}

final_results["random_forest best parameters"] = {
    "confusion_matrix": cm_rf,
    "Precision": prec_rf,
    "Recall": rec_rf,
    "F1 Score": f1_rf
}

final_results["support_vector_machine best parameters"] = {
    "confusion_matrix": cm_svc,
    "Precision": prec_svc,
    "Recall": rec_svc,
    "F1 Score": f1_svc
}


if (
    final_results["logistic_regression best parameters"]["F1 Score"]
    >= final_results["random_forest best parameters"]["F1 Score"]
    and
    final_results["logistic_regression best parameters"]["F1 Score"]
    >= final_results["support_vector_machine best parameters"]["F1 Score"]
):
    best_model = "logistic_regression best parameters"

elif (
    final_results["random_forest best parameters"]["F1 Score"]
    >= final_results["support_vector_machine best parameters"]["F1 Score"]
):
    best_model = "random_forest best parameters"

else:
    best_model = "support_vector_machine best parameters"

print(
    f"The best model is: {best_model} with F1 Score: "
    f"{final_results[best_model]['F1 Score']:.4f}"
)


if best_model == "logistic_regression best parameters":
    best_estimator = random_search_lg.best_estimator_

elif best_model == "random_forest best parameters":
    best_estimator = random_search_rf.best_estimator_

else:
    best_estimator = random_search_svc.best_estimator_


os.makedirs("models", exist_ok=True)
joblib.dump(best_estimator, "models/best_model.joblib")

mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Diabetes_Risk_Prediction"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="Best_Model_Training"):
    mlflow.log_param("Best Model", best_model)

    clustering_inertia = kmeans_final.inertia_
    mlflow.log_metric("Clustering Inertia", float(clustering_inertia))
    mlflow.log_metric(
        "Silhouette Score",
        float(silhouette_scores[best_index])
    )

    os.makedirs("artifacts", exist_ok=True)

    model_path = "artifacts/best_model.joblib"
    scaler_path = "artifacts/scaler.joblib"

    joblib.dump(best_estimator, model_path)


    if os.path.exists("models/scaler.joblib"):
        joblib.copy if False else None

        shutil.copyfile("models/scaler.joblib", scaler_path)


with mlflow.start_run(run_name="Classification_run_best_model"):
    mlflow.log_param("Best Model", best_model)

    # Log parameters individually to avoid MLflow parameter-value
    # restrictions for complex dictionaries.
    for param_name, param_value in best_estimator.get_params().items():
        if isinstance(param_value, (str, int, float, bool)) or param_value is None:
            mlflow.log_param(
                f"model_{param_name}",
                str(param_value)
            )

    # Evaluation results are metrics, not parameters.
    mlflow.log_metric(
        "Precision",
        float(final_results[best_model]["Precision"])
    )
    mlflow.log_metric(
        "Recall",
        float(final_results[best_model]["Recall"])
    )
    mlflow.log_metric(
        "F1 Score",
        float(final_results[best_model]["F1 Score"])
    )

    y_pred = best_estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {best_model}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    os.makedirs("artifacts", exist_ok=True)
    cm_image_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_image_path)
    plt.close()

    mlflow.log_artifact(
        cm_image_path,
        artifact_path="confusion_matrix"
    )

    feature_names = X.columns.tolist()
    mlflow.log_text(str(feature_names), "feature_names.txt")
    mlflow.log_param("feature_names_count", len(feature_names))

    version_sl = importlib.metadata.version("scikit-learn")
    mlflow.log_param("scikit-learn_version", version_sl)

    version_mlflow = importlib.metadata.version("mlflow")
    mlflow.log_param("mlflow_version", version_mlflow)

    version_joblib = importlib.metadata.version("joblib")
    mlflow.log_param("joblib_version", version_joblib)

    version_numpy = importlib.metadata.version("numpy")
    mlflow.log_param("numpy_version", version_numpy)

    version_pandas = importlib.metadata.version("pandas")
    mlflow.log_param("pandas_version", version_pandas)

    input_exemple = X_test.iloc[:3]
    predection_exemple = best_estimator.predict(input_exemple)

    signature = infer_signature(
        input_exemple,
        predection_exemple
    )

    model_info = mlflow.sklearn.log_model(
        sk_model=best_estimator,
        artifact_path="my_final_model",
        signature=signature,
        input_example=input_exemple,
        registered_model_name="Diabetes_Risk_Classifier"
    )

client = MlflowClient(tracking_uri="http://localhost:5000")

model_name = "Diabetes_Risk_Classifier"

registered_version = getattr(
    model_info,
    "registered_model_version",
    None
)

if registered_version is None:
    versions = client.search_model_versions(
        f"name = '{model_name}'"
    )

    if not versions:
        raise RuntimeError(
            f"No registered versions found for {model_name}"
        )

    registered_version = max(
        versions,
        key=lambda version: int(version.version)
    ).version

client.transition_model_version_stage(
    name=model_name,
    version=str(registered_version),
    stage="Production",
    archive_existing_versions=True
)

print(
    f"Model version {registered_version} of "
    f"'{model_name}' transitioned to Production stage."
)