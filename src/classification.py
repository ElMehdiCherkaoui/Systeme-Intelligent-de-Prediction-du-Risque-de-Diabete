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
from sklearn.metrics import  confusion_matrix
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df_scaled = load_clean_and_scale("data/raw/dataset-diabete.csv", "models/scaler.joblib")
df_cleaned = pd.read_csv("data/processed/df_cleaned.csv")

silhouette_scores = []
inertia = []

k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    labels = kmeans.labels_
    score = silhouette_score(df_scaled, labels)
    silhouette_scores.append(score)
    inertia.append(kmeans.inertia_)
    
best_index = silhouette_scores.index(max(silhouette_scores))

best_k = k_range[best_index]

kmeans_final = KMeans(n_clusters=best_k, random_state=42)

cluster_labels = kmeans_final.fit_predict(df_scaled)

df_scaled['Cluster'] = cluster_labels

df_cleaned['Cluster'] = cluster_labels

cluster_summary = df_cleaned.copy()
cluster_summary["Cluster"] = cluster_labels
cluster_summary1 = cluster_summary.groupby("Cluster").mean()

cluster_counts = df_scaled["Cluster"].value_counts()

cluster_haut_risk = cluster_summary1[
    (cluster_summary1['Glucose'] > 126) & 
    (cluster_summary1['BMI'] > 30) &
    (cluster_summary1['DiabetesPedigreeFunction'] > 0.5)
]
list_clusters_haut_risk = cluster_haut_risk.index.tolist()


cluster_summary['risk_category'] = np.where(
    cluster_summary['Cluster'].isin(list_clusters_haut_risk),
    1, 
    0   
)

y = cluster_summary['risk_category']
X = df_scaled.copy()
X = X.drop('Cluster', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)

cluster_total = cluster_summary['risk_category'].value_counts()
high_risk_percent = (cluster_total.get(1, 0) / cluster_total.sum()) * 100
low_risk_percent = (cluster_total.get(0, 0) / cluster_total.sum()) * 100

check_balance = "balanced" if abs(high_risk_percent - low_risk_percent) <= 10 else "imbalanced"

trained_models = {}

lg = LogisticRegression(random_state=42)
rf = RandomForestClassifier(random_state=42)
svc = SVC(random_state=42)

lg.fit(X_train, y_train)
rf.fit(X_train, y_train)
svc.fit(X_train, y_train)

trained_models['Logistic Regression'] = lg
trained_models['Random Forest'] = rf
trained_models['Support Vector Classifier'] = svc

print("Models trained successfully!")

final_results = {}

for name,model in trained_models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    prec = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    rec = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    f1 = 2 * (prec * rec) / (prec + rec)

    
    final_results[name] = {
        'confusion_matrix': cm,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    }


param_grid_lg = {
    'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
    'penalty': ['l1', 'l2'],
    'max_iter': [100, 200, 300]
}  
param_grid_rf = {
    'n_estimators': [25,50, 75, 100, 150, 200, 250, 300],
    'max_depth': [None, 10, 15, 20, 25, 30],
    'min_samples_split': [2, 3, 5, 7, 10],
    'min_samples_leaf': [1, 2, 4, 6, 8],
}
param_grid_svc = {
    'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4]
}

random_search_lg = RandomizedSearchCV(estimator=lg, n_iter=20, param_distributions=param_grid_lg, cv=5, random_state=42)
random_search_rf = RandomizedSearchCV(estimator=rf, n_iter=20, param_distributions=param_grid_rf, cv=5, random_state=42)
random_search_svc = RandomizedSearchCV(estimator=svc, n_iter=20, param_distributions=param_grid_svc, cv=5, random_state=42)

random_search_lg.fit(X_train, y_train)
random_search_rf.fit(X_train, y_train)
random_search_svc.fit(X_train, y_train)

y_pred_optimized_lg = random_search_lg.predict(X_test)
y_pred_optimized_rf = random_search_rf.predict(X_test)
y_pred_optimized_svc = random_search_svc.predict(X_test)



cm = confusion_matrix(y_test, y_pred_optimized_lg)
prec = cm[1, 1] / (cm[1, 1] + cm[0, 1])
rec = cm[1, 1] / (cm[1, 1] + cm[1, 0])
f1 = 2 * (prec * rec) / (prec + rec)

cm_rf = confusion_matrix(y_test, y_pred_optimized_rf)
prec_rf = cm_rf[1, 1] / (cm_rf[1, 1] + cm_rf[0, 1])
rec_rf = cm_rf[1, 1] / (cm_rf[1, 1] + cm_rf[1, 0])
f1_rf = 2 * (prec_rf * rec_rf) / (prec_rf + rec_rf)

cm_svc = confusion_matrix(y_test, y_pred_optimized_svc)
prec_svc = cm_svc[1, 1] / (cm_svc[1, 1] + cm_svc[0, 1])
rec_svc = cm_svc[1, 1] / (cm_svc[1, 1] + cm_svc[1, 0])
f1_svc = 2 * (prec_svc * rec_svc) / (prec_svc + rec_svc)

final_results['logistic_regression best parameters'] = {
    'confusion_matrix': cm,
    'Precision': prec,
    'Recall': rec,
    'F1 Score': f1
}

final_results['random_forest best parameters'] = {
    'confusion_matrix': cm_rf,
    'Precision': prec_rf,
    'Recall': rec_rf,
    'F1 Score': f1_rf
}

final_results['support_vector_machine best parameters'] = {
    'confusion_matrix': cm_svc,
    'Precision': prec_svc,
    'Recall': rec_svc,
    'F1 Score': f1_svc
}





best_model = final_results['logistic_regression best parameters']['F1 Score'] >= final_results['random_forest best parameters']['F1 Score'] and final_results['logistic_regression best parameters']['F1 Score'] >= final_results['support_vector_machine best parameters']['F1 Score']
if best_model:  
    best_model = 'logistic_regression best parameters'
elif final_results['random_forest best parameters']['F1 Score'] >= final_results['support_vector_machine best parameters']['F1 Score']:
    best_model = 'random_forest best parameters'
else:
    best_model = 'support_vector_machine best parameters'

print(f"The best model is: {best_model} with F1 Score: {final_results[best_model]['F1 Score']:.4f}")



if best_model == 'logistic_regression best parameters':
    best_estimator = random_search_lg.best_estimator_

elif best_model == 'random_forest best parameters':
    best_estimator = random_search_rf.best_estimator_

else:
    best_estimator = random_search_svc.best_estimator_

joblib.dump(best_estimator, 'models/best_model.joblib')