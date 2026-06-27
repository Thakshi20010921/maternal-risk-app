"""
train_model.py
----------------
Trains and compares 5 classification models (Logistic Regression, Decision Tree,
Random Forest, KNN, SVM) on the maternal health risk dataset, then saves the
best-performing model + scaler + metadata for use in the Streamlit app.

Run this once locally (or it will auto-run on first Streamlit Cloud deploy)
to generate models/best_model.pkl and models/metrics.json.
"""

import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "data/maternal_health_data.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "Age",
    "Systolic BP",
    "Diastolic",
    "BS",
    "Body Temp",
    "BMI",
    "Previous Complications",
    "Preexisting Diabetes",
    "Gestational Diabetes",
    "Mental Health",
    "Heart Rate",
]
TARGET_COLUMN = "Risk Level"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Low": 0, "High": 1})
    return df


def evaluate(name, y_test, preds, probs=None):
    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "F1 Score": round(f1_score(y_test, preds), 4),
    }
    if probs is not None:
        metrics["ROC AUC"] = round(roc_auc_score(y_test, probs), 4)
    cm = confusion_matrix(y_test, preds).tolist()
    return metrics, cm


def cross_validate(model, X, y, cv_folds=5):
    """5-fold stratified CV accuracy — guards against an optimistic single split."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    return {
        "cv_mean": round(scores.mean(), 4),
        "cv_std": round(scores.std(), 4),
        "cv_scores": [round(s, 4) for s in scores],
    }


def main():
    print("Loading data...")
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Separate scaler fit on the FULL dataset, used only for cross-validation scoring.
    cv_scaler = StandardScaler()
    X_scaled_full = cv_scaler.fit_transform(X)

    all_metrics = []
    all_cms = {}
    fitted_models = {}

    # ---- Logistic Regression ----
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    pred = lr.predict(X_test_scaled)
    proba = lr.predict_proba(X_test_scaled)[:, 1]
    m, cm = evaluate("Logistic Regression", y_test, pred, proba)
    m["CV"] = cross_validate(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), X_scaled_full, y)
    all_metrics.append(m)
    all_cms["Logistic Regression"] = cm
    fitted_models["Logistic Regression"] = lr

    # ---- Decision Tree ----
    print("Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    proba = dt.predict_proba(X_test)[:, 1]
    m, cm = evaluate("Decision Tree", y_test, pred, proba)
    m["CV"] = cross_validate(DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), X, y)
    all_metrics.append(m)
    all_cms["Decision Tree"] = cm
    fitted_models["Decision Tree"] = dt

    # ---- Random Forest ----
    print("Training Random Forest...")
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    proba = rf.predict_proba(X_test)[:, 1]
    m, cm = evaluate("Random Forest", y_test, pred, proba)
    m["CV"] = cross_validate(RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200), X, y)
    all_metrics.append(m)
    all_cms["Random Forest"] = cm
    fitted_models["Random Forest"] = rf

    feature_importance = (
        pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .to_dict(orient="records")
    )

    # ---- KNN (tuned) ----
    print("Tuning KNN...")
    grid_knn = GridSearchCV(
        KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9, 11]}, cv=5, scoring="accuracy"
    )
    grid_knn.fit(X_train_scaled, y_train)
    knn = grid_knn.best_estimator_
    pred = knn.predict(X_test_scaled)
    proba = knn.predict_proba(X_test_scaled)[:, 1]
    m, cm = evaluate(f"KNN (k={grid_knn.best_params_['n_neighbors']})", y_test, pred, proba)
    m["CV"] = cross_validate(KNeighborsClassifier(n_neighbors=grid_knn.best_params_["n_neighbors"]), X_scaled_full, y)
    all_metrics.append(m)
    all_cms["KNN"] = cm
    fitted_models["KNN"] = knn

    # ---- SVM (tuned) ----
    print("Tuning SVM (this can take a minute)...")
    grid_svm = GridSearchCV(
        SVC(probability=True),
        {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.1, 0.01], "kernel": ["rbf", "linear"]},
        cv=5,
        scoring="accuracy",
    )
    grid_svm.fit(X_train_scaled, y_train)
    svm = grid_svm.best_estimator_
    pred = svm.predict(X_test_scaled)
    proba = svm.predict_proba(X_test_scaled)[:, 1]
    m, cm = evaluate("SVM (tuned)", y_test, pred, proba)
    m["CV"] = cross_validate(SVC(probability=True, **grid_svm.best_params_), X_scaled_full, y)
    all_metrics.append(m)
    all_cms["SVM"] = cm
    fitted_models["SVM"] = svm

    # ---- Pick best by F1 score (robust to slight class imbalance) ----
    metrics_df = pd.DataFrame(all_metrics).sort_values("F1 Score", ascending=False)
    best_name = metrics_df.iloc[0]["Model"]
    print("\n=== Model comparison ===")
    print(metrics_df.to_string(index=False))
    print(f"\nBest model: {best_name}")

    # Map back to the underlying fitted model
    if best_name.startswith("KNN"):
        best_key = "KNN"
    else:
        best_key = best_name.replace(" (tuned)", "")
    best_model = fitted_models[best_key]
    needs_scaling = best_key in ("Logistic Regression", "KNN", "SVM")

    # ---- Calibration curve (is "80% confidence" actually right 80% of the time?) ----
    print("Computing calibration curve for best model...")
    best_X_test = X_test_scaled if needs_scaling else X_test
    best_proba_test = best_model.predict_proba(best_X_test)[:, 1]
    frac_positives, mean_predicted = calibration_curve(y_test, best_proba_test, n_bins=10, strategy="uniform")
    calibration_data = {
        "mean_predicted_prob": [round(v, 4) for v in mean_predicted],
        "fraction_of_positives": [round(v, 4) for v in frac_positives],
    }

    # ---- Precision-recall curve for best model ----
    print("Computing precision-recall curve for best model...")
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, best_proba_test)
    step = max(1, len(pr_thresholds) // 50)
    pr_curve_data = {
        "precision": [round(v, 4) for v in precisions[:-1][::step]],
        "recall": [round(v, 4) for v in recalls[:-1][::step]],
        "thresholds": [round(v, 4) for v in pr_thresholds[::step]],
    }

    # ---- Save artifacts ----
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    metadata = {
        "best_model_name": best_name,
        "best_model_key": best_key,
        "needs_scaling": needs_scaling,
        "feature_columns": FEATURE_COLUMNS,
        "all_metrics": metrics_df.to_dict(orient="records"),
        "confusion_matrices": all_cms,
        "feature_importance": feature_importance,
        "calibration_curve": calibration_data,
        "pr_curve": pr_curve_data,
        "class_mapping": {"0": "Low Risk", "1": "High Risk"},
        "dataset_size": len(df),
        "class_distribution": {
            "Low": int((y == 0).sum()),
            "High": int((y == 1).sum()),
        },
        "feature_ranges": {
            col: {
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "mean": float(X[col].mean()),
                "median": float(X[col].median()),
            }
            for col in FEATURE_COLUMNS
        },
    }
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved best model ({best_name}) to {MODEL_DIR}/best_model.pkl")
    print(f"Saved metrics + metadata to {MODEL_DIR}/metrics.json")
    print(f"\n5-fold CV results (mean accuracy ± std):")
    for m in all_metrics:
        if "CV" in m:
            print(f"  {m['Model']:30s} {m['CV']['cv_mean']:.4f} ± {m['CV']['cv_std']:.4f}")


if __name__ == "__main__":
    main()