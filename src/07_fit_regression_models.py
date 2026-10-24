# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:19:38 2026

@author: hussa
"""


from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# DS-A3 STEP 7: BASELINE AND REGRESSION CANDIDATES
# ==========================================================

PROJECT_DIR = Path(
    r"D:\Data Science Assignment\My Projects\DS-A3"
)

DATA_FILE = (
    PROJECT_DIR / "data_processed" / "S2_frozen_split.csv"
)
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20261021
TARGET = "Exp T(ºC)"
GROUP = "study_id"

FEATURES = [
    "Exp P(kbar)",
    "SiO2 (Pl)", "TiO2 (Pl)", "Al2O3 (Pl)",
    "FeO (Pl)", "MnO (Pl)", "MgO (Pl)",
    "CaO (Pl)", "Na2O (Pl)", "K2O (Pl)",
    "SiO2 (Hbl)", "TiO2 (Hbl)", "Al2O3 (Hbl)",
    "FeO (Hbl)", "MnO (Hbl)", "MgO (Hbl)",
    "CaO (Hbl)", "Na2O (Hbl)", "K2O (HBl)",
]

df = pd.read_csv(DATA_FILE)

train = df.loc[df["split"] == "train"].copy()
val = df.loc[df["split"] == "validation"].copy()

if train.empty or val.empty:
    raise ValueError("Training or validation partition is empty.")

X_train = train[FEATURES].apply(pd.to_numeric, errors="coerce")
y_train = pd.to_numeric(train[TARGET], errors="coerce")
X_val = val[FEATURES].apply(pd.to_numeric, errors="coerce")
y_val = pd.to_numeric(val[TARGET], errors="coerce")

if y_train.isna().any() or y_val.isna().any():
    raise ValueError("Missing target values detected.")

# ----------------------------------------------------------
# 1. Baseline: training-set mean
# ----------------------------------------------------------

baseline_prediction = np.repeat(y_train.mean(), len(y_val))

# ----------------------------------------------------------
# 2. Preprocessing and models
# ----------------------------------------------------------

def make_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

models = {
    "Linear Regression": make_pipeline(LinearRegression()),
    "Ridge": make_pipeline(Ridge()),
    "Lasso": make_pipeline(Lasso(max_iter=50000, random_state=SEED)),
}

# ----------------------------------------------------------
# 3. Grouped CV on training data only
# ----------------------------------------------------------

n_groups = train[GROUP].nunique()
n_splits = min(5, n_groups)

if n_splits < 2:
    raise ValueError("Insufficient training study groups for grouped CV.")

cv = GroupKFold(n_splits=n_splits)
groups_train = train[GROUP].to_numpy()

# Record the preprocessing and tuning configuration.
configuration = {
    "seed": SEED,
    "target": TARGET,
    "features": FEATURES,
    "group_column": GROUP,
    "cv": f"GroupKFold(n_splits={n_splits})",
    "selection_metric": "MAE",
    "tuning_data": "Training partition only",
    "test_set_used_for_selection": False,
    "ridge_lasso_alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0],
}

with open(
    RESULTS_DIR / "model_configuration.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(configuration, f, indent=2)

fitted_models = {}
cv_results = {}
validation_predictions = {
    "observed_temperature_C": y_val.to_numpy()
}

# Evaluate the baseline on validation.
validation_predictions["Naive baseline"] = baseline_prediction

def metrics(y_true, y_pred):
    return {
        "MAE_C": mean_absolute_error(y_true, y_pred),
        "RMSE_C": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

summary = [{
    "model": "Naive baseline",
    **metrics(y_val, baseline_prediction),
    "best_alpha": np.nan,
    "best_CV_MAE_C": np.nan,
}]

param_grid = {
    "Ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
}

for name, pipeline in models.items():
    print(f"\nFitting {name}...")

    if name in param_grid:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid[name],
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        search.fit(
            X_train,
            y_train,
            groups=groups_train
        )

        fitted = search.best_estimator_
        best_alpha = search.best_params_["model__alpha"]
        best_cv_mae = -search.best_score_

    else:
        fitted = pipeline
        fitted.fit(X_train, y_train)
        best_alpha = np.nan
        best_cv_mae = np.nan

    pred_val = fitted.predict(X_val)

    fitted_models[name] = fitted
    validation_predictions[name] = pred_val

    summary.append({
        "model": name,
        **metrics(y_val, pred_val),
        "best_alpha": best_alpha,
        "best_CV_MAE_C": best_cv_mae,
    })

# ----------------------------------------------------------
# 4. Save validation results and predictions
# ----------------------------------------------------------

summary_df = pd.DataFrame(summary).sort_values("MAE_C")

summary_df.to_csv(
    RESULTS_DIR / "validation_model_comparison.csv",
    index=False
)

pred_df = pd.DataFrame(validation_predictions)
pred_df["study_id"] = val[GROUP].to_numpy()
pred_df.to_csv(
    RESULTS_DIR / "validation_predictions.csv",
    index=False
)

print("\nVALIDATION MODEL COMPARISON")
print(summary_df.round(3).to_string(index=False))

print("\nSaved:")
print(RESULTS_DIR / "validation_model_comparison.csv")
print(RESULTS_DIR / "validation_predictions.csv")
print(RESULTS_DIR / "model_configuration.json")

print("\nStep 7 completed. The test partition was not evaluated.")
