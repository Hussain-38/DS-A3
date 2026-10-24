# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:21:28 2026

@author: hussa
"""


from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# ==========================================================
# DS-A3 STEP 8: DIAGNOSTICS AND ERROR ANALYSIS
# Test set is NOT used in this step.
# ==========================================================

PROJECT_DIR = Path(
    r"D:\Data Science Assignment\My Projects\DS-A3"
)

DATA_FILE = PROJECT_DIR / "data_processed" / "S2_frozen_split.csv"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "validation"].copy()

X_train = train[FEATURES].apply(pd.to_numeric, errors="coerce")
y_train = pd.to_numeric(train[TARGET], errors="coerce")

X_val = val[FEATURES].apply(pd.to_numeric, errors="coerce")
y_val = pd.to_numeric(val[TARGET], errors="coerce")

# Use the validation-selected Ridge model and linear comparator.
def make_model(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])

ridge = make_model(Ridge(alpha=1.0))
linear = make_model(LinearRegression())

ridge.fit(X_train, y_train)
linear.fit(X_train, y_train)

pred = ridge.predict(X_val)
linear_pred = linear.predict(X_val)

errors = y_val.to_numpy() - pred
abs_errors = np.abs(errors)

diagnostics = val[
    ["Source", "Run", GROUP, TARGET, "Exp P(kbar)"]
].copy()

diagnostics["predicted_temperature_C"] = pred
diagnostics["residual_C"] = errors
diagnostics["absolute_error_C"] = abs_errors
diagnostics["linear_prediction_C"] = linear_pred
diagnostics["linear_residual_C"] = y_val.to_numpy() - linear_pred

diagnostics.to_csv(
    RESULTS_DIR / "validation_error_cases.csv",
    index=False
)

# ----------------------------------------------------------
# Figure 1: Observed versus predicted temperature
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(y_val, pred, s=65, alpha=0.85, label="Ridge")
low = min(y_val.min(), pred.min())
high = max(y_val.max(), pred.max())

ax.plot([low, high], [low, high], "--", color="black",
        label="Perfect prediction")

ax.set_xlabel("Experimental temperature (°C)")
ax.set_ylabel("Predicted temperature (°C)")
ax.set_title("Ridge: observed versus predicted temperature")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "fig01_observed_vs_predicted.png",
    dpi=300, bbox_inches="tight"
)
plt.close(fig)

# ----------------------------------------------------------
# Figure 2: Residuals versus predicted temperature
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(pred, errors, s=65, alpha=0.85)
ax.axhline(0, linestyle="--", color="black")
ax.set_xlabel("Predicted temperature (°C)")
ax.set_ylabel("Residual: observed − predicted (°C)")
ax.set_title("Ridge residual diagnostic")
ax.grid(alpha=0.25)
fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "fig02_residuals_vs_fitted.png",
    dpi=300, bbox_inches="tight"
)
plt.close(fig)

# ----------------------------------------------------------
# Figure 3: Absolute errors versus pressure
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(
    val["Exp P(kbar)"], abs_errors,
    s=65, alpha=0.85
)

ax.set_xlabel("Experimental pressure (kbar)")
ax.set_ylabel("Absolute prediction error (°C)")
ax.set_title("Prediction error across experimental pressure")
ax.grid(alpha=0.25)
fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "fig03_error_vs_pressure.png",
    dpi=300, bbox_inches="tight"
)
plt.close(fig)

# ----------------------------------------------------------
# Figure 4: Learning curve using grouped CV on training data
# ----------------------------------------------------------

n_groups = train[GROUP].nunique()
n_splits = min(5, n_groups)

if n_splits >= 2:
    cv = GroupKFold(n_splits=n_splits)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sizes, train_scores, cv_scores = learning_curve(
            estimator=make_model(Ridge(alpha=1.0)),
            X=X_train,
            y=y_train,
            groups=train[GROUP],
            cv=cv,
            scoring="neg_mean_absolute_error",
            train_sizes=np.linspace(0.3, 1.0, 5),
            n_jobs=-1
        )

    train_mae = -train_scores.mean(axis=1)
    cv_mae = -cv_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(sizes, train_mae, marker="o", label="Training MAE")
    ax.plot(sizes, cv_mae, marker="o", label="Grouped CV MAE")

    ax.set_xlabel("Number of training observations")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Ridge learning curve")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "fig04_learning_curve.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    pd.DataFrame({
        "training_observations": sizes,
        "training_MAE_C": train_mae,
        "grouped_CV_MAE_C": cv_mae
    }).to_csv(
        RESULTS_DIR / "learning_curve_results.csv",
        index=False
    )

# ----------------------------------------------------------
# Figure 5: Influence and heteroscedasticity diagnostics
# ----------------------------------------------------------

# Fit an OLS diagnostic model on training data only.
# Scaling and median imputation are fitted to training data.
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_imp = imputer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_imp)

X_ols = sm.add_constant(X_train_scaled, has_constant="add")
ols = sm.OLS(y_train.to_numpy(), X_ols).fit()

influence = ols.get_influence()
cooks_d = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag
ols_residuals = ols.resid
ols_fitted = ols.fittedvalues

influence_table = pd.DataFrame({
    "row_index": train.index,
    "study_id": train[GROUP].to_numpy(),
    "observed_temperature_C": y_train.to_numpy(),
    "OLS_fitted_temperature_C": ols_fitted,
    "OLS_residual_C": ols_residuals,
    "leverage": leverage,
    "cooks_distance": cooks_d
})

influence_table.to_csv(
    RESULTS_DIR / "training_influence_diagnostics.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    leverage, ols_residuals,
    s=40 + 300 * np.minimum(cooks_d, 1),
    alpha=0.75
)

ax.axhline(0, linestyle="--", color="black")
ax.set_xlabel("Leverage")
ax.set_ylabel("OLS residual (°C)")
ax.set_title("Training-set influence diagnostic")
ax.grid(alpha=0.25)
fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "fig05_influence.png",
    dpi=300, bbox_inches="tight"
)
plt.close(fig)

# Breusch–Pagan test against fitted values.
# This is a diagnostic, not a definitive decision rule.
bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(
    ols_residuals,
    sm.add_constant(ols_fitted)
)

with open(
    RESULTS_DIR / "heteroscedasticity_test.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("Training-set OLS residual diagnostic\n")
    f.write(f"Breusch-Pagan LM statistic: {bp_lm:.6f}\n")
    f.write(f"LM p-value: {bp_lm_p:.6f}\n")
    f.write(f"F statistic: {bp_f:.6f}\n")
    f.write(f"F p-value: {bp_f_p:.6f}\n")
    f.write(
        "\nInterpret cautiously: the sample is small and "
        "the OLS model has many predictors.\n"
    )

print("=" * 65)
print("DS-A3: DIAGNOSTICS COMPLETED")
print("=" * 65)

print("\nVALIDATION RIDGE METRICS")
print("MAE (°C):", round(mean_absolute_error(y_val, pred), 3))
print(
    "RMSE (°C):",
    round(np.sqrt(mean_squared_error(y_val, pred)), 3)
)

print("\nLargest validation errors:")
print(
    diagnostics.nlargest(5, "absolute_error_C")[
        ["Source", "Run", TARGET,
         "predicted_temperature_C", "residual_C",
         "absolute_error_C"]
    ].to_string(index=False)
)

print("\nBREUSCH-PAGAN TEST")
print("LM statistic:", round(bp_lm, 4))
print("LM p-value:", round(bp_lm_p, 4))

print("\nFigures saved in:", FIGURES_DIR)
print("Results saved in:", RESULTS_DIR)
print("\nThe test set was not used.")
