# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:28:13 2026

@author: hussa
"""


from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# DS-A3 STEP 11: FINAL HELD-OUT TEST EVALUATION
# Do not use test results for model selection or tuning.
# ==========================================================

PROJECT_DIR = Path(
    r"D:\Data Science Assignment\My Projects\DS-A3"
)

DATA_FILE = (
    PROJECT_DIR / "data_processed" / "S2_frozen_split.csv"
)

RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Exp T(ºC)"
GROUP = "study_id"
ALPHA = 1.0  # Fixed from the preceding model-selection step.

FEATURES = [
    "Exp P(kbar)",
    "SiO2 (Pl)", "TiO2 (Pl)", "Al2O3 (Pl)",
    "FeO (Pl)", "MnO (Pl)", "MgO (Pl)",
    "CaO (Pl)", "Na2O (Pl)", "K2O (Pl)",
    "SiO2 (Hbl)", "TiO2 (Hbl)", "Al2O3 (Hbl)",
    "FeO (Hbl)", "MnO (Hbl)", "MgO (Hbl)",
    "CaO (Hbl)", "Na2O (Hbl)", "K2O (HBl)",
]

print("=" * 65)
print("DS-A3: FINAL HELD-OUT TEST EVALUATION")
print("=" * 65)

if not DATA_FILE.is_file():
    raise FileNotFoundError(f"Missing frozen split: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

required = FEATURES + [TARGET, GROUP, "split"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise KeyError(f"Missing required columns: {missing}")

if df[required].isna().any().any():
    raise ValueError(
        "Missing values found in the frozen analysis columns. "
        "Do not silently change the frozen dataset."
    )

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "validation"].copy()
test = df[df["split"] == "test"].copy()

if train.empty or val.empty or test.empty:
    raise ValueError("One or more data partitions are empty.")

# ----------------------------------------------------------
# 1. Verify split and group isolation
# ----------------------------------------------------------

train_groups = set(train[GROUP])
val_groups = set(val[GROUP])
test_groups = set(test[GROUP])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

print("\nSPLIT AUDIT")
for name, part in [
    ("Train", train),
    ("Validation", val),
    ("Test", test),
]:
    print(
        f"{name}: {len(part)} observations, "
        f"{part[GROUP].nunique()} study groups"
    )

print("Train/validation overlap:", train_groups & val_groups)
print("Train/test overlap:", train_groups & test_groups)
print("Validation/test overlap:", val_groups & test_groups)

# ----------------------------------------------------------
# 2. Freeze the final training procedure
# ----------------------------------------------------------

# Selection is complete. Refit the chosen Ridge model using
# training + validation data. Do not use test data for fitting.
development = pd.concat([train, val]).copy()

X_dev = development[FEATURES].apply(pd.to_numeric, errors="raise")
y_dev = pd.to_numeric(development[TARGET], errors="raise")

X_test = test[FEATURES].apply(pd.to_numeric, errors="raise")
y_test = pd.to_numeric(test[TARGET], errors="raise")

ridge = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=ALPHA)),
])

ridge.fit(X_dev, y_dev)
ridge_pred = ridge.predict(X_test)

# Baseline: mean temperature from the same development data.
baseline_value = y_dev.mean()
baseline_pred = np.repeat(baseline_value, len(test))

# ----------------------------------------------------------
# 3. Calculate final test metrics
# ----------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    return {
        "MAE_C": mean_absolute_error(y_true, y_pred),
        "RMSE_C": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

comparison = pd.DataFrame([
    {
        "model": "Naive mean baseline",
        **calculate_metrics(y_test, baseline_pred),
    },
    {
        "model": f"Ridge alpha={ALPHA}",
        **calculate_metrics(y_test, ridge_pred),
    },
])

comparison.to_csv(
    RESULTS_DIR / "final_test_model_comparison.csv",
    index=False
)

# Save predictions and errors for every held-out run.
predictions = test[
    [GROUP, "Source", "Run", TARGET, "Exp P(kbar)"]
].copy()

predictions["baseline_prediction_C"] = baseline_pred
predictions["baseline_residual_C"] = y_test.to_numpy() - baseline_pred

predictions["ridge_prediction_C"] = ridge_pred
predictions["ridge_residual_C"] = y_test.to_numpy() - ridge_pred
predictions["ridge_absolute_error_C"] = np.abs(
    predictions["ridge_residual_C"]
)

predictions.to_csv(
    RESULTS_DIR / "final_test_predictions.csv",
    index=False
)

# ----------------------------------------------------------
# 4. Figure: observed versus predicted temperatures
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(
    y_test, baseline_pred,
    marker="s", s=65, alpha=0.8,
    label="Naive mean baseline"
)

ax.scatter(
    y_test, ridge_pred,
    marker="o", s=65, alpha=0.85,
    label="Ridge"
)

lower = min(y_test.min(), baseline_pred.min(), ridge_pred.min())
upper = max(y_test.max(), baseline_pred.max(), ridge_pred.max())

ax.plot(
    [lower, upper], [lower, upper],
    linestyle="--", color="black",
    label="Perfect prediction"
)

ax.set_xlabel("Experimental temperature (°C)")
ax.set_ylabel("Predicted temperature (°C)")
ax.set_title("Final test-set predictions")
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "fig13_final_test_predictions.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# ----------------------------------------------------------
# 5. Figure: residuals for the final Ridge model
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(
    ridge_pred,
    y_test.to_numpy() - ridge_pred,
    s=65,
    alpha=0.85
)

ax.axhline(0, linestyle="--", color="black")
ax.set_xlabel("Ridge predicted temperature (°C)")
ax.set_ylabel("Residual: observed − predicted (°C)")
ax.set_title("Final test-set Ridge residuals")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "fig14_final_test_residuals.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# ----------------------------------------------------------
# 6. Save study-level test results
# ----------------------------------------------------------

study_results = (
    predictions.groupby(GROUP)
    .agg(
        n_runs=(TARGET, "size"),
        ridge_MAE_C=("ridge_absolute_error_C", "mean"),
        ridge_bias_C=("ridge_residual_C", "mean"),
    )
    .reset_index()
)

study_results.to_csv(
    RESULTS_DIR / "final_test_study_errors.csv",
    index=False
)

# Record the evaluation procedure.
protocol = {
    "target": TARGET,
    "features": FEATURES,
    "selected_model": "Ridge",
    "ridge_alpha": ALPHA,
    "final_fit_data": "Training + validation partitions",
    "test_used_for_fitting": False,
    "test_used_for_tuning": False,
    "baseline": "Mean target in training + validation partitions",
    "split_file": str(DATA_FILE),
}

with open(
    RESULTS_DIR / "final_test_protocol.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(protocol, f, indent=2)

print("\nFINAL TEST COMPARISON")
print(comparison.round(3).to_string(index=False))

print("\nTEST-SET PREDICTIONS")
print(
    predictions[
        [
            GROUP, "Run", TARGET,
            "baseline_prediction_C",
            "ridge_prediction_C",
            "ridge_residual_C",
            "ridge_absolute_error_C",
        ]
    ].round(2).to_string(index=False)
)

print("\nERROR BY STUDY")
print(study_results.round(2).to_string(index=False))

print("\nSaved:")
print(RESULTS_DIR / "final_test_model_comparison.csv")
print(RESULTS_DIR / "final_test_predictions.csv")
print(RESULTS_DIR / "final_test_study_errors.csv")
print(RESULTS_DIR / "final_test_protocol.json")
print(FIGURES_DIR / "fig13_final_test_predictions.png")
print(FIGURES_DIR / "fig14_final_test_residuals.png")

print("\nFinal test evaluation completed.")
print("Do not change the model based on these test results.")
