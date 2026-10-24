# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:29:23 2026

@author: hussa
"""


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# ==========================================================
# DS-A3 STEP 12: COEFFICIENTS AND MODEL COMPLEXITY
# Uses training + validation only; does not use test data.
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

development = df[
    df["split"].isin(["train", "validation"])
].copy()

X = development[FEATURES].apply(pd.to_numeric, errors="raise")
y = pd.to_numeric(development[TARGET], errors="raise")

# Standardized coefficients allow a more meaningful comparison
# of coefficient magnitudes across predictors with different units.
def make_model(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])

models = {
    "Linear Regression": make_model(LinearRegression()),
    "Ridge alpha=1.0": make_model(Ridge(alpha=1.0)),
    "Lasso alpha=0.01": make_model(
        Lasso(alpha=0.01, max_iter=50000)
    ),
}

coefficient_tables = []
complexity_rows = []

for name, pipeline in models.items():
    pipeline.fit(X, y)

    estimator = pipeline.named_steps["model"]
    coefficients = estimator.coef_

    coefficient_tables.append(
        pd.DataFrame({
            "model": name,
            "feature": FEATURES,
            "standardized_coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "nonzero": np.abs(coefficients) > 1e-8,
        })
    )

    complexity_rows.append({
        "model": name,
        "n_predictors": len(FEATURES),
        "n_nonzero_coefficients": int(
            (np.abs(coefficients) > 1e-8).sum()
        ),
        "n_zero_coefficients": int(
            (np.abs(coefficients) <= 1e-8).sum()
        ),
        "intercept_C": estimator.intercept_,
    })

coefficients_df = pd.concat(
    coefficient_tables, ignore_index=True
)

complexity_df = pd.DataFrame(complexity_rows)

coefficients_df.to_csv(
    RESULTS_DIR / "standardized_model_coefficients.csv",
    index=False
)

complexity_df.to_csv(
    RESULTS_DIR / "model_complexity_summary.csv",
    index=False
)

print("=" * 65)
print("DS-A3: COEFFICIENT AND COMPLEXITY ANALYSIS")
print("=" * 65)

print("\nMODEL COMPLEXITY")
print(complexity_df.round(4).to_string(index=False))

for name in models:
    print(f"\nLARGEST ABSOLUTE COEFFICIENTS: {name}")

    subset = coefficients_df[
        coefficients_df["model"] == name
    ].sort_values("absolute_coefficient", ascending=False)

    print(
        subset[
            ["feature", "standardized_coefficient"]
        ].head(10).round(4).to_string(index=False)
    )

# Plot Ridge and Lasso coefficients.
for model_name, filename in [
    ("Ridge alpha=1.0", "fig15_ridge_coefficients.png"),
    ("Lasso alpha=0.01", "fig16_lasso_coefficients.png"),
]:
    subset = coefficients_df[
        coefficients_df["model"] == model_name
    ].sort_values("standardized_coefficient")

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(
        subset["feature"],
        subset["standardized_coefficient"]
    )

    ax.axvline(0, linestyle="--", color="black")
    ax.set_xlabel("Standardized regression coefficient")
    ax.set_title(f"{model_name}: standardized coefficients")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

print("\nSaved:")
print(RESULTS_DIR / "standardized_model_coefficients.csv")
print(RESULTS_DIR / "model_complexity_summary.csv")
print(FIGURES_DIR / "fig15_ridge_coefficients.png")
print(FIGURES_DIR / "fig16_lasso_coefficients.png")

print("\nCoefficient analysis completed.")
print("These coefficients are associations, not causal effects.")
