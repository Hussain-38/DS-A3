# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:26:28 2026

@author: hussa
"""


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr

# ==========================================================
# DS-A3 STEP 10: ERROR ASSOCIATIONS
# PH94 / PH21 vs temperature, pressure and mineral chemistry
# ==========================================================

PROJECT_DIR = Path(
    r"D:\Data Science Assignment\My Projects\DS-A3"
)

INPUT_FILE = (
    PROJECT_DIR
    / "data_processed"
    / "S2_with_candidate_source_ids.csv"
)

RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# 1. Load the matched dataset
# ----------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required = [
    "Exp T(ºC)",
    "Exp P(kbar)",
    "T (Hbl-Pl-94-ave)",
    "T (Hbl-Pl-21-ave)",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

for col in df.columns:
    if col in required or "(Pl)" in col or "(Hbl)" in col:
        df[col] = pd.to_numeric(df[col], errors="coerce")

matched = df.dropna(subset=required).copy()

matched["PH94_error_C"] = (
    matched["T (Hbl-Pl-94-ave)"] - matched["Exp T(ºC)"]
)
matched["PH21_error_C"] = (
    matched["T (Hbl-Pl-21-ave)"] - matched["Exp T(ºC)"]
)

matched["PH94_abs_error_C"] = matched["PH94_error_C"].abs()
matched["PH21_abs_error_C"] = matched["PH21_error_C"].abs()

print("=" * 65)
print("DS-A3: ERROR ASSOCIATION ANALYSIS")
print("=" * 65)
print("Matched runs:", len(matched))

# ----------------------------------------------------------
# 2. Select explanatory variables
# ----------------------------------------------------------

candidate_features = [
    "Exp T(ºC)",
    "Exp P(kbar)",
    "SiO2 (Pl)",
    "TiO2 (Pl)",
    "Al2O3 (Pl)",
    "FeO (Pl)",
    "MnO (Pl)",
    "MgO (Pl)",
    "CaO (Pl)",
    "Na2O (Pl)",
    "K2O (Pl)",
    "SiO2 (Hbl)",
    "TiO2 (Hbl)",
    "Al2O3 (Hbl)",
    "FeO (Hbl)",
    "MnO (Hbl)",
    "MgO (Hbl)",
    "CaO (Hbl)",
    "Na2O (Hbl)",
    "K2O (HBl)",
]

features = [
    c for c in candidate_features
    if c in matched.columns
    and matched[c].notna().sum() >= 10
    and matched[c].nunique(dropna=True) > 1
]

# ----------------------------------------------------------
# 3. Spearman correlations with errors
# ----------------------------------------------------------

# Spearman is used for exploratory monotonic associations.
# Pairwise complete observations are used for each calculation.
outcomes = [
    "PH94_error_C",
    "PH94_abs_error_C",
    "PH21_error_C",
    "PH21_abs_error_C",
]

records = []

for feature in features:
    for outcome in outcomes:
        pair = matched[[feature, outcome]].dropna()

        if (
            len(pair) >= 10
            and pair[feature].nunique() > 1
            and pair[outcome].nunique() > 1
        ):
            rho, p_value = spearmanr(
                pair[feature], pair[outcome]
            )

            records.append({
                "feature": feature,
                "error_metric": outcome,
                "n": len(pair),
                "spearman_rho": rho,
                "p_value_unadjusted": p_value,
            })

correlations = pd.DataFrame(records)

# Adjust for multiple exploratory tests using Benjamini-Hochberg.
if not correlations.empty:
    p = correlations["p_value_unadjusted"].to_numpy()
    order = np.argsort(p)
    ranked_p = p[order]
    m = len(p)

    adjusted_sorted = ranked_p * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    correlations["p_value_BH"] = adjusted

correlations.to_csv(
    RESULTS_DIR / "error_feature_correlations.csv",
    index=False
)

print("\nSTRONGEST EXPLORATORY ASSOCIATIONS")
if not correlations.empty:
    for outcome in outcomes:
        print(f"\nOutcome: {outcome}")
        subset = correlations[
            correlations["error_metric"] == outcome
        ].copy()

        subset["abs_rho"] = subset["spearman_rho"].abs()

        print(
            subset.sort_values(
                "abs_rho", ascending=False
            ).head(5)[
                [
                    "feature", "n",
                    "spearman_rho",
                    "p_value_unadjusted",
                    "p_value_BH",
                ]
            ].round(4).to_string(index=False)
        )

# ----------------------------------------------------------
# 4. Correlation heatmap for selected variables
# ----------------------------------------------------------

heatmap_cols = features + [
    "PH94_error_C",
    "PH21_error_C",
]

heatmap_data = matched[heatmap_cols].corr(
    method="spearman"
)

fig, ax = plt.subplots(figsize=(13, 10))

sns.heatmap(
    heatmap_data,
    cmap="vlag",
    center=0,
    vmin=-1,
    vmax=1,
    square=False,
    ax=ax
)

ax.set_title(
    "Spearman correlation: experimental conditions, chemistry and errors"
)
fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "fig10_error_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# ----------------------------------------------------------
# 5. Error against experimental temperature and pressure
# ----------------------------------------------------------

for xvar, xlabel, filename, title in [
    (
        "Exp T(ºC)",
        "Experimental temperature (°C)",
        "fig11_error_vs_temperature.png",
        "Thermometer error versus experimental temperature",
    ),
    (
        "Exp P(kbar)",
        "Experimental pressure (kbar)",
        "fig12_error_vs_pressure_detailed.png",
        "Thermometer error versus experimental pressure",
    ),
]:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        matched[xvar],
        matched["PH94_error_C"],
        label="PH94",
        alpha=0.8
    )

    ax.scatter(
        matched[xvar],
        matched["PH21_error_C"],
        label="PH21",
        alpha=0.8,
        marker="s"
    )

    ax.axhline(0, linestyle="--", color="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Estimated − experimental temperature (°C)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

# ----------------------------------------------------------
# 6. Save the matched error-analysis dataset
# ----------------------------------------------------------

matched.to_csv(
    RESULTS_DIR / "matched_error_driver_data.csv",
    index=False
)

print("\nSaved:")
print(RESULTS_DIR / "error_feature_correlations.csv")
print(RESULTS_DIR / "matched_error_driver_data.csv")
print(FIGURES_DIR / "fig10_error_correlation_heatmap.png")
print(FIGURES_DIR / "fig11_error_vs_temperature.png")
print(FIGURES_DIR / "fig12_error_vs_pressure_detailed.png")

print("\nStep 10 completed.")
print(
    "Interpret correlations cautiously: observations from "
    "the same experimental study may not be independent."
)
