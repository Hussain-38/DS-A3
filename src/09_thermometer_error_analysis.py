# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:24:08 2026

@author: hussa
"""


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# DS-A3 STEP 9: PH94 / PH21 SCIENTIFIC ERROR ANALYSIS
# Descriptive analysis of matched experimental runs.
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
# 1. Load matched observations
# ----------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required = [
    "Exp T(ºC)",
    "Exp P(kbar)",
    "T (Hbl-Pl-94-ave)",
    "T (Hbl-Pl-21-ave)",
    "Source",
    "source_original",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

for col in required[:4]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

matched = df.dropna(subset=required).copy()

if len(matched) < 10:
    raise ValueError("Too few matched observations for this analysis.")

matched["PH94_error_C"] = (
    matched["T (Hbl-Pl-94-ave)"] - matched["Exp T(ºC)"]
)
matched["PH21_error_C"] = (
    matched["T (Hbl-Pl-21-ave)"] - matched["Exp T(ºC)"]
)

matched["PH94_abs_error_C"] = matched["PH94_error_C"].abs()
matched["PH21_abs_error_C"] = matched["PH21_error_C"].abs()

print("=" * 65)
print("DS-A3: PH94 AND PH21 ERROR ANALYSIS")
print("=" * 65)
print("Matched observations:", len(matched))
print("Original source labels:", matched["source_original"].nunique())

# ----------------------------------------------------------
# 2. Overall accuracy summary
# ----------------------------------------------------------

summary_rows = []

for method, signed, absolute in [
    ("PH94", "PH94_error_C", "PH94_abs_error_C"),
    ("PH21", "PH21_error_C", "PH21_abs_error_C"),
]:
    summary_rows.append({
        "method": method,
        "n": len(matched),
        "mean_signed_error_C": matched[signed].mean(),
        "median_signed_error_C": matched[signed].median(),
        "MAE_C": matched[absolute].mean(),
        "RMSE_C": np.sqrt((matched[signed] ** 2).mean()),
        "error_SD_C": matched[signed].std(),
    })

overall = pd.DataFrame(summary_rows)
overall.to_csv(
    RESULTS_DIR / "thermometer_overall_metrics.csv",
    index=False
)

print("\nOVERALL METRICS")
print(overall.round(3).to_string(index=False))

# ----------------------------------------------------------
# Figure 1: Error distributions
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.boxplot(
    [
        matched["PH94_error_C"].to_numpy(),
        matched["PH21_error_C"].to_numpy()
    ],
    tick_labels=["PH94", "PH21"],
    showmeans=True
)

ax.axhline(0, linestyle="--", color="black")
ax.set_ylabel("Estimated − experimental temperature (°C)")
ax.set_title("PH94 and PH21 temperature-error distributions")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "fig06_thermometer_error_distributions.png",
    dpi=300, bbox_inches="tight"
)
plt.close(fig)

# ----------------------------------------------------------
# Figure 2: Error versus experimental temperature
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    matched["Exp T(ºC)"],
    matched["PH94_error_C"],
    label="PH94", alpha=0.8
)

ax.scatter(
    matched["Exp T(ºC)"],
    matched["PH21_error_C"],
    label="PH21", alpha=0.8, marker="s"
)

ax.axhline(0, linestyle="--", color="black")
ax.set_xlabel("Experimental temperature (°C)")
ax.set_ylabel("Thermometer error (°C)")
ax.set_title("Thermometer error across experimental temperatures")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "fig07_error_vs_experimental_temperature.png",
    dpi=300, bbox_inches="tight"
)
plt.close(fig)

# ----------------------------------------------------------
# Figure 3: Error versus experimental pressure
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    matched["Exp P(kbar)"],
    matched["PH94_error_C"],
    label="PH94", alpha=0.8
)

ax.scatter(
    matched["Exp P(kbar)"],
    matched["PH21_error_C"],
    label="PH21", alpha=0.8, marker="s"
)

ax.axhline(0, linestyle="--", color="black")
ax.set_xlabel("Experimental pressure (kbar)")
ax.set_ylabel("Thermometer error (°C)")
ax.set_title("Thermometer error across experimental pressures")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "fig08_error_vs_pressure.png",
    dpi=300, bbox_inches="tight"
)
plt.close(fig)

# ----------------------------------------------------------
# 3. Source-level error summaries
# ----------------------------------------------------------

# Retain the original labels here. Source normalization should
# be verified before interpreting differences as study effects.
source_summary = (
    matched.groupby("source_original", dropna=False)
    .agg(
        n=("Exp T(ºC)", "size"),
        PH94_bias_C=("PH94_error_C", "mean"),
        PH94_MAE_C=("PH94_abs_error_C", "mean"),
        PH21_bias_C=("PH21_error_C", "mean"),
        PH21_MAE_C=("PH21_abs_error_C", "mean"),
    )
    .reset_index()
    .sort_values("n", ascending=False)
)

source_summary.to_csv(
    RESULTS_DIR / "thermometer_source_error_summary.csv",
    index=False
)

print("\nSOURCE-LEVEL ERROR SUMMARY")
print(source_summary.round(2).to_string(index=False))

# ----------------------------------------------------------
# Figure 4: Source-level MAE
# ----------------------------------------------------------

plot_data = source_summary.sort_values(
    "PH94_MAE_C", ascending=True
)

positions = np.arange(len(plot_data))
height = max(6, 0.35 * len(plot_data))

fig, ax = plt.subplots(figsize=(10, height))

ax.barh(
    positions - 0.18,
    plot_data["PH94_MAE_C"],
    height=0.35,
    label="PH94"
)

ax.barh(
    positions + 0.18,
    plot_data["PH21_MAE_C"],
    height=0.35,
    label="PH21"
)
# Shorten long bibliographic labels for Figure 9 only
plot_data["source_original"] = (
    plot_data["source_original"]
    .str.replace(
        r"Ernst, W\.G\., Liu, J\., 1998.*",
        "Ernst & Liu (1998)",
        regex=True
    )
    .str.replace(
        r"Caricchi, L\., Ulmer, P\., Peccerillo, A\., 2006.*",
        "Caricchi et al. (2006)",
        regex=True
    )
    .str.replace(
        r"Prouteau, G\., Scaillet, B\., 2003.*",
        "Prouteau & Scaillet (2003)",
        regex=True
        )
    .str.replace(
    r"Blatter, D\.L\., Carmichael, I\.S\.E\., 2001.*",
    "Blatter & Carmichael (2001)",
    regex=True
)
    )
ax.set_yticks(positions)
ax.set_yticklabels(plot_data["source_original"])
ax.set_xlabel("Mean absolute error (°C)")
ax.set_title("PH94 and PH21 error by original source label")
ax.legend()
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "fig09_source_level_MAE.png",
    dpi=300, bbox_inches="tight"
)
plt.close(fig)

# ----------------------------------------------------------
# 4. Identify largest thermometer errors
# ----------------------------------------------------------

error_cases = matched.copy()

error_cases["larger_method"] = np.where(
    error_cases["PH94_abs_error_C"]
    > error_cases["PH21_abs_error_C"],
    "PH94",
    np.where(
        error_cases["PH21_abs_error_C"]
        > error_cases["PH94_abs_error_C"],
        "PH21",
        "Equal"
    )
)

error_cases.to_csv(
    RESULTS_DIR / "thermometer_error_cases.csv",
    index=False
)

print("\nLARGEST PH94 ERRORS")
print(
    matched.nlargest(5, "PH94_abs_error_C")[
        ["Source", "Run", "Exp T(ºC)",
         "T (Hbl-Pl-94-ave)", "PH94_error_C"]
    ].to_string(index=False)
)

print("\nLARGEST PH21 ERRORS")
print(
    matched.nlargest(5, "PH21_abs_error_C")[
        ["Source", "Run", "Exp T(ºC)",
         "T (Hbl-Pl-21-ave)", "PH21_error_C"]
    ].to_string(index=False)
)

print("\nSaved overall metrics and source summaries.")
print("Figures saved in:", FIGURES_DIR)
print("Results saved in:", RESULTS_DIR)
print("\nAnalysis finished.")
