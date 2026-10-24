# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 09:59:27 2026

@author: hussa
"""


from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================================
# DS-A3 STEP 2: PREPARE AND AUDIT S2
# ==========================================================

# Use the exact workbook location verified in Step 1.
DATA_FILE = Path(
    r"D:\Data Science Assignment\My Projects\DS-A2"
    r"\DS_A2_S2_stage1\DS_A2_S2_stage1"
    r"\DS-A2-Inference-Uncertainty\data\S2_source.xlsx"
)

# Create output folders inside the DS-A3 project.
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
DATA_DIR = PROJECT_DIR / "data_processed"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("DS-A3: DATA PREPARATION AND AUDIT")
print("=" * 65)

if not DATA_FILE.is_file():
    raise FileNotFoundError(
        f"Workbook not found. Check this path:\n{DATA_FILE}"
    )

# Read the original worksheet.
raw = pd.read_excel(DATA_FILE, sheet_name="Table S2")

# Remove entirely empty separator columns.
df = raw.dropna(axis=1, how="all").copy()

# Remove rows without experimental temperature or pressure.
df["Exp T(ºC)"] = pd.to_numeric(
    df["Exp T(ºC)"], errors="coerce"
)
df["Exp P(kbar)"] = pd.to_numeric(
    df["Exp P(kbar)"], errors="coerce"
)

experimental = df.dropna(
    subset=["Exp T(ºC)", "Exp P(kbar)"]
).copy()

print("\n1. DATASET COUNTS")
print("Original spreadsheet rows:", len(raw))
print("Rows with experimental T and P:", len(experimental))

# Check the repeated experimental temperature column.
if "Exp T(ºC).1" in experimental.columns:
    repeated_t = pd.to_numeric(
        experimental["Exp T(ºC).1"], errors="coerce"
    )
    mismatch = (
        experimental["Exp T(ºC)"] - repeated_t
    ).abs()

    print(
        "Rows with inconsistent repeated temperatures:",
        int((mismatch > 1e-8).sum())
    )

# Report missing source identifiers.
print("\n2. SOURCE IDENTIFIERS")
print("Missing Source values:", experimental["Source"].isna().sum())

print("\nRuns per source:")
print(
    experimental.groupby("Source", dropna=False)
    .size()
    .sort_values(ascending=False)
    .to_string()
)

# Define the two main thermometer columns.
ph94 = "T (Hbl-Pl-94-ave)"
ph21 = "T (Hbl-Pl-21-ave)"

# Check that both columns exist.
for col in [ph94, ph21]:
    if col not in experimental.columns:
        raise KeyError(f"Expected column not found: {col}")

# Keep runs where both methods have estimates.
paired = experimental.dropna(
    subset=[ph94, ph21, "Source"]
).copy()

paired["PH94_error_C"] = (
    paired[ph94] - paired["Exp T(ºC)"]
)
paired["PH21_error_C"] = (
    paired[ph21] - paired["Exp T(ºC)"]
)

paired["PH94_absolute_error_C"] = (
    paired["PH94_error_C"].abs()
)
paired["PH21_absolute_error_C"] = (
    paired["PH21_error_C"].abs()
)

print("\n3. MATCHED THERMOMETER COMPARISON")
print("Runs with both PH94 and PH21:", len(paired))
print("Distinct source labels:", paired["Source"].nunique())

print("\nSummary of thermometer errors (°C):")
print(
    paired[
        [
            "PH94_error_C",
            "PH21_error_C",
            "PH94_absolute_error_C",
            "PH21_absolute_error_C",
        ]
    ].describe().round(2).to_string()
)

# Save the cleaned experimental dataset.
experimental.to_csv(
    DATA_DIR / "S2_experimental_runs.csv",
    index=False
)

# Save the matched PH94/PH21 comparison dataset.
paired.to_csv(
    DATA_DIR / "S2_matched_PH94_PH21.csv",
    index=False
)

# Save source-level audit.
source_counts = (
    experimental.groupby("Source", dropna=False)
    .size()
    .rename("n_runs")
    .reset_index()
)

source_counts.to_csv(
    RESULTS_DIR / "source_run_counts.csv",
    index=False
)

print("\n4. FILES SAVED")
print(DATA_DIR / "S2_experimental_runs.csv")
print(DATA_DIR / "S2_matched_PH94_PH21.csv")
print(RESULTS_DIR / "source_run_counts.csv")

print("\nStep 2 preparation finished.")
