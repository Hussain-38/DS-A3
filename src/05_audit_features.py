# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:15:18 2026

@author: hussa
"""


from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================================
# DS-A3 STEP 5: FEATURE AND MISSINGNESS AUDIT
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"Could not find:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

# Standardize empty strings as missing values.
df = df.replace(r"^\s*$", np.nan, regex=True)

print("=" * 65)
print("DS-A3: FEATURE AND MISSINGNESS AUDIT")
print("=" * 65)

print("\n1. DATASET DIMENSIONS")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n2. ALL COLUMN NAMES")
for col in df.columns:
    print(repr(col))

# Identify columns that are entirely empty.
empty_cols = df.columns[df.isna().all()].tolist()

print("\n3. COMPLETELY EMPTY COLUMNS")
print(empty_cols if empty_cols else "None")

# Identify numeric columns.
numeric = df.select_dtypes(include="number")

print("\n4. NUMERIC COLUMNS")
for col in numeric.columns:
    print("-", col)

# Create a missingness and uniqueness audit.
audit = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "missing_count": [df[c].isna().sum() for c in df.columns],
    "missing_percent": [
        round(100 * df[c].isna().mean(), 2)
        for c in df.columns
    ],
    "unique_nonmissing": [
        df[c].nunique(dropna=True) for c in df.columns
    ]
})

audit.to_csv(
    RESULTS_DIR / "feature_missingness_audit.csv",
    index=False
)

print("\n5. MISSINGNESS AUDIT")
print(
    audit.sort_values(
        "missing_percent", ascending=False
    ).to_string(index=False)
)

# Examine the target and existing thermometer estimates.
key_columns = [
    "Exp T(ºC)",
    "Exp P(kbar)",
    "T (Hbl-Pl-94-ave)",
    "T (Hbl-Pl-21-ave)",
]

print("\n6. KEY VARIABLE AVAILABILITY")

for col in key_columns:
    if col in df.columns:
        print(
            f"{col}: "
            f"{df[col].notna().sum()} non-missing "
            f"of {len(df)} rows"
        )
    else:
        print(f"NOT FOUND: {col}")

# Report the distribution of source-group sizes.
if "study_id" in df.columns:
    group_sizes = df.groupby(
        "study_id", dropna=False
    ).size()

    print("\n7. STUDY GROUP SIZES")
    print(group_sizes.describe().round(2).to_string())
    print("Singleton groups:", int((group_sizes == 1).sum()))
    print("Groups with at least 3 runs:", int((group_sizes >= 3).sum()))

print("\nSaved:")
print(RESULTS_DIR / "feature_missingness_audit.csv")

print("\nFeature audit completed.")
