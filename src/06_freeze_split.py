# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:17:55 2026

@author: hussa
"""


from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ==========================================================
# DS-A3 STEP 6: DEFINE TASK AND FREEZE GROUPED DATA SPLIT
# ==========================================================

PROJECT_DIR = Path(
    r"D:\Data Science Assignment\My Projects\DS-A3"
)

INPUT_FILE = (
    PROJECT_DIR
    / "data_processed"
    / "S2_with_candidate_source_ids.csv"
)

MAPPING_FILE = (
    PROJECT_DIR
    / "results"
    / "verified_source_mapping.csv"
)

RESULTS_DIR = PROJECT_DIR / "results"
DATA_DIR = PROJECT_DIR / "data_processed"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20261021

# -------------------------------
# 1. Load data and source mapping
# -------------------------------

df = pd.read_csv(INPUT_FILE)
mapping = pd.read_csv(MAPPING_FILE)

required_mapping = {"source_original", "study_id"}
if not required_mapping.issubset(mapping.columns):
    raise ValueError(
        "Source mapping must contain source_original and study_id."
    )

mapping = mapping[
    ["source_original", "study_id"]
].drop_duplicates()

# A source label must map to exactly one study ID.
conflicts = mapping.groupby("source_original")[
    "study_id"
].nunique()

if (conflicts > 1).any():
    raise ValueError(
        "Some source labels map to multiple study IDs. "
        "Resolve these conflicts before splitting."
    )

df["source_original"] = df["source_original"].astype("string").str.strip()
mapping["source_original"] = (
    mapping["source_original"].astype("string").str.strip()
)

df = df.drop(columns=["study_id"], errors="ignore")

df = df.merge(
    mapping,
    on="source_original",
    how="left",
    validate="many_to_one"
)

# -------------------------------
# 2. Define target and features
# -------------------------------

target = "Exp T(ºC)"

features = [
    "Exp P(kbar)",

    # Plagioclase major oxides
    "SiO2 (Pl)",
    "TiO2 (Pl)",
    "Al2O3 (Pl)",
    "FeO (Pl)",
    "MnO (Pl)",
    "MgO (Pl)",
    "CaO (Pl)",
    "Na2O (Pl)",
    "K2O (Pl)",

    # Hornblende major oxides
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

required = [target] + features + [
    "Source", "source_original", "study_id",
    "T (Hbl-Pl-94-ave)", "T (Hbl-Pl-21-ave)"
]

missing_columns = [c for c in required if c not in df.columns]

if missing_columns:
    raise KeyError(f"Missing required columns: {missing_columns}")

# Convert selected model variables to numeric.
for col in [target] + features + [
    "T (Hbl-Pl-94-ave)", "T (Hbl-Pl-21-ave)"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Complete-case primary dataset:
# all selected features, target, and both thermometer estimates.
analysis = df.dropna(subset=required).copy()

if analysis["study_id"].isna().any():
    raise ValueError(
        "Some eligible observations have no verified study ID."
    )

if analysis["study_id"].nunique() < 6:
    raise ValueError(
        "Too few independent study groups for the proposed split."
    )

if analysis["study_id"].nunique() < 10:
    print(
        "WARNING: Fewer than 10 study groups remain. "
        "Grouped test metrics may be highly uncertain."
    )

print("=" * 65)
print("DS-A3: TARGET, FEATURES AND GROUPED SPLIT")
print("=" * 65)
print("\nComplete-case observations:", len(analysis))
print("Independent study IDs:", analysis["study_id"].nunique())
print("Number of predictors:", len(features))

print("\nRows per study:")
print(
    analysis.groupby("study_id").size()
    .sort_values(ascending=False).to_string()
)

# -------------------------------
# 3. Create group-separated splits
# -------------------------------

# First, hold out approximately 20% of study groups for testing.
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

trainval_idx, test_idx = next(
    split1.split(
        analysis,
        groups=analysis["study_id"]
    )
)

trainval = analysis.iloc[trainval_idx].copy()
test = analysis.iloc[test_idx].copy()

# From the remaining groups, hold out approximately 25%
# for validation. This is about 20% of the original dataset.
split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=SEED + 1
)

train_idx, val_idx = next(
    split2.split(
        trainval,
        groups=trainval["study_id"]
    )
)

train = trainval.iloc[train_idx].copy()
val = trainval.iloc[val_idx].copy()

# Verify no study occurs in more than one partition.
train_groups = set(train["study_id"])
val_groups = set(val["study_id"])
test_groups = set(test["study_id"])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

# -------------------------------
# 4. Freeze and save assignments
# -------------------------------

analysis["split"] = ""

analysis.loc[train.index, "split"] = "train"
analysis.loc[val.index, "split"] = "validation"
analysis.loc[test.index, "split"] = "test"

analysis.to_csv(
    DATA_DIR / "S2_frozen_split.csv",
    index=False
)

# Save an audit of group allocation.
audit_rows = []

for name, part in [
    ("train", train),
    ("validation", val),
    ("test", test)
]:
    for study, group in part.groupby("study_id"):
        audit_rows.append({
            "split": name,
            "study_id": study,
            "n_runs": len(group)
        })

split_audit = pd.DataFrame(audit_rows)

split_audit.to_csv(
    RESULTS_DIR / "split_leakage_audit.csv",
    index=False
)

# -------------------------------
# 5. Establish the naive baseline
# -------------------------------

baseline_value = train[target].mean()

baseline_results = pd.DataFrame({
    "split": ["train", "validation", "test"],
    "n_observations": [
        len(train), len(val), len(test)
    ],
    "n_studies": [
        train["study_id"].nunique(),
        val["study_id"].nunique(),
        test["study_id"].nunique()
    ],
    "baseline_prediction_C": [
        baseline_value, baseline_value, baseline_value
    ],
    "observed_mean_temperature_C": [
        train[target].mean(),
        val[target].mean(),
        test[target].mean()
    ],
})

baseline_results.to_csv(
    RESULTS_DIR / "baseline_split_summary.csv",
    index=False
)

print("\nSPLIT SUMMARY")
print(
    baseline_results.round(3).to_string(index=False)
)

print("\nGROUP OVERLAP CHECK")
print("Train/validation overlap:", train_groups & val_groups)
print("Train/test overlap:", train_groups & test_groups)
print("Validation/test overlap:", val_groups & test_groups)

print("\nBASELINE")
print(
    "Training-set mean temperature:",
    round(baseline_value, 3), "°C"
)

print("\nSaved:")
print(DATA_DIR / "S2_frozen_split.csv")
print(RESULTS_DIR / "split_leakage_audit.csv")
print(RESULTS_DIR / "baseline_split_summary.csv")

print("\nSplit and baseline definition completed.")
