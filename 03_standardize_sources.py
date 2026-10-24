# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:07:09 2026

@author: hussa
"""


from pathlib import Path
import pandas as pd
import re
import unicodedata

# ==========================================================
# DS-A3 STEP 3: SOURCE IDENTIFIER AUDIT
# ==========================================================

# Change this to the actual location of your S2 workbook.
DATA_FILE = Path(
    r"D:\Data Science Assignment\My Projects\DS-A2"
    r"\DS_A2_S2_stage1\DS_A2_S2_stage1"
    r"\DS-A2-Inference-Uncertainty\data\S2_source.xlsx"
)

PROJECT_DIR = Path(
    r"D:\Data Science Assignment\My Projects\DS-A3"
)
RESULTS_DIR = PROJECT_DIR / "results"
DATA_DIR = PROJECT_DIR / "data_processed"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("DS-A3: SOURCE IDENTIFIER AUDIT")
print("=" * 65)

if not DATA_FILE.is_file():
    raise FileNotFoundError(
        f"S2 workbook not found:\n{DATA_FILE}\n"
        "Update DATA_FILE to the correct location."
    )

# Read original workbook and retain experimental observations.
raw = pd.read_excel(DATA_FILE, sheet_name="Table S2")
raw = raw.dropna(axis=1, how="all").copy()

raw["Exp T(ºC)"] = pd.to_numeric(
    raw["Exp T(ºC)"], errors="coerce"
)
raw["Exp P(kbar)"] = pd.to_numeric(
    raw["Exp P(kbar)"], errors="coerce"
)

df = raw.dropna(
    subset=["Exp T(ºC)", "Exp P(kbar)"]
).copy()

if "Source" not in df.columns:
    raise KeyError("The expected 'Source' column is missing.")

# Preserve original source labels.
df["source_original"] = df["Source"].astype("string").str.strip()

# Conservative text normalization for inspection only.
def clean_label(value):
    if pd.isna(value) or not str(value).strip():
        return pd.NA

    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["source_clean"] = df["source_original"].map(clean_label)

# Extract a candidate first-author/year identifier.
# This is a REVIEW AID, not a verified publication identity.
def candidate_id(value):
    if pd.isna(value):
        return "REVIEW_MISSING_SOURCE"

    text = str(value).strip()

    # Find a four-digit year, optionally inside parentheses.
    match = re.search(r"\b(19|20)\d{2}\b", text)
    if not match:
        return "REVIEW_NO_YEAR"

    year = match.group(0)
    author_part = text[:match.start()].strip()

    # Remove initials and punctuation when possible.
    author_part = re.sub(
        r"\b[A-Z]\.?(?:\s*[A-Z]\.?)?\b", " ", author_part
    )
    author_part = re.sub(r"[^A-Za-zÀ-ÿ\s-]", " ", author_part)
    words = author_part.split()

    if not words:
        return f"REVIEW_UNKNOWN_AUTHOR_{year}"

    # First surname + year only; collisions must be reviewed.
    first_author = words[0].lower()
    return f"{first_author}_{year}"

df["candidate_study_id"] = df["source_clean"].map(candidate_id)

# Produce a review table of source labels and candidate IDs.
source_review = (
    df.groupby(
        ["source_original", "source_clean", "candidate_study_id"],
        dropna=False
    )
    .size()
    .reset_index(name="n_runs")
    .sort_values(
        ["candidate_study_id", "source_original"],
        na_position="last"
    )
)

# Flag candidate IDs shared by different original labels.
label_counts = source_review.groupby(
    "candidate_study_id", dropna=False
)["source_original"].nunique()

source_review["needs_review"] = source_review[
    "candidate_study_id"
].map(label_counts).gt(1)

# Save audit files without changing the original workbook.
source_review.to_csv(
    RESULTS_DIR / "source_mapping_review.csv",
    index=False
)

df.to_csv(
    DATA_DIR / "S2_with_candidate_source_ids.csv",
    index=False
)

print("\nEXPERIMENTAL OBSERVATIONS:", len(df))
print("Original source labels:", df["source_original"].nunique())
print("Candidate study IDs:", df["candidate_study_id"].nunique())
print(
    "Candidate IDs requiring review:",
    int(source_review.loc[
        source_review["needs_review"], "candidate_study_id"
    ].nunique())
)

print("\nSOURCE MAPPING REVIEW:")
print(source_review.to_string(index=False))

print("\nSaved:")
print(RESULTS_DIR / "source_mapping_review.csv")
print(DATA_DIR / "S2_with_candidate_source_ids.csv")

print("\nIMPORTANT:")
print("Candidate IDs are provisional and must be checked.")
print("Do not use them for the final train/test split yet.")
