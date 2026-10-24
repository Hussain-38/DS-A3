# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:12:56 2026

@author: hussa
"""


from pathlib import Path
import pandas as pd

# ======================================================
# DS-A3 STEP 4: VERIFY STUDY IDENTIFIERS
# ======================================================

PROJECT_DIR = Path(
    r"D:\Data Science Assignment\My Projects\DS-A3"
)

INPUT_FILE = (
    PROJECT_DIR
    / "data_processed"
    / "S2_with_candidate_source_ids.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "verified_source_mapping.csv"
)

if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

# Preserve original source labels.
df["source_original"] = df["source_original"].astype("string").str.strip()

# Explicitly document the aliases identified in the audit.
# These mappings are provisional until bibliographic details
# are checked against the source paper.
aliases = {
    "Caricchi et al. (2006)": "caricchi_2006",
    (
        "Caricchi, L., Ulmer, P., Peccerillo, A., 2006. "
        "A high-pressure experimental study on the evolution "
        "of the silicic magmatism of the Main Ethiopian Rift. "
        "Lithos 91, 46–58."
    ): "caricchi_2006",

    "Ernst and Liu (1998)": "ernst_liu_1998",
    (
        "Ernst, W.G., Liu, J., 1998. Experimental phase-equilibrium "
        "study of Al- and Ti-contents of calcic amphibole in MORB—"
        "A semiquantitative thermobarometer. American Mineralogist "
        "83, 952–969."
    ): "ernst_liu_1998",

    "Prouteau and Scaillet (2003)": "prouteau_scaillet_2003",
    (
        "Prouteau, G., Scaillet, B., 2003. Experimental constraints "
        "on the origin of the 1991 Pinatubo dacite. Journral of "
        "Petrology 44, 2203–2241."
    ): "prouteau_scaillet_2003",
}

# Start with the existing provisional identifiers.
df["study_id"] = df["candidate_study_id"].astype("string")

# Apply explicit aliases only to the reviewed labels.
for label, study_id in aliases.items():
    mask = df["source_original"].eq(label)
    if mask.any():
        df.loc[mask, "study_id"] = study_id

# Create a compact mapping audit.
mapping = (
    df.groupby(
        ["source_original", "study_id"],
        dropna=False
    )
    .size()
    .reset_index(name="n_runs")
    .sort_values(["study_id", "source_original"])
)

# Check whether multiple original labels share an identifier.
labels_per_id = mapping.groupby("study_id")[
    "source_original"
].nunique()

mapping["multiple_source_labels"] = (
    mapping["study_id"].map(labels_per_id) > 1
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
mapping.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("DS-A3: VERIFIED SOURCE MAPPING AUDIT")
print("=" * 60)

print("\nExperimental runs:", len(df))
print("Original source labels:", df["source_original"].nunique())
print("Study identifiers:", df["study_id"].nunique())

print("\nRuns per study identifier:")
print(
    df.groupby("study_id", dropna=False)
      .size()
      .sort_values(ascending=False)
      .to_string()
)

print("\nLabels sharing study identifiers:")
print(
    mapping.loc[
        mapping["multiple_source_labels"]
    ].to_string(index=False)
)

print("\nSaved mapping audit:")
print(OUTPUT_FILE)

print(
    "\nImportant: confirm the bibliographic aliases "
    "before using study_id for grouped validation."
)
