# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 09:52:32 2026

@author: hussa
"""


from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# STEP 1: Locate and inspect the S2 Excel workbook
# --------------------------------------------------

# Change this path if your workbook is stored elsewhere.
DATA_FILE = Path(
    r"D:\Data Science Assignment\My Projects\DS-A2"
    r"\DS_A2_S2_stage1\DS_A2_S2_stage1"
    r"\DS-A2-Inference-Uncertainty\data\S2_source.xlsx"
)

print("=" * 65)
print("DS-A3: S2 DATASET INSPECTION")
print("=" * 65)

# 1. Verify the file exists
if not DATA_FILE.is_file():
    print("\nWorkbook not found at:")
    print(DATA_FILE)
    print("\nPlease correct DATA_FILE to the actual Excel path.")
else:
    print("\nWorkbook:", DATA_FILE.name)
    print("File size (MB):", round(DATA_FILE.stat().st_size / 1e6, 3))

    # 2. List all worksheet names
    excel_file = pd.ExcelFile(DATA_FILE)
    print("\nWORKSHEETS:")
    for sheet in excel_file.sheet_names:
        print("-", sheet)

    # 3. Load each sheet and inspect its structure
    for sheet in excel_file.sheet_names:
        df = pd.read_excel(DATA_FILE, sheet_name=sheet)

        print("\n" + "=" * 65)
        print("SHEET:", sheet)
        print("=" * 65)

        print("\nShape (rows, columns):", df.shape)

        print("\nColumn names:")
        for col in df.columns:
            print("-", repr(col))

        print("\nFirst five rows:")
        print(df.head().to_string(index=False))

        print("\nData types:")
        print(df.dtypes.to_string())

        print("\nMissing values per column:")
        missing = df.isna().sum()
        print(missing[missing > 0].to_string()
              if missing.any() else "No missing values detected.")

        print("\nDuplicate rows:", df.duplicated().sum())

        print("\nNumeric summary:")
        print(df.describe().round(3).to_string())

print("\nInspection finished.")
