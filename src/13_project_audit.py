# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:30:59 2026

@author: hussa
"""


from pathlib import Path

PROJECT_DIR = Path(
    r"D:\Data Science Assignment\My Projects\DS-A3"
)

print("=" * 65)
print("DS-A3 PROJECT FILE AUDIT")
print("=" * 65)

required_files = [
    "src/04_verify_study_ids.py",
    "src/05_audit_features.py",
    "src/06_freeze_split.py",
    "src/07_fit_regression_models.py",
    "src/08_diagnostics.py",
    "src/09_thermometer_error_analysis.py",
    "src/10_error_driver_analysis.py",
    "src/11_final_test_evaluation.py",
    "src/12_coefficient_interpretation.py",
    "data_processed/S2_with_candidate_source_ids.csv",
    "data_processed/S2_frozen_split.csv",
    "results/verified_source_mapping.csv",
    "results/feature_missingness_audit.csv",
    "results/split_leakage_audit.csv",
    "results/baseline_split_summary.csv",
    "results/validation_model_comparison.csv",
    "results/validation_predictions.csv",
    "results/model_configuration.json",
    "results/validation_error_cases.csv",
    "results/training_influence_diagnostics.csv",
    "results/learning_curve_results.csv",
    "results/heteroscedasticity_test.txt",
    "results/thermometer_overall_metrics.csv",
    "results/thermometer_source_error_summary.csv",
    "results/error_feature_correlations.csv",
    "results/final_test_model_comparison.csv",
    "results/final_test_predictions.csv",
    "results/final_test_protocol.json",
    "results/standardized_model_coefficients.csv",
    "results/model_complexity_summary.csv",
]

missing = []
present = []

for relative_path in required_files:
    path = PROJECT_DIR / relative_path

    if path.is_file():
        present.append(relative_path)
        print("[FOUND]  ", relative_path)
    else:
        missing.append(relative_path)
        print("[MISSING]", relative_path)

print("\nSummary")
print("Files found:", len(present))
print("Files missing:", len(missing))

print("\nMissing files:")
for item in missing:
    print("-", item)

print("\nFigure files:")
figure_dir = PROJECT_DIR / "figures"

if figure_dir.is_dir():
    for path in sorted(figure_dir.glob("*.png")):
        print("-", path.name)
else:
    print("Figures folder not found.")

print("\nProject audit completed.")
