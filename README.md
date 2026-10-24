# DS-A3: Regression Baseline and Diagnostics

## Research question

How reliably do the PH94 and PH21 plagioclase–hornblende geothermometers reconstruct experimental temperatures, and how do their errors vary across studies and with pressure, temperature, and mineral chemistry?

## Project overview

This project analyses experimental mineral data from the S2 supplementary dataset. It has two related objectives:

1. Evaluate PH94 and PH21 temperature estimates against experimental temperatures.
2. Build interpretable regression baselines to predict experimental temperature from experimental pressure and mineral chemistry, and investigate the limits of those predictions.

Experimental temperature is the reference value for evaluating thermometer estimates.

## Dataset

The source dataset contains 90 experimental runs and 60 columns. PH94 and PH21 temperature estimates are both available for 65 matched runs. The regression analysis retained 65 complete-case observations, 19 selected predictors, and 19 provisional study identifiers.

Original source labels are retained for traceability. Study identifiers and bibliographic aliases require verification against the original supplementary material before being treated as definitive.

## Methods

- Defined a naive baseline using the mean experimental temperature of the relevant development partition.
- Created a fixed, study-grouped train/validation/test split.
- Compared ordinary linear regression, Ridge, and Lasso.
- Used cross-validation for Ridge and Lasso penalty selection without using the test set for model selection.
- Inspected residuals, influential observations, learning curves, and heteroscedasticity.
- Compared PH94 and PH21 signed errors, absolute errors, and source-level error summaries.
- Examined exploratory associations between thermometer errors, experimental conditions, and mineral chemistry.
- Examined standardized regression coefficients to assess model complexity and interpretability.

## Main results

### Regression performance

On the validation set of eight observations from four studies, Ridge achieved an MAE of 43.81 °C and an RMSE of 55.48 °C, compared with a naive-baseline MAE of 61.22 °C.

On the held-out test set of 12 observations from four studies, Ridge achieved an MAE of 58.89 °C, an RMSE of 79.04 °C, and an R² of 0.432. The naive baseline achieved an MAE of 97.92 °C, an RMSE of 120.13 °C, and an R² of −0.312.

### Thermometer performance

Across 65 matched runs, PH94 had an MAE of 58.43 °C and a mean signed error of 3.20 °C. PH21 had an MAE of 61.40 °C and a mean signed error of 52.52 °C.

These results indicate differences in aggregate error magnitude and bias. They do not establish that either thermometer performs better under every experimental condition.

## Data splitting and leakage control

The frozen split contains:

- Training: 45 observations from 11 study groups.
- Validation: 8 observations from 4 study groups.
- Test: 12 observations from 4 study groups.

The split audit reported no study-group overlap between partitions. Study identifiers are provisional and should be reconciled against the original sources.

The final test partition was reserved for evaluation after model selection. Test results must not be used to select or tune models.

## Project structure

- `src/`: analysis and report-generation scripts.
- `data_processed/`: processed data and the frozen split.
- `results/`: audit tables, model comparisons, predictions, diagnostics, and environment records.
- `figures/`: generated diagnostic and scientific figures.
- `report/`: generated final PDF report.
- `AI_USE_LOG.md`: disclosure of AI assistance and verification responsibilities.
- `README.md`: project overview and reproduction instructions.

## Environment

The recorded environment is documented in:

- `results/requirements-lock.txt`
- `results/environment-details.txt`

These files describe the recorded environment but do not, by themselves, demonstrate successful reproduction in a clean environment.

## Reproduction guide

Run the analysis from the project root directory. Preserve the original dataset and the submitted frozen split.

1. Verify the original dataset's location, version, and provenance.
2. Configure the Python environment using `results/requirements-lock.txt` and `results/environment-details.txt`.
3. Run `02_prepare_analysis_data.py` from the project root to prepare the analysis dataset.
4. Run `03_standardize_sources.py` from the project root to standardize source labels.
5. Run `src/04_verify_study_ids.py` to audit study identifiers and source mappings.
6. Run `src/05_audit_features.py` to inspect dataset features and missingness.
7. Verify the existing frozen split using `src/06_freeze_split.py`. Do not overwrite the submitted split without documenting the change.
8. Run the modelling, diagnostic, and thermometer-error analysis scripts in their documented dependency order.
9. Run `src/11_final_test_evaluation.py` only after model selection is complete. Do not use test results to tune models.
10. Run `src/12_coefficient_interpretation.py`, `src/13_project_audit.py`, and `src/14_environment_audit.py` as appropriate.
11. Generate the final report using `src/15_generate_final_report.py`.
12. Review the generated PDF and compare its claims, tables, and figures with the saved results.

Confirm script dependencies and input/output paths before rerunning the workflow. The presence of an environment record does not establish successful reproduction in a clean environment.

## Limitations

- The held-out test set contains only 12 observations from four studies.
- Some source labels are alternate forms of bibliographic citations and require reconciliation.
- Complete-case selection may limit representativeness.
- Mineral-chemistry predictors may be correlated.
- Error correlations are exploratory associations and do not establish causal effects.
- Regression performance on experimental data does not automatically imply equivalent performance on natural geological samples.

## AI-use disclosure

AI assistance supported code scaffolding, debugging guidance, methodological discussion, and documentation. Scripts were executed in Spyder and outputs were inspected. The student remains responsible for independently verifying data provenance, bibliographic mappings, model assumptions, reported metrics, and execution steps. See `AI_USE_LOG.md` for the detailed disclosure and outstanding verification requirements.