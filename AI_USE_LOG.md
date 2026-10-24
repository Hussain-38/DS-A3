# DS-A3 AI-Use Log

## 1\. Purpose of AI assistance

ChatGPT was used as an academic and programming assistant during the DS-A3 project on regression baselines and diagnostics. The research question concerns the reliability of the PH94 and PH21 plagioclase–hornblende geothermometers in reconstructing experimental temperatures.

## 2\. Tasks supported by AI

* Developing and refining the research question.
* Planning the data-cleaning, source-mapping, and missingness audits.
* Providing Python code scaffolding for regression baselines, Linear Regression, Ridge, and Lasso.
* Advising on study-grouped data splitting and leakage checks.
* Developing diagnostic analyses for residuals, influence, learning curves, and heteroscedasticity.
* Supporting PH94/PH21 error analysis and exploratory associations with pressure, temperature, and mineral chemistry.
* Interpreting model metrics and standardized coefficients.
* Preparing documentation and guidance for reproducibility.

## 3\. Advice accepted and verification

Code and methodological suggestions were incorporated where appropriate and executed in the Spyder environment. Console outputs were reviewed to verify that the reported scripts completed and that expected files were generated.

The split audit reported no overlap between study groups across training, validation, and test partitions. The regression test evaluation was performed separately from validation-based model selection.

Reported model metrics were taken from the executed scripts rather than invented for the report.

## 4\. Critical evaluation and limitations

AI-generated code and interpretations are not assumed to be correct merely because execution succeeds. Data provenance, variable definitions, bibliographic source mappings, model assumptions, and generated figures require independent review.

The source-mapping audit identified bibliographic aliases that need reconciliation. The final test set contains only 12 observations from four studies, limiting confidence in generalization. Exploratory correlations are associations and do not establish causal effects.

## 5\. Rejected or modified advice

AI-generated suggestions were reviewed against the assignment requirements, the existing code, and the observed analytical results. Some guidance was adapted during implementation, including changes to the report structure, figure captions, explanatory text, and formatting. Methodological decisions were retained only where they were consistent with the intended evaluation protocol and the available evidence. A complete contemporaneous record of every rejected suggestion was not maintained. Therefore, specific rejected suggestions and their reasons are not reconstructed retrospectively. Any remaining uncertainty about the analysis, code, or interpretation is documented in the verification requirements below.

## 6\. Independent verification still required

* Verify source citations and study identifiers against the original supplementary material.
* Confirm the original dataset version and document its provenance.
* Inspect diagnostic plots and validate their interpretations.
* Verify script execution order and environment dependencies.
* Confirm that reported metrics match the saved result files.
* Re-run the complete workflow from a clean environment where feasible.
* Review the final PDF, repository contents, and reproduction instructions.

## 7\. Disclosure statement

AI assistance supported coding, methodological discussion, debugging guidance, and documentation. The student remains responsible for verifying the analysis, critically evaluating the advice, accurately reporting the evidence, and ensuring that the submitted work complies with course requirements.

