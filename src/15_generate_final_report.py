# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:53:14 2026

@author: hussa
"""


from pathlib import Path
import sys
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)

# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------
PROJECT = Path(
    r"D:\Data Science Assignment\My Projects\DS-A3"
)

FIGURES = PROJECT / "figures"
RESULTS = PROJECT / "results"
REPORT_DIR = PROJECT / "report"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = REPORT_DIR / "DS_A3_Final_Report.pdf"

# ---------------------------------------------------------
# 2. DOCUMENT STYLES
# ---------------------------------------------------------
styles = getSampleStyleSheet()

# =========================================================
# REPORT FORMATTING
# =========================================================

# Main body text: 12 pt, justified, wider line spacing
styles["BodyText"].fontName = "Helvetica"
styles["BodyText"].fontSize = 12
styles["BodyText"].leading = 18
styles["BodyText"].alignment = TA_JUSTIFY
styles["BodyText"].spaceBefore = 0
styles["BodyText"].spaceAfter = 12

# Main section headings: 14 pt, bold
if "SectionHeading" not in styles.byName:
    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"]
    ))

styles["SectionHeading"].fontName = "Helvetica-Bold"
styles["SectionHeading"].fontSize = 14
styles["SectionHeading"].leading = 19
styles["SectionHeading"].alignment = TA_LEFT
styles["SectionHeading"].spaceBefore = 14
styles["SectionHeading"].spaceAfter = 10
styles["SectionHeading"].keepWithNext = True

# Subheadings: 12 pt, bold
styles["Heading2"].fontName = "Helvetica-Bold"
styles["Heading2"].fontSize = 12
styles["Heading2"].leading = 16
styles["Heading2"].alignment = TA_LEFT
styles["Heading2"].spaceBefore = 10
styles["Heading2"].spaceAfter = 6
styles["Heading2"].keepWithNext = True

styles.add(ParagraphStyle(
    name="ReportTitle",
    parent=styles["Title"],
    fontSize=22,
    leading=28,
    alignment=TA_CENTER,
    spaceAfter=20,
))

styles.add(ParagraphStyle(
    name="FigureCaption",
    parent=styles["Normal"],
    fontSize=9,
    leading=12,
    alignment=TA_CENTER,
    spaceAfter=12,
))

styles.add(ParagraphStyle(
    name="SmallText",
    parent=styles["Normal"],
    fontSize=8,
    leading=10,
))

story = []

# ---------------------------------------------------------
# 3. HELPERS
# ---------------------------------------------------------
def add_paragraph(text, style="BodyText"):
    story.append(Paragraph(text, styles[style]))
    story.append(Spacer(1, 7))


def add_figure(filename, caption):
    path = FIGURES / filename

    if not path.is_file():
        add_paragraph(
            f"Figure unavailable: {filename}",
            "SmallText"
        )
        return

    img = Image(str(path))
    max_width = 16.5 * cm
    max_height = 19 * cm

    scale = min(
        max_width / img.imageWidth,
        max_height / img.imageHeight,
        1.0
    )

    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale

    story.append(img)
    story.append(Paragraph(
        caption, styles["FigureCaption"]
    ))


def add_csv_table(filename, title, max_rows=12):
    path = RESULTS / filename

    story.append(Paragraph(
        title, styles["Heading2"]
    ))
    story.append(Spacer(1, 5))

    if not path.is_file():
        add_paragraph(
            f"Result file not found: {filename}",
            "SmallText"
        )
        return

    df = pd.read_csv(path)

    if df.empty:
        add_paragraph("The result file is empty.")
        return

    # Limit very wide tables to the first six columns.
    df = df.iloc[:max_rows, :6].copy()

    headers = [str(c) for c in df.columns]
    rows = [headers]

    for _, row in df.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value)[:55])
        rows.append(values)

    table = Table(
        rows,
        repeatRows=1,
        colWidths=None,
        hAlign="LEFT"
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#24476B")),
        ("TEXTCOLOR", (0, 0), (-1, 0),
         colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("LEADING", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.35,
         colors.HexColor("#BBBBBB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F0F4F8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))

    story.append(table)
    story.append(Spacer(1, 12))


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(
        2 * cm, 1.1 * cm,
        "DS-A3 | Regression Baseline and Diagnostics"
    )
    canvas.drawRightString(
        A4[0] - 2 * cm, 1.1 * cm,
        f"Page {doc.page}"
    )
    canvas.restoreState()


# ---------------------------------------------------------
# 4. TITLE AND RESEARCH QUESTION
# ---------------------------------------------------------
story.append(Spacer(1, 0.8 * cm))

story.append(Paragraph(
    "DS-A3: Regression Baseline and Diagnostics",
    styles["ReportTitle"]
))

story.append(Paragraph(
    "Experimental Temperature Prediction and Plagioclase–Hornblende Geothermometer Errors",
    styles["Heading2"]
))

story.append(Spacer(1, 1 * cm))

add_paragraph(
    "<b>Research question</b><br/>"
    "How reliably do the PH94 and PH21 geothermometers "
    "reconstruct experimental temperatures, and how do "
    "their errors vary across studies and with pressure, "
    "temperature, and mineral chemistry?"
)

add_paragraph(
    "<b>Study scope</b><br/>"
    "This report contains two related but distinct analyses: "
    "a regression experiment predicting experimental "
    "temperature, and a comparison of PH94 and PH21 "
    "thermometer errors. Experimental temperatures are the "
    "reference values for assessing thermometer estimates."
)

add_paragraph(
    "<b>Data</b><br/>"
    "The audited S2 dataset contains 90 experimental runs "
    "and 60 columns. The thermometer comparison uses 65 "
    "matched runs. The regression analysis uses 65 "
    "complete-case observations and 19 predictors."
)

add_paragraph(
    "<b>Introduction</b><br/>"
    "Mineral-based geothermometers are used to estimate "
    "temperatures from mineral compositions, but their "
    "estimates may differ from experimentally measured "
    "temperatures. Quantifying these differences is "
    "important for understanding the reliability and "
    "limitations of geothermometric methods."
)

add_paragraph(
    "This study evaluates the PH94 and PH21 "
    "plagioclase–hornblende geothermometers against "
    "experimental temperatures and examines whether "
    "their errors vary across published studies and "
    "experimental conditions. In a complementary "
    "regression analysis, Linear Regression, Ridge, and "
    "Lasso are compared with a naive baseline to assess "
    "how well experimental temperatures can be predicted "
    "from pressure and mineral chemistry."
)

add_paragraph(
    "<b>Why this analysis matters</b><br/>"
    "A geothermometer's accuracy cannot be assessed only "
    "by examining a few estimated temperatures. Its errors "
    "must be compared against experimental reference "
    "temperatures across different conditions and published "
    "studies. Positive errors indicate overestimation, "
    "whereas negative errors indicate underestimation. "
    "Mean absolute error (MAE) measures the magnitude of "
    "errors, while root mean squared error (RMSE) gives "
    "greater weight to large errors."
)

add_paragraph(
    "<b>Data science approach</b><br/>"
    "The regression component provides a complementary "
    "assessment of how experimental temperature relates "
    "to pressure and measured mineral chemistry. A naive "
    "mean-temperature baseline establishes a reference "
    "level of predictive performance. Linear Regression, "
    "Ridge, and Lasso are then compared to examine whether "
    "a more complex model provides useful predictive gains. "
    "Preprocessing and model selection must be performed "
    "without using the held-out test observations."
)

add_paragraph(
    "<b>Diagnostics and interpretation</b><br/>"
    "Predictive metrics alone cannot reveal every weakness "
    "of a model. Residual plots, influence diagnostics, "
    "learning curves, heteroscedasticity testing, and "
    "error comparisons across experimental conditions "
    "help identify where predictions are less reliable. "
    "The study also examines whether thermometer errors "
    "are associated with pressure, temperature, mineral "
    "chemistry, and experimental source."
)

add_paragraph(
    "<b>Scope and limitations</b><br/>"
    "The analysis is based on a relatively small dataset "
    "of experimental runs drawn from multiple studies. "
    "Observations from the same study may share "
    "experimental characteristics, so study-group "
    "separation is important when evaluating predictive "
    "performance. Associations between mineral chemistry "
    "and prediction errors are exploratory and do not "
    "establish causal relationships. Findings from "
    "experimental data should not automatically be "
    "generalised to natural geological samples."
)

# ---------------------------------------------------------
# 5. METHODS
# ---------------------------------------------------------
story.append(Paragraph("1. Methods", styles["SectionHeading"]))


add_paragraph(
    "During model selection, the naive baseline predicts "
    "the mean experimental temperature calculated from the "
    "training partition only. For final held-out test "
    "evaluation, the Ridge model and naive baseline both "
    "use the development data, comprising the training and "
    "validation partitions. The test partition is excluded "
    "from model fitting, baseline calculation, and model "
    "selection. This explains why the baseline prediction "
    "differs between validation and final test evaluation."
)


add_paragraph(
    "Model selection was based on validation and training "
    "cross-validation rather than test-set performance. "
    "The final held-out test set was evaluated after model "
    "selection. Regression diagnostics included residual "
    "inspection, influence, learning curves, and a "
    "Breusch–Pagan test."
)

add_paragraph(
    "Thermometer errors are calculated relative to "
    "experimental temperature. Signed error indicates "
    "direction of bias; mean absolute error (MAE) measures "
    "average error magnitude. Exploratory correlations "
    "are treated as associations rather than causal effects."
)

add_csv_table(
    "split_leakage_audit.csv",
    "Split and leakage audit",
    max_rows=10
)

add_csv_table(
    "validation_model_comparison.csv",
    "Validation model comparison",
    max_rows=10
)

# ---------------------------------------------------------
# 6. REGRESSION DIAGNOSTICS
# ---------------------------------------------------------
story.append(PageBreak())
story.append(Paragraph(
    "2. Regression Diagnostics",
    styles["SectionHeading"]
))

add_figure(
    "fig02_residuals_vs_fitted.png",
    "Figure 1. Residuals versus fitted values. "
    "Inspect for systematic patterns and changing variance."
)
add_paragraph(
    "The residuals represent observed minus predicted "
    "temperatures. Positive residuals indicate underprediction, "
    "whereas negative residuals indicate overprediction. "
    "The residual pattern suggests that some high-temperature "
    "observations are overestimated by the Ridge model. "
    "However, given the small validation sample, this pattern "
    "should be treated cautiously rather than as conclusive "
    "evidence of systematic prediction bias."
)
add_figure(
    "fig13_final_test_predictions.png",
    "Figure 2. Final held-out test predictions and baseline "
    "comparison. Test results were not used to select the model."
)

add_paragraph(
    "The final held-out test results show that the Ridge model "
    "outperforms the naive mean baseline in reconstructing "
    "experimental temperatures. Ridge achieves a mean absolute "
    "error of 58.89 °C and a root mean squared error of "
    "79.04 °C, compared with 97.92 °C and 120.13 °C, "
    "respectively, for the baseline. This improvement indicates "
    "that mineral chemistry and experimental pressure provide "
    "useful predictive information beyond a constant mean "
    "temperature. Nevertheless, the test set contains only "
    "12 observations from four experimental studies, and "
    "errors remain substantial. The results therefore support "
    "the value of the regression approach for this dataset, "
    "but do not establish reliable generalisation to all "
    "experimental conditions or natural geological samples."
)

add_figure(
    "fig15_ridge_coefficients.png",
    "Figure 3. Standardized Ridge coefficients. "
    "Coefficients represent conditional associations, "
    "not causal effects."
)

add_paragraph(
    "The standardized Ridge coefficients indicate which "
    "predictors have stronger positive or negative associations "
    "with predicted experimental temperature within the fitted "
    "model. Mineral-chemistry variables, including MgO in "
    "hornblende and CaO in plagioclase, have relatively large "
    "coefficient magnitudes, while experimental pressure also "
    "contributes to the prediction. Ridge regularization reduces "
    "coefficient magnitudes to help control model complexity "
    "when predictors are correlated. However, these coefficients "
    "should not be interpreted as independent causal effects, "
    "because mineral compositions may be interrelated and the "
    "analysis uses a relatively small dataset."
)

# Additional diagnostics required by DS-A3

add_figure(
    "fig04_learning_curve.png",
    "Figure 4. Learning curve. Compare training and "
    "cross-validation performance as the training sample "
    "size increases. Interpret cautiously because the "
    "dataset is small."
)

add_paragraph(
    "The learning curve illustrates how the Ridge model's "
    "training and cross-validation errors change as more "
    "observations are used for model fitting. A narrowing "
    "gap between the two curves may indicate improved "
    "generalisation, whereas persistently high validation "
    "error suggests that predictive performance remains "
    "limited. Because the dataset contains relatively few "
    "complete observations distributed across experimental "
    "studies, the curve may be sensitive to the particular "
    "samples included at each training size. The results "
    "should therefore be interpreted as an indication of "
    "learning behaviour rather than a definitive estimate "
    "of performance on a much larger dataset."
)

add_figure(
    "fig05_influence.png",
    "Figure 5. Influence diagnostics for identifying "
    "observations that may disproportionately affect "
    "the fitted regression."
)

add_paragraph(
    "The influence diagnostics identify observations that may "
    "have a disproportionate effect on the fitted regression. "
    "Observations with high leverage or large residuals deserve "
    "closer examination because they can affect coefficient "
    "estimates and predictive performance. However, an influential "
    "observation is not necessarily an erroneous measurement; "
    "it may represent a genuine experimental condition that is "
    "underrepresented in the dataset. These cases should be "
    "checked against their source records before any decision "
    "to exclude them. Their presence also highlights the need "
    "for caution when interpreting a model fitted to a small "
    "number of experimental runs."
)

add_figure(
    "fig14_final_test_residuals.png",
    "Figure 6. Final held-out test residuals. Residuals "
    "are observed minus predicted temperatures; the test "
    "set is used for evaluation, not model selection."
)

add_paragraph(
    "The test residuals are calculated as observed minus "
    "predicted experimental temperature. Positive residuals "
    "indicate underprediction, whereas negative residuals "
    "indicate overprediction. The spread of residuals shows "
    "that the Ridge model does not predict every test observation "
    "with the same accuracy. Some errors remain substantial, "
    "so the improvement over the naive baseline should not be "
    "interpreted as precise reconstruction for every experimental "
    "run. Because the test set contains only 12 observations "
    "from four studies, these results provide a limited assessment "
    "of performance on entirely new experimental sources."
)

add_figure(
    "fig03_error_vs_pressure.png",
    "Figure 7. Regression prediction error versus "
    "experimental pressure."
)

add_paragraph(
    "<b>Heteroscedasticity assessment.</b> The "
    "Breusch-Pagan test returned an LM statistic of 2.2716 "
    "and p = 0.1318. At the conventional 0.05 significance "
    "level, this test does not provide sufficient evidence "
    "to reject the null hypothesis of constant error "
    "variance. This is not proof that the variance is "
    "constant, particularly given the small dataset."
)

add_csv_table(
    "final_test_model_comparison.csv",
    "Final held-out test metrics",
    max_rows=10
)

add_csv_table(
    "final_test_study_errors.csv",
    "Held-out test errors by study",
    max_rows=10
)

# ---------------------------------------------------------
# 7. THERMOMETER ERROR ANALYSIS
# ---------------------------------------------------------
story.append(PageBreak())
story.append(Paragraph(
    "3. PH94 and PH21 Error Analysis",
    styles["SectionHeading"]
))

add_paragraph(
    "The thermometer comparison includes 65 matched "
    "observations. The reported overall MAE was 58.43 °C "
    "for PH94 and 61.40 °C for PH21. Mean signed error "
    "was 3.20 °C for PH94 and 52.52 °C for PH21. These "
    "summaries describe this dataset and do not establish "
    "that either thermometer will perform similarly "
    "in other geological settings."
)

add_figure(
    "fig06_thermometer_error_distributions.png",
    "Figure 8. Distributions of PH94 and PH21 errors"
)

add_paragraph(
    "The error distributions compare the temperature estimates "
    "from PH94 and PH21 against experimental reference "
    "temperatures. PH94 has a mean absolute error of 58.43 °C, "
    "compared with 61.40 °C for PH21, indicating a slightly "
    "smaller average error magnitude for PH94 in these matched "
    "observations. Their mean signed errors also differ: "
    "3.20 °C for PH94 and 52.52 °C for PH21, suggesting a "
    "stronger overall tendency toward overestimation for PH21. "
    "These aggregate differences do not imply that one method "
    "performs better for every experimental source or condition."
)

add_figure(
    "fig09_source_level_MAE.png",
    "Figure 9. Source-level mean absolute errors. Some bibliographic aliases still require reconciliation."
)

add_paragraph(
    "The source-level comparison shows that thermometer error "
    "magnitude varies across the experimental sources represented "
    "in the dataset. This suggests that the overall performance "
    "metrics can conceal important differences between groups "
    "of experiments. However, the number of runs per source varies "
    "considerably, and some sources contribute only one observation. "
    "In addition, alternative bibliographic labels may refer to "
    "the same publication. Source-specific comparisons should "
    "therefore remain descriptive until the source mapping is "
    "fully reconciled and the small group sizes are considered."
)

add_figure(
    "fig10_error_correlation_heatmap.png",
    "Figure 10. Exploratory associations between selected variables and thermometer errors."
)
 
story.append(Spacer(1, 0.4 * cm))
add_paragraph(
    "The correlation analysis explores associations between "
    "mineral chemistry and the signed or absolute errors of "
    "the two geothermometers. Several plagioclase composition "
    "variables show associations with signed errors, suggesting "
    "that error direction may vary with mineral composition. "
    "However, signed error and absolute error describe different "
    "properties: the former captures bias, while the latter "
    "measures error magnitude. Mineral-chemistry variables may "
    "also be correlated with one another, and observations from "
    "the same study may not be independent. These exploratory "
    "associations do not establish causal relationships."
)
story.append(Spacer(1, 0.5 * cm))
add_csv_table(
    "thermometer_overall_metrics.csv",
    "Overall thermometer metrics",
    max_rows=10
)


story.append(PageBreak())
add_csv_table(
    "thermometer_source_error_summary.csv",
    "Thermometer errors by source",
    max_rows=12
)

add_csv_table(
    "error_feature_correlations.csv",
    "Exploratory error-feature associations",
    max_rows=12
)

# ---------------------------------------------------------
# 8. INTERPRETATION AND LIMITATIONS
# ---------------------------------------------------------
story.append(Paragraph(
    "4. Interpretation and Limitations",
    styles["SectionHeading"]
))

add_paragraph(
    "On validation data, Ridge achieved an MAE of 43.81 °C "
    "and RMSE of 55.48 °C, compared with a naive-baseline "
    "MAE of 61.22 °C and RMSE of 89.52 °C."
)

add_paragraph(
    "On the held-out test set of 12 runs from four studies, "
    "Ridge achieved an MAE of 58.89 °C, RMSE of 79.04 °C, "
    "and R² of 0.432. The naive baseline achieved an MAE "
    "of 97.92 °C, RMSE of 120.13 °C, and R² of -0.312. "
    "The small number of independent test studies limits "
    "the precision and generalizability of this comparison."
)

add_paragraph(
    "PH94 showed a smaller overall MAE than PH21 in the "
    "matched sample, while PH21 had a larger positive mean "
    "signed error. Error patterns differed across source "
    "labels. These findings are descriptive; the source "
    "mapping must be verified before making strong "
    "study-specific claims."
)

add_paragraph(
    "Important limitations include the small grouped test "
    "set, complete-case selection, correlated mineral "
    "chemistry predictors, uncertain source aliases, and "
    "the possibility that experimental conditions differ "
    "from natural geological settings. Correlation and "
    "regression coefficients do not prove causal effects."
)

# ---------------------------------------------------------
# 9. REPRODUCTION AND AI DISCLOSURE
# ---------------------------------------------------------
story.append(Paragraph(
    "5. Reproduction and Transparency",
    styles["SectionHeading"]
))

add_paragraph(
    "The project includes numbered Python scripts in src/, "
    "processed data in data_processed/, result tables in "
    "results/, and diagnostic figures in figures/. "
    "The environment audit records Python and package "
    "versions in results/requirements-lock.txt and "
    "results/environment-details.txt."
)

add_paragraph(
    "Before submission, verify that the original supplementary "
    "dataset and its version are documented, reconcile source "
    "aliases, and confirm that the scripts run from a clean "
    "environment. The environment audit records the current "
    "environment and does not alone prove bit-for-bit "
    "reproducibility."
)

add_paragraph(
    "AI assistance supported code scaffolding, debugging "
    "guidance, methodological discussion, and documentation. "
    "The accompanying AI_USE_LOG.md records the intended "
    "disclosure and verification responsibilities. The student "
    "is responsible for checking all data provenance, outputs, "
    "interpretations, and final claims."
)

# ---------------------------------------------------------
# 10. BUILD PDF
# ---------------------------------------------------------
doc = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=A4,
    rightMargin=2 * cm,
    leftMargin=2 * cm,
    topMargin=1.8 * cm,
    bottomMargin=2 * cm,
    title="DS-A3 Final Scientific Report",
    author="DS-A3 Student",
)

doc.build(
    story,
    onFirstPage=add_page_number,
    onLaterPages=add_page_number
)

print("=" * 65)
print("DS-A3 REPORT GENERATION FINISHED")
print("=" * 65)
print("PDF:", PDF_PATH)
print("PDF exists:", PDF_PATH.is_file())

if PDF_PATH.is_file():
    print("File size (KB):", round(
        PDF_PATH.stat().st_size / 1024, 1
    ))

print(
    "\nReview the PDF manually before submission. "
    "Check every figure, table, source label, and claim."
)
