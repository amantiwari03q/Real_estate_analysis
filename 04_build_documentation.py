"""
Step 4 - Build Project_Documentation.docx (methodology, findings, use cases)
Usage:  python 04_build_documentation.py
Run AFTER scripts 01-03: findings, charts and metrics are pulled from their real outputs.
"""
import json
import re

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt

import common as C

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)
meta = json.load(open(C.APP_META_JSON))
df = pd.read_csv(C.CLEAN_CSV)


def para(t, style=None):
    return doc.add_paragraph(t, style=style)


def bullets(items):
    for i in items:
        para(i, "List Bullet")


def df_table(t):
    tb = doc.add_table(rows=1, cols=len(t.columns))
    tb.style = "Light Grid Accent 1"
    for i, c in enumerate(t.columns):
        tb.rows[0].cells[i].text = str(c)
    for _, r in t.iterrows():
        cells = tb.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = f"{v:,.4f}" if isinstance(v, float) else str(v)


def add_markup(text):
    """Render **bold** inline markdown into a paragraph."""
    p = doc.add_paragraph(style="List Bullet")
    for i, part in enumerate(re.split(r"\*\*", text)):
        r = p.add_run(part.replace("`", ""))
        r.bold = i % 2 == 1


doc.add_heading("Real Estate Investment Advisor", 0)
para("Predicting Property Profitability & Future Value - Project Documentation")

doc.add_heading("1. Problem statement & objectives", 1)
para("Build a machine-learning application that helps investors (a) classify whether a property is a "
     "'Good Investment' and (b) estimate its price after 5 years, delivered through an interactive "
     "Streamlit app with MLflow experiment tracking.")

doc.add_heading("2. Business use cases", 1)
bullets(["Give investors a data-backed screen of long-term return potential.",
         "Help buyers compare high-return properties across cities and localities.",
         "Let real-estate platforms automate first-pass investment analysis for new listings.",
         "Build customer trust by showing confidence scores and the factors behind each verdict."])

doc.add_heading("3. Data", 1)
para(f"Source: india_housing_prices.csv. After cleaning: {len(df):,} properties, {df['City'].nunique()} cities, "
     f"{df['State'].nunique()} states. See eda_outputs/preprocessing_report.md for the full audit trail.")

doc.add_heading("4. Methodology", 1)
doc.add_heading("4.1 Preprocessing", 2)
bullets(["Trimmed text, standardised blanks, coerced numeric types.",
         "Removed exact duplicates and repeated IDs.",
         "Imputed numeric gaps with the city median and categorical gaps with the city mode; dropped rows missing City, Price or Size.",
         "Removed logically impossible rows (non-positive price/size/BHK, impossible Year_Built); repaired Floor_No > Total_Floors.",
         "Outliers: IQR rule flags (Outlier_* columns); only extreme errors (> 3 x IQR) were removed.",
         "Scaling (StandardScaler) and one-hot encoding happen inside the model pipeline to avoid leakage."])
doc.add_heading("4.2 Feature engineering", 2)
bullets(["Price_per_SqFt (Rs), Price_vs_City_Median, Floor_Ratio, Age_of_Property.",
         "School_Density_Score, Hospital_Density_Score (0-10) and Infrastructure_Score (0-100, weighted blend of "
         "schools, hospitals, public transport, parking and amenities).",
         "Amenities_Count, Transport_Score, Security_Flag, Under_Construction."])
doc.add_heading("4.3 Target definition (important assumption)", 2)
para("The dataset contains a single price snapshot, not price history, so the two targets are derived from a "
     "transparent domain rule: Annual growth = 4% base + up to 4% from infrastructure + 1.5% if under construction "
     "+ up to +-1.5% for under/over-valuation versus the city median + 0.2% for security - 0.03% per year of "
     "building age (max 1.2%). Future_Price_5Y = Price x (1 + growth)^5. Good_Investment = growth >= "
     f"{C.GOOD_INVESTMENT_RATE_THRESHOLD:.0%} p.a. Because the targets are rule-based, high model scores show that the "
     "models recover the rule; with real historical resale data the same pipeline can be retrained unchanged.")
doc.add_heading("4.4 Models & tracking", 2)
para("Classification: Logistic Regression, Random Forest, Gradient Boosting (XGBoost when installed). "
     "Regression: Ridge, Random Forest, Gradient Boosting (XGBoost when installed). 80/20 stratified split; "
     "every run is logged to MLflow (parameters, metrics, model) and to experiment_log.csv.")

doc.add_heading("5. Exploratory data analysis - findings", 1)
for line in open(C.EDA_DIR / "EDA_Findings.md").read().splitlines():
    if line.startswith("# "):
        continue
    if line.startswith("## "):
        doc.add_heading(line[3:], 2)
    elif line.startswith("!["):
        img = C.EDA_DIR / re.search(r"\((.*?)\)", line).group(1)
        if img.exists():
            doc.add_picture(str(img), width=Inches(5.8))
    elif line.startswith("- "):
        add_markup(line[2:])
    elif line.strip():
        para(re.sub(r"\*\*", "", line))

doc.add_heading("6. Model results (held-out test set)", 1)
doc.add_heading("Classification - Good Investment", 2)
df_table(pd.read_csv(C.MODELS_DIR / "classification_metrics.csv"))
para(f"Best model: {meta['best_classifier']}.")
doc.add_heading("Regression - price after 5 years (Rs lakhs)", 2)
df_table(pd.read_csv(C.MODELS_DIR / "regression_metrics.csv"))
para(f"Best model: {meta['best_regressor']}.")
doc.add_heading("Top drivers (permutation importance)", 2)
for tag, ttl in [("clf", "Classifier"), ("reg", "Regressor")]:
    imp = pd.read_csv(C.MODELS_DIR / f"feature_importance_{tag}.csv").head(8)[["feature", "importance_pct"]]
    para(ttl + ":")
    df_table(imp)

doc.add_heading("7. Streamlit application", 1)
bullets(["Sidebar filters: state, city, price range, area range, BHK, property type.",
         "Predict tab: property form -> Good Investment verdict, 5-year price with likely range, confidence score.",
         "Market insights: city x property-type heatmap, price trends, city comparisons.",
         "Matching listings: filtered properties ranked by investment probability, CSV download.",
         "Model performance: metric tables and feature importance."])

doc.add_heading("8. Limitations & next steps", 1)
bullets(["Targets are rule-derived (no historical prices); retrain on real resale data when available.",
         "No crime-rate column exists in the data, so crime impact could not be measured.",
         "Year_Built is used as a proxy time axis for locality trends.",
         "Locality is excluded from the models (very high cardinality); City carries location effects.",
         "Next: add real price history, city-level macro indicators, SHAP explanations, cloud deployment."])

out = C.ROOT / "Project_Documentation.docx"
doc.save(out)
print("Saved", out)
