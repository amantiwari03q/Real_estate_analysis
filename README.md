# Real Estate Investment Advisor
Predict whether a property is a **Good Investment** (classification) and its **price after 5 years** (regression),
with EDA, MLflow tracking and a Streamlit app.

## Run order
```bash
pip install -r requirements.txt
# put india_housing_prices.csv inside the data/ folder, then:
python 01_data_preprocessing.py      # cleaning + feature engineering  -> data/cleaned_housing.csv
python 02_eda_analysis.py            # all 20 EDA questions            -> eda_outputs/
python 03_train_models.py            # models + MLflow                 -> models/
python 04_build_documentation.py     # Word report with real findings  -> Project_Documentation.docx
streamlit run app.py                 # interactive app
mlflow ui --backend-store-uri sqlite:///mlflow.db   # experiment tracking UI
```

## Files
| File | Purpose |
|---|---|
| `common.py` | Shared feature engineering + target rules (used by pipeline AND app) |
| `01_data_preprocessing.py` | Duplicates, missing values, validity, outlier flags, features, targets |
| `02_eda_analysis.py` | Q1-Q20 charts/tables + `EDA_Findings.md` (numbers generated from data) |
| `03_train_models.py` | LogReg/RF/GradientBoosting(+XGBoost) models, metrics, feature importance, MLflow |
| `04_build_documentation.py` | Builds the Word documentation from real outputs |
| `app.py` | Streamlit app |

## Target definitions (read this)
The data has one price snapshot and no price history, so targets come from a documented rule in
`common.add_targets()`: annual growth = 4% base + infrastructure + under-construction uplift + valuation vs city
median + security - age penalty; `Future_Price_5Y = Price x (1+growth)^5`; `Good_Investment = growth >= 8%`.
Change `GOOD_INVESTMENT_RATE_THRESHOLD` in `common.py` if the class balance printed by script 01 is extreme.
Models therefore learn this rule; replace the rule with real resale data if you obtain it.

## Notes
- The brief mentions crime rate; the dataset has no such column, so that analysis is flagged as not possible.
- Without XGBoost/MLflow installed, scripts fall back to scikit-learn boosting and CSV run logging.
