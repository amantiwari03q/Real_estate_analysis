"""
Step 1 - Data preprocessing
Usage:  python 01_data_preprocessing.py [path/to/india_housing_prices.csv]

Outputs
  data/cleaned_housing.csv           cleaned + engineered dataset
  models/reference.json              statistics needed to engineer features later (app)
  eda_outputs/preprocessing_report.md  what was fixed (audit trail)
"""
import json
import sys

import numpy as np
import pandas as pd

import common as C

raw_path = sys.argv[1] if len(sys.argv) > 1 else C.RAW_CSV
df = pd.read_csv(raw_path)
df.columns = df.columns.str.strip()
log = []          # (step, detail) rows for the report


def note(step, detail):
    log.append((step, detail))
    print(f"[{step}] {detail}")


note("Load", f"{len(df):,} rows x {df.shape[1]} columns from {raw_path}")

# ---- 1. Standardise text & blanks ----------------------------------------
obj_cols = df.select_dtypes(include=["object", "string"]).columns
for c in obj_cols:
    df[c] = df[c].astype("string").str.strip()
    df[c] = df[c].replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "null": pd.NA,
                           "NULL": pd.NA, "nan": pd.NA, "?": pd.NA})
    df[c] = df[c].astype(object).where(df[c].notna(), np.nan)

for c in C.NUMERIC_COLS_RAW:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---- 2. Duplicates ---------------------------------------------------------
before = len(df)
df = df.drop_duplicates()
note("Duplicates", f"removed {before - len(df):,} fully duplicated rows")
if "ID" in df.columns:
    before = len(df)
    df = df.drop_duplicates(subset="ID", keep="first")
    note("Duplicates", f"removed {before - len(df):,} rows with repeated ID")

# ---- 3. Missing values ------------------------------------------------------
missing = df.isna().sum()
missing = missing[missing > 0]
note("Missing (before)", ", ".join(f"{k}={v}" for k, v in missing.items()) or "none")

# Rows that lack the core fields cannot be repaired sensibly -> drop
core = ["City", "Price_in_Lakhs", "Size_in_SqFt"]
before = len(df)
df = df.dropna(subset=core)
note("Missing", f"dropped {before - len(df):,} rows missing City / Price / Size")

num_fill = [c for c in C.NUMERIC_COLS_RAW if c in df.columns and df[c].isna().any()]
for c in num_fill:                                   # median within city, else global
    df[c] = df[c].fillna(df.groupby("City")[c].transform("median"))
    df[c] = df[c].fillna(df[c].median())
cat_fill = [c for c in C.CATEGORICAL_COLS_RAW if c in df.columns and df[c].isna().any()]
for c in cat_fill:
    if c == "Amenities":
        df[c] = df[c].fillna("None")
    elif c == "Locality":
        df[c] = df[c].fillna("Unknown")
    else:                                            # mode within city, else global
        mode_by_city = df.groupby("City")[c].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
        df[c] = df[c].fillna(df["City"].map(mode_by_city))
        df[c] = df[c].fillna(df[c].mode().iloc[0])
note("Missing (imputed)", f"numeric (median by city): {num_fill or 'none'}; "
                          f"categorical (mode by city): {cat_fill or 'none'}")

# ---- 4. Logical validity -----------------------------------------------------
before = len(df)
valid = (
    (df["Price_in_Lakhs"] > 0) & (df["Size_in_SqFt"] > 0) & (df["BHK"] > 0)
    & df["Year_Built"].between(1800, C.CURRENT_YEAR)
)
df = df[valid].copy()
note("Validity", f"dropped {before - len(df):,} rows with non-positive price/size/BHK or impossible Year_Built")

bad_floor = df["Floor_No"] > df["Total_Floors"]
df.loc[bad_floor, "Total_Floors"] = df.loc[bad_floor, "Floor_No"]
note("Validity", f"fixed {bad_floor.sum():,} rows where Floor_No > Total_Floors (Total_Floors raised)")

int_cols = ["BHK", "Year_Built", "Floor_No", "Total_Floors", "Nearby_Schools",
            "Nearby_Hospitals", "Parking_Space"]
df[int_cols] = df[int_cols].round().astype(int)

# ---- 5. Outlier detection (IQR rule) - flagged, never silently deleted ------
df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]


def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


outlier_rows = []
for col, flag in [("Price_in_Lakhs", "Outlier_Price"), ("Size_in_SqFt", "Outlier_Size"),
                  ("Price_per_SqFt", "Outlier_PricePerSqFt")]:
    lo, hi = iqr_bounds(df[col])
    df[flag] = ((df[col] < lo) | (df[col] > hi)).astype(int)
    outlier_rows.append((col, round(lo, 2), round(hi, 2), int(df[flag].sum()),
                         round(100 * df[flag].mean(), 2)))
    note("Outliers", f"{col}: IQR bounds [{lo:,.2f}, {hi:,.2f}] -> {df[flag].sum():,} flagged "
                     f"({100 * df[flag].mean():.2f}%)")

# Extreme errors (beyond 3x IQR on Size / ppsf) are removed - they are almost
# certainly data-entry errors; milder outliers stay (flagged) because they are
# real luxury / large properties that investors care about.
before = len(df)
keep = pd.Series(True, index=df.index)
for col in ["Size_in_SqFt", "Price_per_SqFt"]:
    lo, hi = iqr_bounds(df[col], k=3.0)
    keep &= df[col].between(max(lo, 0), hi)
df = df[keep].copy()
note("Outliers", f"removed {before - len(df):,} extreme rows (beyond 3 x IQR on size / price-per-sqft)")

# ---- 6. Feature engineering & targets ----------------------------------------
ref = C.build_reference(df)
df = C.add_features(df, ref)
df = C.add_targets(df)
with open(C.REFERENCE_JSON, "w") as f:
    json.dump(ref, f, indent=2)

balance = df["Good_Investment"].mean()
note("Features", "added Price_per_SqFt (Rs), Amenities_Count, Transport_Score, Security_Flag, "
                 "Under_Construction, School/Hospital Density Score, Infrastructure_Score, "
                 "Price_vs_City_Median, Floor_Ratio")
note("Targets", f"Good_Investment (annual growth >= {C.GOOD_INVESTMENT_RATE_THRESHOLD:.0%}): "
                f"{balance:.1%} positive; Future_Price_5Y_Lakhs created")
if balance < 0.10 or balance > 0.90:
    print(f"WARNING: class balance is {balance:.1%}. Consider changing "
          "GOOD_INVESTMENT_RATE_THRESHOLD in common.py so both classes are well represented.")

df.to_csv(C.CLEAN_CSV, index=False)
note("Save", f"{len(df):,} rows x {df.shape[1]} columns -> {C.CLEAN_CSV}")

# ---- 7. Audit report -----------------------------------------------------------
with open(C.EDA_DIR / "preprocessing_report.md", "w") as f:
    f.write("# Preprocessing report\n\n| Step | Detail |\n|---|---|\n")
    for s, d in log:
        f.write(f"| {s} | {d} |\n")
    f.write("\n## Outlier summary (IQR rule, 1.5 x IQR)\n\n"
            "| Column | Lower bound | Upper bound | Outliers | % |\n|---|---|---|---|---|\n")
    for r in outlier_rows:
        f.write("| " + " | ".join(map(str, r)) + " |\n")
    f.write("\nNumeric scaling (StandardScaler) and categorical encoding (One-Hot) are "
            "applied inside the model pipeline in `03_train_models.py`, so the cleaned CSV stays "
            "human-readable and test data never leaks into the scaler.\n")
