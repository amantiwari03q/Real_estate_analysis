"""
common.py - shared constants and feature-engineering logic.

Used by BOTH the offline pipeline (01..04 scripts) and the Streamlit app so a
property typed into the app is transformed exactly like the training data.
"""
from pathlib import Path
import datetime
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "india_housing_prices.csv"
CLEAN_CSV = DATA_DIR / "cleaned_housing.csv"
MODELS_DIR = ROOT / "models"
EDA_DIR = ROOT / "eda_outputs"
CHARTS_DIR = EDA_DIR / "charts"
TABLES_DIR = EDA_DIR / "tables"
REFERENCE_JSON = MODELS_DIR / "reference.json"
APP_META_JSON = MODELS_DIR / "app_metadata.json"

for _d in (DATA_DIR, MODELS_DIR, CHARTS_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CURRENT_YEAR = datetime.date.today().year
HORIZON_YEARS = 5
RANDOM_STATE = 42

# ---- Business rule for the "Good Investment" label -------------------------
# A property is a Good Investment when its modelled annual appreciation rate
# is at least this value (8 % p.a. ~ 47 % growth over 5 years).
GOOD_INVESTMENT_RATE_THRESHOLD = 0.08

TRANSPORT_MAP = {"low": 1, "medium": 2, "high": 3}

NUMERIC_COLS_RAW = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
    "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools",
    "Nearby_Hospitals", "Parking_Space",
]
CATEGORICAL_COLS_RAW = [
    "State", "City", "Locality", "Property_Type", "Furnished_Status",
    "Public_Transport_Accessibility", "Security", "Amenities", "Facing",
    "Owner_Type", "Availability_Status",
]


def count_amenities(value) -> int:
    if pd.isna(value):
        return 0
    items = [a.strip() for a in str(value).split(",")]
    return len([a for a in items if a and a.lower() not in ("none", "nan", "na")])


def build_reference(df: pd.DataFrame) -> dict:
    """Statistics learned from the cleaned data that feature engineering needs."""
    ppsf = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
    tmp = df.assign(_ppsf=ppsf)
    return {
        "global_median_ppsf": float(ppsf.median()),
        "city_median_ppsf": tmp.groupby("City")["_ppsf"].median().to_dict(),
        "max_schools": float(df["Nearby_Schools"].max()),
        "max_hospitals": float(df["Nearby_Hospitals"].max()),
        "max_parking": float(df["Parking_Space"].max()),
        "max_amenities": float(df["Amenities"].map(count_amenities).max()),
    }


def load_reference() -> dict:
    with open(REFERENCE_JSON) as f:
        return json.load(f)


def add_features(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Derive model features. Works on a full dataset or one app input row."""
    df = df.copy()
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]      # Rs / sq ft
    df["Age_of_Property"] = (CURRENT_YEAR - df["Year_Built"]).clip(lower=0)

    df["Amenities_Count"] = df["Amenities"].map(count_amenities)
    df["Transport_Score"] = (
        df["Public_Transport_Accessibility"].astype(str).str.strip().str.lower()
        .map(TRANSPORT_MAP).fillna(2).astype(int)
    )
    df["Security_Flag"] = (
        ~df["Security"].astype(str).str.strip().str.lower()
        .isin(["no", "none", "0", "nan", "", "unknown"])
    ).astype(int)
    df["Under_Construction"] = (
        df["Availability_Status"].astype(str).str.lower().str.contains("construction")
    ).astype(int)

    sch = (df["Nearby_Schools"] / max(ref["max_schools"], 1)).clip(0, 1)
    hos = (df["Nearby_Hospitals"] / max(ref["max_hospitals"], 1)).clip(0, 1)
    par = (df["Parking_Space"] / max(ref["max_parking"], 1)).clip(0, 1)
    ame = (df["Amenities_Count"] / max(ref["max_amenities"], 1)).clip(0, 1)
    trn = (df["Transport_Score"] - 1) / 2

    df["School_Density_Score"] = (sch * 10).round(2)
    df["Hospital_Density_Score"] = (hos * 10).round(2)
    df["Infrastructure_Score"] = (
        100 * (0.25 * sch + 0.25 * hos + 0.30 * trn + 0.10 * par + 0.10 * ame)
    ).round(2)

    city_med = df["City"].map(ref["city_median_ppsf"]).fillna(ref["global_median_ppsf"])
    df["City_Median_PPSF"] = city_med
    df["Price_vs_City_Median"] = (df["Price_per_SqFt"] / city_med).round(4)
    df["Floor_Ratio"] = np.where(df["Total_Floors"] > 0,
                                 df["Floor_No"] / df["Total_Floors"], 0).round(4)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the two supervised targets (the raw data has no price history, so
    they are derived from transparent domain rules - see README):

      Annual_Growth_Rate = 4 % base
                         + up to 4 %   from Infrastructure_Score
                         + 1.5 %       if under construction (pre-completion uplift)
                         + up to +-1.5 % for under/over-valuation vs city median
                         + 0.2 %       if the project has security
                         - up to 1.2 % for building age (0.03 % per year, max 40 yrs)

      Future_Price_5Y_Lakhs = Price_in_Lakhs * (1 + rate) ** 5   (regression)
      Good_Investment       = rate >= GOOD_INVESTMENT_RATE_THRESHOLD (classification)
    """
    df = df.copy()
    undervaluation = (1 - df["Price_vs_City_Median"]).clip(-0.3, 0.3)
    rate = (
        0.04
        + 0.04 * df["Infrastructure_Score"] / 100
        + 0.015 * df["Under_Construction"]
        + 0.05 * undervaluation
        + 0.002 * df["Security_Flag"]
        - 0.0003 * df["Age_of_Property"].clip(upper=40)
    )
    df["Annual_Growth_Rate"] = rate.round(5)
    df["Future_Price_5Y_Lakhs"] = (df["Price_in_Lakhs"] * (1 + rate) ** HORIZON_YEARS).round(2)
    df["Good_Investment"] = (rate >= GOOD_INVESTMENT_RATE_THRESHOLD).astype(int)
    return df


# ---- Model feature lists (shared by training and app) ----------------------
CAT_FEATURES = [
    "State", "City", "Property_Type", "Furnished_Status", "Facing",
    "Owner_Type", "Availability_Status", "Public_Transport_Accessibility",
]
NUM_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property",
    "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
    "Parking_Space", "Amenities_Count", "Security_Flag", "Under_Construction",
    "School_Density_Score", "Hospital_Density_Score", "Infrastructure_Score",
    "Price_vs_City_Median", "Floor_Ratio",
]
MODEL_FEATURES = CAT_FEATURES + NUM_FEATURES
