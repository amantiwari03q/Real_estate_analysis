"""
Step 5 - Streamlit app: Real Estate Investment Advisor
Run:  streamlit run app.py
Needs the outputs of scripts 01 and 03 (data/cleaned_housing.csv and models/*).
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import common as C

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")
sns.set_theme(style="whitegrid")


# ------------------------------------------------------------------ loaders
@st.cache_resource
def load_models():
    return (joblib.load(C.MODELS_DIR / "best_classifier.joblib"),
            joblib.load(C.MODELS_DIR / "best_regressor.joblib"))


@st.cache_data
def load_meta():
    with open(C.APP_META_JSON) as f:
        return json.load(f), C.load_reference()


@st.cache_data
def load_data():
    return pd.read_csv(C.CLEAN_CSV)


@st.cache_data
def load_csv(name):
    p = C.MODELS_DIR / name
    return pd.read_csv(p) if p.exists() else None


try:
    clf, reg = load_models()
    meta, ref = load_meta()
    data = load_data()
except FileNotFoundError:
    st.error("Model files not found. Run `01_data_preprocessing.py` then `03_train_models.py` first.")
    st.stop()


def predict(rows: pd.DataFrame):
    """rows: raw property columns -> engineered features -> predictions."""
    feat = C.add_features(rows, ref)
    X = feat[C.MODEL_FEATURES]
    proba = clf.predict_proba(X)[:, 1]
    price5 = reg.predict(X)
    return feat, proba, price5


# --------------------------------------------------------------------- header
st.title("🏠 Real Estate Investment Advisor")
st.caption("Is this property a good investment, and what could it be worth in "
           f"{meta['horizon_years']} years? Models: {meta['best_classifier']} (classifier) / "
           f"{meta['best_regressor']} (regressor).")

# ------------------------------------------------------------ sidebar filters
st.sidebar.header("🔎 Filter properties")
f_states = st.sidebar.multiselect("State", meta["states"])
city_pool = sorted({c for s in (f_states or meta["states"]) for c in meta["state_to_cities"][s]})
f_cities = st.sidebar.multiselect("City", city_pool)
pmin, pmax = float(data["Price_in_Lakhs"].min()), float(data["Price_in_Lakhs"].max())
f_price = st.sidebar.slider("Price (Rs lakhs)", pmin, pmax, (pmin, float(data["Price_in_Lakhs"].quantile(0.95))))
smin, smax = int(data["Size_in_SqFt"].min()), int(data["Size_in_SqFt"].max())
f_size = st.sidebar.slider("Area (sq ft)", smin, smax, (smin, smax))
f_bhk = st.sidebar.multiselect("BHK", sorted(data["BHK"].unique().tolist()))
f_type = st.sidebar.multiselect("Property type", meta["options"]["Property_Type"])

fdf = data[data["Price_in_Lakhs"].between(*f_price) & data["Size_in_SqFt"].between(*f_size)]
if f_states:
    fdf = fdf[fdf["State"].isin(f_states)]
if f_cities:
    fdf = fdf[fdf["City"].isin(f_cities)]
if f_bhk:
    fdf = fdf[fdf["BHK"].isin(f_bhk)]
if f_type:
    fdf = fdf[fdf["Property_Type"].isin(f_type)]
st.sidebar.metric("Matching properties", f"{len(fdf):,}")

tab_pred, tab_insight, tab_list, tab_model = st.tabs(
    ["🎯 Predict a property", "📊 Market insights", "📋 Matching listings", "🧠 Model performance"])

# ------------------------------------------------------------- tab 1: predict
with tab_pred:
    st.subheader("Enter property details")
    with st.form("property_form"):
        c1, c2, c3 = st.columns(3)
        state = c1.selectbox("State", meta["states"])
        city = c2.selectbox("City", meta["state_to_cities"][state])
        locality = c3.selectbox("Locality", meta["city_to_localities"].get(city, ["Unknown"]))

        c1, c2, c3, c4 = st.columns(4)
        ptype = c1.selectbox("Property type", meta["options"]["Property_Type"])
        bhk = c2.number_input("BHK", 1, 10, 3)
        size = c3.number_input("Size (sq ft)", 100, 20000, 1500, step=50)
        price = c4.number_input("Asking price (Rs lakhs)", 1.0, 100000.0,
                                float(round(data["Price_in_Lakhs"].median(), 1)), step=5.0)

        c1, c2, c3, c4 = st.columns(4)
        year = c1.number_input("Year built", 1950, C.CURRENT_YEAR, 2015)
        furn = c2.selectbox("Furnished status", meta["options"]["Furnished_Status"])
        floor_no = c3.number_input("Floor no.", 0, 100, 2)
        total_floors = c4.number_input("Total floors", 1, 100, 10)

        c1, c2, c3, c4 = st.columns(4)
        schools = c1.number_input("Nearby schools", 0, 50, 5)
        hospitals = c2.number_input("Nearby hospitals", 0, 50, 5)
        transport = c3.selectbox("Public transport", meta["options"]["Public_Transport_Accessibility"])
        parking = c4.number_input("Parking spaces", 0, 10, 1)

        c1, c2, c3, c4 = st.columns(4)
        security = c1.selectbox("Security", meta["options"]["Security"])
        facing = c2.selectbox("Facing", meta["options"]["Facing"])
        owner = c3.selectbox("Owner type", meta["options"]["Owner_Type"])
        avail = c4.selectbox("Availability", meta["options"]["Availability_Status"])
        amenities = st.multiselect("Amenities", meta["amenities"])
        submitted = st.form_submit_button("Analyse investment", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "State": state, "City": city, "Locality": locality, "Property_Type": ptype, "BHK": bhk,
            "Size_in_SqFt": size, "Price_in_Lakhs": price, "Year_Built": year, "Furnished_Status": furn,
            "Floor_No": min(floor_no, total_floors), "Total_Floors": total_floors, "Nearby_Schools": schools,
            "Nearby_Hospitals": hospitals, "Public_Transport_Accessibility": transport, "Parking_Space": parking,
            "Security": security, "Amenities": ", ".join(amenities) if amenities else "None", "Facing": facing,
            "Owner_Type": owner, "Availability_Status": avail}])
        feat, proba, price5 = predict(row)
        p, fut = float(proba[0]), float(price5[0])
        lo, hi = fut * (1 + meta["residual_rel_q10"]), fut * (1 + meta["residual_rel_q90"])
        cagr = (fut / price) ** (1 / meta["horizon_years"]) - 1
        confidence = max(p, 1 - p)

        st.divider()
        a, b, c = st.columns(3)
        a.metric("Is this a Good Investment?", "✅ YES" if p >= 0.5 else "❌ NO")
        b.metric(f"Estimated price after {meta['horizon_years']} years", f"Rs {fut:,.1f} lakhs",
                 f"{fut - price:+,.1f} lakhs ({(fut / price - 1) * 100:+.1f}%)")
        c.metric("Implied annual growth", f"{cagr * 100:.1f}% p.a.",
                 f"threshold {meta['good_investment_threshold'] * 100:.0f}%")

        st.write(f"**Model confidence in the verdict:** {confidence * 100:.1f}% "
                 f"(P[Good Investment] = {p * 100:.1f}%)")
        st.progress(min(max(p, 0.0), 1.0))
        st.write(f"**Likely range for the {meta['horizon_years']}-year price** (80% of test errors fall inside): "
                 f"Rs {lo:,.1f} - {hi:,.1f} lakhs")

        f = feat.iloc[0]
        d1, d2, d3 = st.columns(3)
        d1.metric("Price per sq ft", f"Rs {f['Price_per_SqFt']:,.0f}",
                  f"{(f['Price_vs_City_Median'] - 1) * 100:+.1f}% vs {city} median")
        d2.metric("Infrastructure score", f"{f['Infrastructure_Score']:.0f} / 100")
        d3.metric("Property age", f"{int(f['Age_of_Property'])} years")
        if f["Price_vs_City_Median"] > 1.2:
            st.warning("Priced more than 20% above the city median per sq ft - limits upside.")
        elif f["Price_vs_City_Median"] < 0.9:
            st.info("Priced more than 10% below the city median per sq ft - relative value.")
        st.caption("Estimates come from models trained on rule-derived targets, not observed resale prices. "
                   "Use as a screening aid, not financial advice.")

        imp = load_csv("feature_importance_clf.csv")
        if imp is not None:
            st.markdown("**What drives the verdict (global feature importance)**")
            top = imp.head(8).iloc[::-1]
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.barh(top["feature"], top["importance_pct"], color="#2f6db5")
            ax.set_xlabel("Importance (%)")
            st.pyplot(fig)

# ----------------------------------------------------------- tab 2: insights
with tab_insight:
    st.subheader(f"Market insights for the {len(fdf):,} filtered properties")
    if len(fdf) < 20:
        st.warning("Too few properties for meaningful charts - widen the filters.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Heatmap: avg price per sq ft (Rs), city x property type**")
            top_c = fdf["City"].value_counts().head(12).index
            hm = fdf[fdf["City"].isin(top_c)].pivot_table(index="City", columns="Property_Type",
                                                          values="Price_per_SqFt", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.heatmap(hm, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, annot_kws={"size": 7})
            st.pyplot(fig)
        with c2:
            st.markdown("**Trend: avg price per sq ft by construction period**")
            tmp = fdf.assign(Period=(fdf["Year_Built"] // 5 * 5))
            st.line_chart(tmp.groupby("Period")["Price_per_SqFt"].mean())
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Average price by city (Rs lakhs)**")
            st.bar_chart(fdf.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(15))
        with c4:
            st.markdown("**Share of Good Investments by city (%)**")
            st.bar_chart((fdf.groupby("City")["Good_Investment"].mean() * 100).sort_values(ascending=False).head(15))
        c5, c6 = st.columns(2)
        with c5:
            st.markdown("**Price per sq ft by transport accessibility (Rs)**")
            st.bar_chart(fdf.groupby("Public_Transport_Accessibility")["Price_per_SqFt"].mean())
        with c6:
            st.markdown("**Average 5-year price growth by BHK (%)**")
            g = (fdf["Future_Price_5Y_Lakhs"] / fdf["Price_in_Lakhs"] - 1) * 100
            st.bar_chart(g.groupby(fdf["BHK"]).mean())

# ------------------------------------------------------------ tab 3: listings
with tab_list:
    st.subheader("Matching listings ranked by investment probability")
    if len(fdf) == 0:
        st.warning("No properties match the filters.")
    else:
        view = fdf.sample(min(3000, len(fdf)), random_state=1)
        _, proba, price5 = predict(view[[c for c in view.columns if c in
                                         C.CATEGORICAL_COLS_RAW + C.NUMERIC_COLS_RAW]])
        out = view[["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs"]].copy() \
            if "ID" in view.columns else view[["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs"]].copy()
        out["P_Good_Investment_%"] = (proba * 100).round(1)
        out["Predicted_Price_5Y_Lakhs"] = price5.round(1)
        out = out.sort_values("P_Good_Investment_%", ascending=False).head(200)
        st.caption(f"Showing the top 200 of {len(view):,} sampled matches.")
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button("Download results (CSV)", out.to_csv(index=False).encode(), "matching_listings.csv", "text/csv")

# ----------------------------------------------------------- tab 4: model info
with tab_model:
    st.subheader("Model comparison (held-out test set)")
    cm, rm = load_csv("classification_metrics.csv"), load_csv("regression_metrics.csv")
    c1, c2 = st.columns(2)
    if cm is not None:
        c1.markdown("**Classification - Good Investment**")
        c1.dataframe(cm, hide_index=True)
    if rm is not None:
        c2.markdown("**Regression - price after 5 years (Rs lakhs)**")
        c2.dataframe(rm, hide_index=True)
    st.subheader("Feature importance")
    c1, c2 = st.columns(2)
    for col, tag, ttl in [(c1, "clf", "Classifier"), (c2, "reg", "Regressor")]:
        imp = load_csv(f"feature_importance_{tag}.csv")
        if imp is not None:
            col.markdown(f"**{ttl}: top 12 features**")
            col.bar_chart(imp.head(12).set_index("feature")["importance_pct"])
    st.info("MLflow: run `mlflow ui --backend-store-uri sqlite:///mlflow.db` to explore every experiment run.")
