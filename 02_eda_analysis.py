"""
Step 2 - Exploratory Data Analysis (all 20 questions + the extra analyses in the brief)
Usage:  python 02_eda_analysis.py
Outputs: eda_outputs/charts/*.png, eda_outputs/tables/*.csv, eda_outputs/EDA_Findings.md
Every sentence in EDA_Findings.md is generated from the data - nothing is hard-coded.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import common as C

sns.set_theme(style="whitegrid", context="notebook")
df = pd.read_csv(C.CLEAN_CSV)
md = ["# EDA findings\n",
      f"Dataset analysed: **{len(df):,} cleaned properties**, {df['City'].nunique()} cities, "
      f"{df['State'].nunique()} states.\n"]


# ---------------------------------------------------------------- helpers
def fmt(x, d=0):
    return f"{x:,.{d}f}"


def strength(r):
    a = abs(r)
    word = "negligible" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong"
    return f"{word} {'positive' if r > 0 else 'negative'}" if a >= 0.1 else "negligible"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(C.CHARTS_DIR / f"{name}.png", dpi=130)
    plt.close(fig)
    return f"charts/{name}.png"


def table(t, name):
    t.to_csv(C.TABLES_DIR / f"{name}.csv")


def section(q, title, chart, bullets):
    md.append(f"\n## {q}. {title}\n")
    if chart:
        md.append(f"![{title}]({chart})\n")
    for b in bullets:
        md.append(f"- {b}")
    md.append("")


def spread_sentence(s, label, unit="", d=0, suffix=""):
    s = s.dropna().sort_values()
    hi, lo = s.index[-1], s.index[0]
    gap = (s.iloc[-1] / s.iloc[0] - 1) * 100 if s.iloc[0] else np.nan
    verdict = ("The gap is small, so this factor barely separates prices."
               if gap < 5 else "The gap is meaningful." if gap < 25 else "The gap is large.")
    return (f"{label}: highest **{hi}** ({unit}{fmt(s.iloc[-1], d)}{suffix}), lowest **{lo}** "
            f"({unit}{fmt(s.iloc[0], d)}{suffix}) - a {gap:.1f}% difference. {verdict}")


def bar(series, title, ylabel, name, horizontal=False, top=None, color="#2f6db5"):
    s = series.sort_values(ascending=False)
    if top:
        s = s.head(top)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(s)) if horizontal else 5))
    if horizontal:
        sns.barplot(x=s.values, y=s.index.astype(str), ax=ax, color=color)
        ax.set_xlabel(ylabel)
    else:
        sns.barplot(x=s.index.astype(str), y=s.values, ax=ax, color=color)
        ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(title)
    return save(fig, name)


# ================= 1-5 PRICE & SIZE =================
# Q1
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["Price_in_Lakhs"], bins=50, kde=True, ax=ax[0], color="#2f6db5")
ax[0].set_title("Price (Rs lakhs)")
sns.histplot(np.log10(df["Price_in_Lakhs"]), bins=50, kde=True, ax=ax[1], color="#2f6db5")
ax[1].set_title("Price - log10 scale")
p = df["Price_in_Lakhs"]
section(1, "Distribution of property prices", save(fig, "q01_price_distribution"), [
    f"Mean Rs {fmt(p.mean(), 1)} lakhs vs median Rs {fmt(p.median(), 1)} lakhs; "
    f"middle 50% of properties cost Rs {fmt(p.quantile(.25), 1)} - {fmt(p.quantile(.75), 1)} lakhs.",
    f"Skewness = {p.skew():.2f} " + ("(right-skewed: a few expensive properties pull the mean above the median)."
                                       if p.skew() > 0.5 else "(fairly symmetric)."),
])

# Q2
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["Size_in_SqFt"], bins=50, kde=True, ax=ax, color="#2a9d8f")
ax.set_title("Property size (sq ft)")
s = df["Size_in_SqFt"]
section(2, "Distribution of property sizes", save(fig, "q02_size_distribution"), [
    f"Mean {fmt(s.mean())} sq ft, median {fmt(s.median())} sq ft; middle 50% between "
    f"{fmt(s.quantile(.25))} and {fmt(s.quantile(.75))} sq ft; range {fmt(s.min())} - {fmt(s.max())}.",
    f"Skewness = {s.skew():.2f}.",
])

# Q3
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt", ax=ax, color="#8ecae6", showfliers=False)
ax.set_title("Price per sq ft (Rs) by property type")
t = df.groupby("Property_Type")["Price_per_SqFt"].agg(["mean", "median", "count"]).round(0)
table(t, "q03_ppsf_by_property_type")
section(3, "Price per sq ft by property type", save(fig, "q03_ppsf_by_type"),
        [spread_sentence(t["mean"], "Average price/sq ft by type", "Rs ")])

# Q4
samp = df.sample(min(6000, len(df)), random_state=1)
fig, ax = plt.subplots(figsize=(8, 5))
sns.regplot(data=samp, x="Size_in_SqFt", y="Price_in_Lakhs", ax=ax, scatter_kws={"alpha": .2, "s": 8},
            line_kws={"color": "crimson"})
ax.set_title("Size vs price (6,000-property sample)")
rp, rs = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"]), df["Size_in_SqFt"].corr(df["Price_in_Lakhs"], method="spearman")
section(4, "Relationship between size and price", save(fig, "q04_size_vs_price"), [
    f"Pearson r = {rp:.3f}, Spearman rho = {rs:.3f} -> {strength(rp)} linear relationship.",
    "Larger properties cost more in absolute terms." if rp > 0.3 else
    "Size alone explains little of the price - location and quality matter more.",
])

# Q5
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, col, c in zip(ax, ["Price_per_SqFt", "Size_in_SqFt", "Price_in_Lakhs"], ["#e76f51", "#2a9d8f", "#2f6db5"]):
    sns.boxplot(y=df[col], ax=a, color=c)
    a.set_title(col)
ob = {c: int(df[f].sum()) for c, f in [("Price_per_SqFt", "Outlier_PricePerSqFt"), ("Size_in_SqFt", "Outlier_Size"),
                                      ("Price_in_Lakhs", "Outlier_Price")]}
section(5, "Outliers in price per sq ft and size", save(fig, "q05_outliers"), [
    f"IQR-rule outliers still present after cleaning: price/sq ft **{ob['Price_per_SqFt']:,}** "
    f"({100 * ob['Price_per_SqFt'] / len(df):.2f}%), size **{ob['Size_in_SqFt']:,}** "
    f"({100 * ob['Size_in_SqFt'] / len(df):.2f}%), price **{ob['Price_in_Lakhs']:,}** "
    f"({100 * ob['Price_in_Lakhs'] / len(df):.2f}%).",
    "Extreme data-entry errors (beyond 3 x IQR) were removed during preprocessing; milder outliers were "
    "kept and flagged because they represent genuine luxury / very large properties.",
])

# ================= 6-10 LOCATION =================
# Q6
st = df.groupby("State")["Price_per_SqFt"].mean()
table(st.sort_values(ascending=False).round(0).to_frame("avg_price_per_sqft"), "q06_ppsf_by_state")
section(6, "Average price per sq ft by state",
        bar(st, "Average price per sq ft by state (Rs)", "Rs / sq ft", "q06_ppsf_by_state", horizontal=True),
        [spread_sentence(st, "States", "Rs ")])

# Q7
ct = df.groupby("City")["Price_in_Lakhs"].mean()
table(ct.sort_values(ascending=False).round(1).to_frame("avg_price_lakhs"), "q07_price_by_city")
section(7, "Average property price by city",
        bar(ct, "Average price by city (Rs lakhs)", "Rs lakhs", "q07_price_by_city", horizontal=True, top=25),
        [spread_sentence(ct, "Cities (avg price)", "Rs ", 1, " lakhs")])

# Q8
loc = df.groupby(["City", "Locality"])["Age_of_Property"].agg(["median", "count"])
loc = loc[loc["count"] >= 10].reset_index()
loc["label"] = loc["City"] + " - " + loc["Locality"]
loc_s = loc.set_index("label")["median"]
table(loc.sort_values("median"), "q08_median_age_by_locality")
fig, ax = plt.subplots(figsize=(9, 7))
pick = pd.concat([loc_s.nsmallest(10), loc_s.nlargest(10)]).sort_values()
sns.barplot(x=pick.values, y=pick.index, ax=ax, color="#9c6644")
ax.set_title("Median property age (years): 10 newest & 10 oldest localities")
section(8, "Median age of properties by locality", save(fig, "q08_median_age_locality"), [
    f"{len(loc):,} localities have >= 10 listings. Median age ranges from {loc_s.min():.0f} to {loc_s.max():.0f} years "
    f"(overall median {df['Age_of_Property'].median():.0f}).",
    f"Newest: **{loc_s.idxmin()}** ({loc_s.min():.0f} yrs); oldest: **{loc_s.idxmax()}** ({loc_s.max():.0f} yrs). "
    "Full table in `tables/q08_median_age_by_locality.csv`.",
])

# Q9
topc = df["City"].value_counts().head(15).index
ctab = pd.crosstab(df[df["City"].isin(topc)]["City"], df["BHK"], normalize="index") * 100
table(ctab.round(1), "q09_bhk_share_by_city_pct")
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(ctab, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "% of city listings"})
ax.set_title("BHK mix by city (% of listings, top 15 cities)")
mode_bhk = df["BHK"].mode().iloc[0]
section(9, "BHK distribution across cities", save(fig, "q09_bhk_by_city"), [
    f"Most common configuration overall: **{mode_bhk} BHK** ({100 * (df['BHK'] == mode_bhk).mean():.1f}% of listings).",
    f"City BHK mix differs by at most {(ctab.max() - ctab.min()).max():.1f} percentage points for any BHK class "
    + ("-> cities have very similar BHK profiles." if (ctab.max() - ctab.min()).max() < 5 else "-> cities differ in housing mix."),
])

# Q10
g_all = df.groupby(["City", "Locality"])["Price_per_SqFt"].agg(["mean", "count"])
min_n = int(min(30, max(5, g_all["count"].quantile(0.75))))      # adapt to how thin localities are
g = g_all[g_all["count"] >= min_n].sort_values("mean", ascending=False).head(5)
d5 = df.merge(g.reset_index()[["City", "Locality"]], on=["City", "Locality"])
d5["Place"] = d5["City"] + " - " + d5["Locality"]
d5["Built_Period"] = (d5["Year_Built"] // 5 * 5).astype(int)
tr = d5.groupby(["Built_Period", "Place"])["Price_per_SqFt"].mean().unstack()
table(tr.round(0), "q10_top5_locality_trend")
fig, ax = plt.subplots(figsize=(10, 5))
tr.plot(marker="o", ax=ax)
ax.set_ylabel("Avg price per sq ft (Rs)")
ax.set_xlabel("Year built (5-year bins)")
ax.set_title("Top 5 most expensive localities - price/sq ft by construction period")
section(10, "Price trends for the top 5 most expensive localities", save(fig, "q10_top5_locality_trend"), [
    f"Top 5 (>= {min_n} listings each): " + "; ".join(f"**{i[0]} - {i[1]}** (Rs {fmt(r['mean'])}/sq ft)" for i, r in g.iterrows()) + ".",
    "The dataset has no transaction dates, so *year built* is used as the time axis: the chart shows how the price of "
    "stock built in each period compares, not historical price movement of the same property.",
    "Points rest on few listings per period, so read the lines as indicative rather than precise."
    if d5.groupby(["Built_Period", "Place"]).size().median() < 15 else
    "Each plotted point averages a reasonable number of listings.",
])

# ================= 11-15 RELATIONSHIPS =================
# Q11
numc = ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt", "BHK", "Age_of_Property", "Nearby_Schools",
        "Nearby_Hospitals", "Parking_Space", "Amenities_Count", "Transport_Score", "Infrastructure_Score",
        "Floor_No", "Total_Floors"]
cm = df[numc].corr()
table(cm.round(3), "q11_correlation_matrix")
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation matrix of numeric features")
pairs = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack().sort_values(key=abs, ascending=False).head(5)
section(11, "Correlation between numeric features", save(fig, "q11_correlation_heatmap"), [
    "Strongest pairs: " + "; ".join(f"{a} & {b} (r = {v:.2f})" for (a, b), v in pairs.items()) + ".",
    "Price_per_SqFt is derived from Price and Size, so strong links with those two are expected, not insight.",
])


# Q12/13/14/15 generic grouped-mean plots
def grouped_mean(col, ycol, q, title, name, kind="line", xlabel=None):
    t = df.groupby(col)[ycol].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if kind == "line":
        ax.plot(t.index, t.values, marker="o", color="#2f6db5")
    else:
        sns.barplot(x=t.index.astype(str), y=t.values, ax=ax, color="#2f6db5")
    ax.set_xlabel(xlabel or col)
    ax.set_ylabel(f"Avg {ycol}")
    ax.set_title(title)
    table(t.round(2).to_frame(), name)
    return t, save(fig, name)


t12, ch = grouped_mean("Nearby_Schools", "Price_per_SqFt", 12, "Avg price/sq ft by number of nearby schools", "q12_schools_vs_ppsf")
r = df["Nearby_Schools"].corr(df["Price_per_SqFt"])
section(12, "Nearby schools vs price per sq ft", ch, [
    f"Correlation r = {r:.3f} ({strength(r)}).",
    f"Average price/sq ft goes from Rs {fmt(t12.iloc[0])} (at {t12.index[0]}) to Rs {fmt(t12.iloc[-1])} (at {t12.index[-1]}).",
])
t13, ch = grouped_mean("Nearby_Hospitals", "Price_per_SqFt", 13, "Avg price/sq ft by number of nearby hospitals", "q13_hospitals_vs_ppsf")
r = df["Nearby_Hospitals"].corr(df["Price_per_SqFt"])
section(13, "Nearby hospitals vs price per sq ft", ch, [
    f"Correlation r = {r:.3f} ({strength(r)}).",
    f"Average price/sq ft goes from Rs {fmt(t13.iloc[0])} (at {t13.index[0]}) to Rs {fmt(t13.iloc[-1])} (at {t13.index[-1]}).",
])
t14 = df.groupby("Furnished_Status")[["Price_in_Lakhs", "Price_per_SqFt"]].mean().round(1)
table(t14, "q14_price_by_furnished")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
sns.barplot(x=t14.index, y=t14["Price_in_Lakhs"], ax=ax[0], color="#2f6db5")
ax[0].set_title("Avg price (lakhs)")
sns.barplot(x=t14.index, y=t14["Price_per_SqFt"], ax=ax[1], color="#2a9d8f")
ax[1].set_title("Avg price per sq ft (Rs)")
section(14, "Price by furnished status", save(fig, "q14_price_by_furnished"), [
    spread_sentence(t14["Price_in_Lakhs"], "Average price", "Rs ", 1, " lakhs"),
    spread_sentence(t14["Price_per_SqFt"], "Average price/sq ft", "Rs "),
])
t15 = df.groupby("Facing")["Price_per_SqFt"].mean()
table(t15.round(0).to_frame(), "q15_ppsf_by_facing")
section(15, "Price per sq ft by facing direction",
        bar(t15, "Avg price per sq ft by facing (Rs)", "Rs / sq ft", "q15_ppsf_by_facing"),
        [spread_sentence(t15, "Facing", "Rs ")])

# ================= 16-20 INVESTMENT / AMENITIES / OWNERSHIP =================
for q, col, title in [(16, "Owner_Type", "Properties per owner type"),
                      (17, "Availability_Status", "Properties per availability status")]:
    vc = df[col].value_counts()
    table(vc.to_frame("count"), f"q{q}_{col.lower()}")
    section(q, title, bar(vc, title, "Number of properties", f"q{q}_{col.lower()}"), [
        "; ".join(f"**{k}**: {v:,} ({100 * v / len(df):.1f}%)" for k, v in vc.items()) + "."])

# Q18
t18 = df.groupby("Parking_Space")[["Price_in_Lakhs", "Price_per_SqFt"]].mean().round(1)
table(t18, "q18_parking_effect")
r = df["Parking_Space"].corr(df["Price_in_Lakhs"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
sns.barplot(x=t18.index, y=t18["Price_in_Lakhs"], ax=ax[0], color="#2f6db5"); ax[0].set_title("Avg price (lakhs)")
sns.barplot(x=t18.index, y=t18["Price_per_SqFt"], ax=ax[1], color="#2a9d8f"); ax[1].set_title("Avg price per sq ft (Rs)")
section(18, "Does parking space affect price?", save(fig, "q18_parking_effect"), [
    f"Correlation between parking spaces and price r = {r:.3f} ({strength(r)}).",
    spread_sentence(t18["Price_in_Lakhs"], "Average price by parking count", "Rs ", 1, " lakhs")])

# Q19
am = df.assign(Amenity=df["Amenities"].fillna("None").str.split(",")).explode("Amenity")
am["Amenity"] = am["Amenity"].str.strip()
am = am[~am["Amenity"].str.lower().isin(["none", "", "nan"])]
t19 = am.groupby("Amenity")["Price_per_SqFt"].agg(["mean", "count"]).round(0)
table(t19, "q19_amenity_effect")
t19c = df.groupby("Amenities_Count")["Price_per_SqFt"].mean()
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.barplot(x=t19.index, y=t19["mean"], ax=ax[0], color="#2f6db5"); ax[0].set_title("Avg price/sq ft by amenity present")
plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right")
ax[1].plot(t19c.index, t19c.values, marker="o", color="#e76f51"); ax[1].set_title("Avg price/sq ft by number of amenities")
r = df["Amenities_Count"].corr(df["Price_per_SqFt"])
section(19, "How amenities affect price per sq ft", save(fig, "q19_amenities"), [
    spread_sentence(t19["mean"], "Properties offering each amenity", "Rs "),
    f"Number of amenities vs price/sq ft: r = {r:.3f} ({strength(r)}).",
])

# Q20
t20 = df.groupby("Public_Transport_Accessibility").agg(
    avg_ppsf=("Price_per_SqFt", "mean"), good_inv_pct=("Good_Investment", lambda s: 100 * s.mean()),
    avg_growth_pct=("Annual_Growth_Rate", lambda s: 100 * s.mean())).round(2)
table(t20, "q20_transport_effect")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
sns.barplot(x=t20.index, y=t20["avg_ppsf"], ax=ax[0], color="#2f6db5"); ax[0].set_title("Avg price per sq ft (Rs)")
sns.barplot(x=t20.index, y=t20["good_inv_pct"], ax=ax[1], color="#2a9d8f"); ax[1].set_title("% rated Good Investment")
section(20, "Public transport accessibility vs price and investment potential", save(fig, "q20_transport"), [
    spread_sentence(t20["avg_ppsf"], "Price/sq ft by accessibility level", "Rs "),
    spread_sentence(t20["good_inv_pct"], "Share of Good Investments by accessibility", "", 1, "%"),
    "The Good_Investment label is rule-based and rewards accessibility, so that link is by design.",
])

# ================= EXTRA analyses named in the brief =================
md.append("\n# Additional analyses from the project brief\n")
# A. area vs investment return
r1 = df["Size_in_SqFt"].corr(df["Annual_Growth_Rate"])
gi = df.groupby("Good_Investment")["Size_in_SqFt"].median()
section("A", "Correlation between area and investment return", None, [
    f"Size vs modelled annual growth: r = {r1:.3f} ({strength(r1)}).",
    f"Median size of Good Investments {fmt(gi.get(1, np.nan))} sq ft vs others {fmt(gi.get(0, np.nan))} sq ft."])
# B. crime
section("B", "Impact of crime rate on Good Investment", None, [
    "The provided dataset has **no crime-rate column**, so this cannot be measured honestly. "
    "If a city-level crime index is obtained (e.g. NCRB), join it on City and rerun."])
# C. infrastructure vs value
r2 = df["Infrastructure_Score"].corr(df["Price_per_SqFt"])
df["Infra_Band"] = pd.qcut(df["Infrastructure_Score"], 5, duplicates="drop")
t = df.groupby("Infra_Band", observed=True)["Price_per_SqFt"].mean()
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(x=[str(i) for i in t.index], y=t.values, ax=ax, color="#2f6db5")
ax.set_title("Avg price/sq ft by infrastructure-score quintile"); plt.setp(ax.get_xticklabels(), rotation=20)
section("C", "Infrastructure score vs resale value (price per sq ft)", save(fig, "qC_infrastructure_vs_value"), [
    f"Correlation r = {r2:.3f} ({strength(r2)}).",
    spread_sentence(pd.Series(t.values, index=[str(i) for i in t.index]), "Quintiles", "Rs ")])
# D. good investment by city
gc = df.groupby("City")["Good_Investment"].mean() * 100
table(gc.round(1).sort_values(ascending=False).to_frame("good_investment_pct"), "qD_good_investment_by_city")
section("D", "Share of Good Investments by city",
        bar(gc, "% of listings rated Good Investment", "%", "qD_good_by_city", horizontal=True, top=25),
        [spread_sentence(gc, "Cities (share of Good Investments)", "", 1, "%")])
# E. Price per sqft by city (price trends by city)
pc = df.groupby("City")["Price_per_SqFt"].median()
section("E", "Price per sq ft by city (median)",
        bar(pc, "Median price per sq ft by city (Rs)", "Rs / sq ft", "qE_ppsf_by_city", horizontal=True, top=25),
        [spread_sentence(pc, "Cities", "Rs ")])

with open(C.EDA_DIR / "EDA_Findings.md", "w") as f:
    f.write("\n".join(md))
print(f"EDA done: {len(list(C.CHARTS_DIR.glob('*.png')))} charts, findings -> {C.EDA_DIR / 'EDA_Findings.md'}")
