"""
Step 3 - Train, evaluate and track models (classification + regression) with MLflow.
Usage:  python 03_train_models.py
View experiments:  mlflow ui --backend-store-uri sqlite:///mlflow.db

Outputs
  models/best_classifier.joblib, models/best_regressor.joblib   (full sklearn Pipelines)
  models/classification_metrics.csv, models/regression_metrics.csv
  models/feature_importance_clf.csv, models/feature_importance_reg.csv
  models/app_metadata.json  (dropdown options, residual quantiles, best-model info for the app)
  experiment_log.csv        (every run, whether or not MLflow is installed)
"""
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import common as C

warnings.filterwarnings("ignore")
MAX_TRAIN_ROWS = 80_000          # speeds up Random Forest on the full 250k-row file

try:
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri(f"sqlite:///{C.ROOT / 'mlflow.db'}")
    HAS_MLFLOW = True
except Exception:                                   # MLflow missing -> still log to CSV
    HAS_MLFLOW = False
    print("MLflow not installed - run `pip install mlflow` for experiment tracking UI. "
          "Runs are still saved to experiment_log.csv")
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("XGBoost not installed - using scikit-learn HistGradientBoosting instead "
          "(`pip install xgboost` to add XGBoost).")

df = pd.read_csv(C.CLEAN_CSV)
X = df[C.MODEL_FEATURES]
y_clf = df["Good_Investment"]
y_reg = df["Future_Price_5Y_Lakhs"]

X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
    X, y_clf, y_reg, test_size=0.2, random_state=C.RANDOM_STATE, stratify=y_clf)
if len(X_tr) > MAX_TRAIN_ROWS:
    idx = X_tr.sample(MAX_TRAIN_ROWS, random_state=C.RANDOM_STATE).index
    X_tr, yc_tr, yr_tr = X_tr.loc[idx], yc_tr.loc[idx], yr_tr.loc[idx]
print(f"Train {len(X_tr):,} rows | Test {len(X_te):,} rows | positive rate {y_clf.mean():.1%}")


def make_pipe(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), C.NUM_FEATURES),                                   # feature scaling
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.CAT_FEATURES),
    ])
    return Pipeline([("prep", pre), ("model", model)])


clf_models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=3,
                                           n_jobs=-1, random_state=C.RANDOM_STATE, class_weight="balanced"),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                                           random_state=C.RANDOM_STATE),
}
reg_models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=3,
                                          n_jobs=-1, random_state=C.RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                                          random_state=C.RANDOM_STATE),
}
if HAS_XGB:
    clf_models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9,
                                          colsample_bytree=0.9, eval_metric="logloss",
                                          random_state=C.RANDOM_STATE, n_jobs=-1)
    reg_models["XGBoost"] = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9,
                                         colsample_bytree=0.9, random_state=C.RANDOM_STATE, n_jobs=-1)

run_log = []


def log_run(experiment, name, params, metrics, pipe):
    run_log.append({"experiment": experiment, "model": name, **metrics})
    if not HAS_MLFLOW:
        return
    try:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            try:
                mlflow.sklearn.log_model(pipe, "model")
            except Exception as e:
                print(f"   (model artifact not logged to MLflow: {e})")
    except Exception as e:
        print(f"   (MLflow logging skipped: {e})")


def short_params(model):
    return {k: v for k, v in model.get_params().items()
            if isinstance(v, (int, float, str, bool, type(None)))}


# ----------------------------------------------------------- classification
clf_rows, clf_pipes = [], {}
for name, model in clf_models.items():
    t0 = time.time()
    pipe = make_pipe(model).fit(X_tr, yc_tr)
    pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
    m = {"accuracy": accuracy_score(yc_te, pred), "precision": precision_score(yc_te, pred, zero_division=0),
         "recall": recall_score(yc_te, pred), "f1": f1_score(yc_te, pred), "roc_auc": roc_auc_score(yc_te, proba),
         "train_seconds": time.time() - t0}
    clf_rows.append({"model": name, **m})
    clf_pipes[name] = pipe
    log_run("Good_Investment_Classification", name, short_params(model), m, pipe)
    print(f"[CLF] {name:22s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")
clf_df = pd.DataFrame(clf_rows).round(4).sort_values("f1", ascending=False)
best_clf_name = clf_df.iloc[0]["model"]
clf_df.to_csv(C.MODELS_DIR / "classification_metrics.csv", index=False)

# --------------------------------------------------------------- regression
reg_rows, reg_pipes = [], {}
for name, model in reg_models.items():
    t0 = time.time()
    pipe = make_pipe(model).fit(X_tr, yr_tr)
    pred = pipe.predict(X_te)
    m = {"rmse": float(np.sqrt(mean_squared_error(yr_te, pred))), "mae": mean_absolute_error(yr_te, pred),
         "r2": r2_score(yr_te, pred), "mape_pct": float(np.mean(np.abs((yr_te - pred) / yr_te)) * 100),
         "train_seconds": time.time() - t0}
    reg_rows.append({"model": name, **m})
    reg_pipes[name] = pipe
    log_run("Future_Price_5Y_Regression", name, short_params(model), m, pipe)
    print(f"[REG] {name:22s} rmse={m['rmse']:.3f} mae={m['mae']:.3f} r2={m['r2']:.4f}")
reg_df = pd.DataFrame(reg_rows).round(4).sort_values("rmse")
best_reg_name = reg_df.iloc[0]["model"]
reg_df.to_csv(C.MODELS_DIR / "regression_metrics.csv", index=False)

best_clf, best_reg = clf_pipes[best_clf_name], reg_pipes[best_reg_name]
joblib.dump(best_clf, C.MODELS_DIR / "best_classifier.joblib")
joblib.dump(best_reg, C.MODELS_DIR / "best_regressor.joblib")
pd.DataFrame(run_log).round(4).to_csv(C.ROOT / "experiment_log.csv", index=False)
print(f"Best classifier: {best_clf_name} | Best regressor: {best_reg_name}")

# ------------------------------------------------------ feature importance
# Permutation importance on the whole pipeline -> importance per ORIGINAL column
sample = X_te.sample(min(4000, len(X_te)), random_state=1)
for tag, pipe, y_all, scoring in [("clf", best_clf, yc_te, "f1"), ("reg", best_reg, yr_te, "neg_root_mean_squared_error")]:
    r = permutation_importance(pipe, sample, y_all.loc[sample.index], scoring=scoring,
                               n_repeats=3, random_state=C.RANDOM_STATE, n_jobs=1)
    imp = pd.DataFrame({"feature": sample.columns, "importance": r.importances_mean}) \
        .sort_values("importance", ascending=False)
    imp["importance_pct"] = (100 * imp["importance"].clip(lower=0) / max(imp["importance"].clip(lower=0).sum(), 1e-12)).round(2)
    imp.to_csv(C.MODELS_DIR / f"feature_importance_{tag}.csv", index=False)

# --------------------------------------------------- metadata for the app
resid = yr_te - best_reg.predict(X_te)
loc_map = (df.groupby("City")["Locality"].agg(lambda s: sorted(s.unique().tolist())).to_dict())
amen = sorted({a.strip() for v in df["Amenities"].dropna() for a in str(v).split(",")
               if a.strip() and a.strip().lower() not in ("none", "nan")})
meta = {
    "best_classifier": best_clf_name, "best_regressor": best_reg_name,
    "residual_q10": float(resid.quantile(0.10)), "residual_q90": float(resid.quantile(0.90)),
    "residual_rel_q10": float((resid / yr_te).quantile(0.10)), "residual_rel_q90": float((resid / yr_te).quantile(0.90)),
    "states": sorted(df["State"].unique().tolist()),
    "state_to_cities": df.groupby("State")["City"].agg(lambda s: sorted(s.unique().tolist())).to_dict(),
    "city_to_localities": loc_map,
    "amenities": amen,
    "options": {c: sorted(df[c].dropna().unique().tolist()) for c in
                ["Property_Type", "Furnished_Status", "Facing", "Owner_Type", "Availability_Status",
                 "Public_Transport_Accessibility", "Security"]},
    "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in
               ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No", "Total_Floors",
                "Nearby_Schools", "Nearby_Hospitals", "Parking_Space"]},
    "good_investment_threshold": C.GOOD_INVESTMENT_RATE_THRESHOLD,
    "horizon_years": C.HORIZON_YEARS,
}
with open(C.APP_META_JSON, "w") as f:
    json.dump(meta, f, indent=2)

def md_table(t):
    head = "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in t.values)


with open(C.MODELS_DIR / "model_report.md", "w") as f:
    f.write("## Classification (Good Investment) - test set\n\n" + md_table(clf_df))
    f.write("\n\n## Regression (price after 5 years, Rs lakhs) - test set\n\n" + md_table(reg_df))
    f.write(f"\n\nBest classifier: **{best_clf_name}** | Best regressor: **{best_reg_name}**\n")
print("All artefacts saved in /models")
