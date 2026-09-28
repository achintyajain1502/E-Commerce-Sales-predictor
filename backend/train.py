"""Module 3: train Linear Regression / Random Forest / XGBoost, keep the best."""
import json, joblib, pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from common import FEATURES, NUMERIC, CATEGORICAL, TARGET, clean, add_history_features

try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9, random_state=42)
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    xgb = GradientBoostingRegressor(random_state=42)


def train(csv_path="data/sales.csv"):
    df = add_history_features(clean(pd.read_csv(csv_path)))
    cutoff = df["date"].max() - pd.Timedelta(days=90)      # time-based split, never random
    tr, te = df[df["date"] <= cutoff], df[df["date"] > cutoff]
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
                             remainder="passthrough")
    models = {"Linear Regression": LinearRegression(),
              "Random Forest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
              "XGBoost": xgb}
    results, best, best_rmse = [], None, float("inf")
    for name, m in models.items():
        pipe = Pipeline([("prep", prep), ("model", m)]).fit(tr[FEATURES], tr[TARGET])
        p = pipe.predict(te[FEATURES])
        r = dict(model=name, MAE=round(mean_absolute_error(te[TARGET], p), 2),
                 RMSE=round(float(np.sqrt(mean_squared_error(te[TARGET], p))), 2),
                 R2=round(r2_score(te[TARGET], p), 3))
        results.append(r); print(r)
        if r["RMSE"] < best_rmse:
            best, best_rmse = (name, pipe), r["RMSE"]
    # refit winner on all data before saving
    best[1].fit(df[FEATURES], df[TARGET])
    joblib.dump(best[1], "models/best_model.joblib")
    json.dump({"best_model": best[0], "results": results}, open("models/metrics.json", "w"), indent=2)
    print("Best:", best[0])


if __name__ == "__main__":
    train()
