"""FastAPI backend: analytics, prediction, forecasting, insights.
Run locally: uvicorn main:app --reload
Run on Render: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import json
import io
import joblib
import pandas as pd

from typing import Optional
from datetime import timedelta

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from common import FEATURES, PRODUCTS, is_holiday, clean, add_calendar


app = FastAPI(title="E-Commerce Sales Predictor")


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
# Allows the deployed React/Vite frontend on Vercel
# to communicate with this FastAPI backend on Render.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# DATA & MODEL
# ---------------------------------------------------------

DATA = "data/sales.csv"

df = clean(pd.read_csv(DATA))

model = joblib.load("models/best_model.joblib")


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def _row(pid, date, price, discount, marketing, hol, prev_week, a7, a30):
    d = add_calendar(
        pd.DataFrame({"date": [date]})
    ).iloc[0]

    return pd.DataFrame([
        dict(
            price=price,
            discount=discount,
            marketing_spend=marketing,
            is_holiday=hol,
            is_weekend=int(d.is_weekend),
            day_of_week=int(d.day_of_week),
            month=int(d.month),
            week_of_year=int(d.week_of_year),
            prev_week_sales=prev_week,
            avg_7d=a7,
            avg_30d=a30,
            category=PRODUCTS[pid][1],
        )
    ])[FEATURES]


def _hist(pid):
    if pid not in PRODUCTS:
        raise HTTPException(404, "Unknown product")

    return df[df.product_id == pid].sort_values("date")


# ---------------------------------------------------------
# PRODUCTS
# ---------------------------------------------------------

@app.get("/products")
def products():
    return [
        {
            "product_id": k,
            "name": v[0],
            "category": v[1],
            "base_price": v[2],
        }
        for k, v in PRODUCTS.items()
    ]


# ---------------------------------------------------------
# MODEL METRICS
# ---------------------------------------------------------

@app.get("/models/metrics")
def metrics():
    return json.load(open("models/metrics.json"))


# ---------------------------------------------------------
# ANALYTICS SUMMARY
# ---------------------------------------------------------

@app.get("/analytics/summary")
def summary():
    total_units = int(df.units_sold.sum())
    revenue = float(df.revenue.sum())

    monthly = (
        df.assign(
            m=df.date.dt.to_period("M").astype(str)
        )
        .groupby("m")[["units_sold", "revenue"]]
        .sum()
    )

    by_prod = (
        df.groupby("product_name")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    by_cat = (
        df.groupby("category")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    disc = df.assign(
        bucket=pd.cut(
            df.discount,
            [-1, 0, 10, 20, 100],
            labels=["0%", "1-10%", "11-20%", "20%+"],
        )
    )

    return {
        "total_units": total_units,
        "total_revenue": revenue,
        "avg_revenue_per_unit": round(
            revenue / total_units,
            2
        ),
        "monthly_trend": monthly.reset_index().to_dict("records"),
        "top_products": by_prod.head(5).round(0).to_dict(),
        "top_categories": by_cat.round(0).to_dict(),
        "discount_vs_sales": (
            disc.groupby(
                "bucket",
                observed=True
            )["units_sold"]
            .mean()
            .round(1)
            .to_dict()
        ),
    }


# ---------------------------------------------------------
# PREDICTION INPUT
# ---------------------------------------------------------

class PredictIn(BaseModel):
    product_id: str
    date: Optional[str] = None
    price: Optional[float] = None
    discount: float = 0
    marketing_spend: float = 10000
    is_holiday: Optional[int] = None
    prev_week_sales: Optional[float] = None
    avg_7d: Optional[float] = None
    avg_30d: Optional[float] = None
    current_stock: Optional[int] = None


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

@app.post("/predict")
def predict(b: PredictIn):

    h = _hist(b.product_id)

    date = (
        pd.Timestamp(b.date)
        if b.date
        else h.date.max() + timedelta(days=1)
    )

    base_price = PRODUCTS[b.product_id][2]

    price = (
        b.price
        or round(
            base_price * (1 - b.discount / 100)
        )
    )

    prev = (
        b.prev_week_sales
        if b.prev_week_sales is not None
        else float(h.units_sold.iloc[-7])
    )

    a7 = (
        b.avg_7d
        if b.avg_7d is not None
        else float(h.units_sold.tail(7).mean())
    )

    a30 = (
        b.avg_30d
        if b.avg_30d is not None
        else float(h.units_sold.tail(30).mean())
    )

    hol = (
        b.is_holiday
        if b.is_holiday is not None
        else is_holiday(date)
    )

    x = _row(
        b.product_id,
        date,
        price,
        b.discount,
        b.marketing_spend,
        hol,
        prev,
        a7,
        a30,
    )

    units = max(
        0,
        round(float(model.predict(x)[0]))
    )

    change = (
        round(
            (units - prev) / prev * 100,
            1
        )
        if prev
        else 0.0
    )

    insights = [
        f"Expected "
        f"{'increase' if change >= 0 else 'decrease'} "
        f"of {abs(change)}% vs the same day last week."
    ]

    if b.discount >= 10:
        insights.append(
            f"A {b.discount:g}% discount is boosting "
            f"expected demand for this product."
        )

    if hol:
        insights.append(
            "A holiday/festival window is lifting demand."
        )

    out = {
        "predicted_units": units,
        "expected_revenue": units * price,
        "change_vs_last_week_pct": change,
        "insights": insights,
    }

    if b.current_stock is not None:

        need = round(
            units * 7 * 1.1
        )

        out["inventory"] = {
            "current_stock": b.current_stock,
            "recommended_stock": need,
            "alert": b.current_stock < need,
        }

    return out


# ---------------------------------------------------------
# FORECAST
# ---------------------------------------------------------

@app.get("/forecast/{product_id}")
def forecast(
    product_id: str,
    days: int = 7
):
    """
    Recursive forecast:
    each predicted day feeds the next day's lag features.
    """

    if not 1 <= days <= 90:
        raise HTTPException(
            400,
            "days must be 1-90"
        )

    h = _hist(product_id)

    sales = list(
        h.units_sold.astype(float)
    )

    last = h.iloc[-30:]

    price = float(last.price.mean())
    disc = float(last.discount.mean())
    mkt = float(last.marketing_spend.mean())

    out = []
    d0 = h.date.max()

    for i in range(1, days + 1):

        d = d0 + timedelta(days=i)

        x = _row(
            product_id,
            d,
            price,
            disc,
            mkt,
            is_holiday(d),
            sales[-7],
            sum(sales[-7:]) / 7,
            sum(sales[-30:]) / 30,
        )

        u = max(
            0.0,
            float(model.predict(x)[0])
        )

        sales.append(u)

        out.append({
            "date": str(d.date()),
            "predicted_units": round(u),
        })

    return {
        "product_id": product_id,
        "forecast": out,
        "total_units": sum(
            o["predicted_units"]
            for o in out
        ),
    }


# ---------------------------------------------------------
# INSIGHTS
# ---------------------------------------------------------

@app.get("/insights")
def insights():

    recent = df[
        df.date >
        df.date.max() - timedelta(days=28)
    ]

    prior = df[
        (df.date <= df.date.max() - timedelta(days=28))
        &
        (df.date > df.date.max() - timedelta(days=56))
    ]

    r = (
        recent.groupby("product_name")
        .units_sold
        .sum()
    )

    p = (
        prior.groupby("product_name")
        .units_sold
        .sum()
    )

    growth = (
        ((r - p) / p * 100)
        .round(1)
        .sort_values()
    )

    return {
        "best_sellers": (
            r.sort_values(ascending=False)
            .head(3)
            .to_dict()
        ),

        "low_performers": (
            r.sort_values()
            .head(3)
            .to_dict()
        ),

        "growth_pct_28d": growth.to_dict(),

        "top_category": (
            recent.groupby("category")
            .revenue
            .sum()
            .idxmax()
        ),

        "alerts": [
            f"{n} demand is down "
            f"{abs(g)}% over the last 4 weeks."
            for n, g in growth.items()
            if g < -10
        ],
    }


# ---------------------------------------------------------
# CSV UPLOAD
# ---------------------------------------------------------

@app.post("/upload")
async def upload(
    file: UploadFile = File(...)
):
    """
    Upload your own CSV, validate it,
    then retrain via `python train.py`.
    """

    global df

    try:

        new = clean(
            pd.read_csv(
                io.BytesIO(
                    await file.read()
                )
            )
        )

    except Exception as e:

        raise HTTPException(
            400,
            str(e)
        )

    new.to_csv(
        DATA,
        index=False
    )

    df = new

    return {
        "rows": len(new),
        "next_step":
            "Run `python train.py` and restart "
            "the API to use the new data."
    }


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "E-Commerce Sales Predictor API is running"
    }