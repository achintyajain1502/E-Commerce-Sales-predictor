"""Shared constants and feature engineering (used by training AND the API)."""
import pandas as pd

PRODUCTS = {  # id: (name, category, base_price, base_daily_demand)
    "P1001": ("Wireless Headphones", "Electronics", 2499, 60),
    "P1002": ("Laptop", "Electronics", 55999, 12),
    "P1003": ("Mechanical Keyboard", "Electronics", 3499, 30),
    "P1004": ("Cotton T-Shirt", "Clothing", 699, 90),
    "P1005": ("Running Shoes", "Clothing", 2999, 40),
    "P1006": ("Air Fryer", "Home Appliances", 6499, 22),
    "P1007": ("Mixer Grinder", "Home Appliances", 3999, 25),
    "P1008": ("Yoga Mat", "Sports", 899, 35),
}

_HOLIDAY_DATES = ["2024-01-26", "2024-03-25", "2024-08-15", "2024-11-01", "2024-12-25",
                  "2025-01-26", "2025-03-14", "2025-08-15", "2025-10-20", "2025-12-25",
                  "2026-01-26", "2026-03-04", "2026-08-15", "2026-11-08", "2026-12-25"]
HOLIDAYS = pd.to_datetime(_HOLIDAY_DATES)


def is_holiday(date) -> int:
    """1 if within 2 days of a festival/holiday."""
    d = pd.Timestamp(date)
    return int(any(abs((d - h).days) <= 2 for h in HOLIDAYS))


NUMERIC = ["price", "discount", "marketing_spend", "is_holiday", "is_weekend",
           "day_of_week", "month", "week_of_year", "prev_week_sales", "avg_7d", "avg_30d"]
CATEGORICAL = ["category"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "units_sold"


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag / rolling features per product (only uses past data -> no leakage)."""
    df = add_calendar(df).sort_values(["product_id", "date"])
    g = df.groupby("product_id")[TARGET]
    df["prev_week_sales"] = g.shift(7)
    df["avg_7d"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["avg_30d"] = g.transform(lambda s: s.shift(1).rolling(30).mean())
    return df.dropna(subset=["prev_week_sales", "avg_7d", "avg_30d"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Module 1: validation, duplicates, missing values."""
    required = {"date", "product_id", "category", "price", "discount",
                "marketing_spend", "is_holiday", "units_sold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.drop_duplicates(subset=["date", "product_id"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["price", "discount", "marketing_spend", "units_sold"]:
        df[c] = df.groupby("product_id")[c].transform(lambda s: s.fillna(s.median()))
    return df[df["units_sold"] >= 0]
