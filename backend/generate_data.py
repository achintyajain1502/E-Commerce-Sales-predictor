"""Creates a realistic synthetic dataset (swap with your real CSV later)."""
import numpy as np, pandas as pd
from common import PRODUCTS, is_holiday

rng = np.random.default_rng(42)
dates = pd.date_range("2024-01-01", "2026-09-27")
rows = []
for pid, (name, cat, price0, base) in PRODUCTS.items():
    stock = base * 12
    for i, d in enumerate(dates):
        discount = rng.choice([0, 0, 0, 5, 10, 15, 20, 30], p=[.3, .2, .1, .1, .1, .08, .07, .05])
        price = round(price0 * (1 - discount / 100))
        marketing = int(rng.uniform(2000, 20000))
        hol = is_holiday(d)
        trend = 1 + 0.0004 * i
        season = 1 + 0.15 * np.sin(2 * np.pi * (d.dayofyear - 60) / 365)
        lam = (base * trend * season * (1.25 if d.dayofweek >= 5 else 1)
               * (1.5 if hol else 1) * (1 + 0.02 * discount) * (1 + marketing / 100000))
        units = int(rng.poisson(max(lam, 1)))
        stock = stock - units if stock > units else base * 12
        rows.append(dict(date=d.date(), product_id=pid, product_name=name, category=cat,
                         price=price, discount=discount, marketing_spend=marketing,
                         is_holiday=hol, stock=int(stock), units_sold=units,
                         revenue=units * price))
pd.DataFrame(rows).to_csv("data/sales.csv", index=False)
print(f"Wrote {len(rows)} rows to data/sales.csv")
