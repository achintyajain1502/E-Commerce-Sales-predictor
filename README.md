# E-Commerce Sales Forecasting & Predictive Analytics System

## Backend quick start
```bash
cd backend
pip install -r requirements.txt
python generate_data.py     # synthetic data -> data/sales.csv (skip if using your own CSV)
python train.py             # compares LR / RF / XGBoost, saves the best model
uvicorn main:app --reload   # API docs: http://localhost:8000/docs
```

## Endpoints
| Endpoint | Purpose |
|---|---|
| GET /products | product list for dropdowns |
| GET /analytics/summary | KPIs, monthly trend, top products/categories, discount impact |
| GET /models/metrics | MAE / RMSE / R² comparison |
| POST /predict | single prediction + revenue + insights + inventory alert |
| GET /forecast/{id}?days=30 | recursive multi-day forecast |
| GET /insights | best/low performers, growth, alerts |
| POST /upload | upload your own CSV |

Required CSV columns: date, product_id, category, price, discount, marketing_spend, is_holiday, units_sold
(optional: product_name, stock, revenue). Edit `PRODUCTS` in common.py for your own catalogue.

## Next: Angular dashboard (Chart.js + PrimeNG) calling these endpoints.
