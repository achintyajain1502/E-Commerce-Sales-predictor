const BASE = import.meta.env.VITE_API || "http://127.0.0.1:8000";

async function call(path, options) {
  const res = await fetch(BASE + path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export const api = {
  products: () => call("/products"),
  summary: () => call("/analytics/summary"),
  insights: () => call("/insights"),
  metrics: () => call("/models/metrics"),
  forecast: (id, days) => call(`/forecast/${id}?days=${days}`),
  predict: (body) =>
    call("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
};

export const inr = (n) => "₹" + Math.round(n).toLocaleString("en-IN");
export const inrShort = (n) =>
  n >= 1e7 ? `₹${(n / 1e7).toFixed(2)} Cr` : n >= 1e5 ? `₹${(n / 1e5).toFixed(1)} L` : inr(n);
