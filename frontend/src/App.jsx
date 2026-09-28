import { useEffect, useState } from "react";
import { ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { api, inr, inrShort } from "./api.js";

const C = { blue: "#2f5bea", amber: "#e9a23b", grid: "#dfe4ea", ink: "#14213d" };
const axis = { stroke: "#6b7686", fontSize: 12, tickLine: false, axisLine: false };

function useLoad(fn, deps = []) {
  const [state, set] = useState({ data: null, error: null, loading: true });
  useEffect(() => {
    let live = true;
    set((s) => ({ ...s, loading: true, error: null }));
    fn().then((data) => live && set({ data, error: null, loading: false }))
        .catch((e) => live && set({ data: null, error: e.message, loading: false }));
    return () => { live = false; };
  }, deps); // eslint-disable-line
  return state;
}

const Status = ({ s }) =>
  s.loading ? <p className="muted">Loading…</p> :
  s.error ? <p className="error">Can't reach the API: {s.error}. Check that uvicorn is running on port 8000.</p> : null;

const objRows = (o, key = "name") => Object.entries(o).map(([k, v]) => ({ [key]: k, value: v }));

function Dashboard() {
  const s = useLoad(api.summary);
  const ins = useLoad(api.insights);
  if (!s.data) return <Status s={s} />;
  const d = s.data;
  return (
    <>
      <div className="kpis">
        <div><span>Units sold</span><b>{d.total_units.toLocaleString("en-IN")}</b></div>
        <div><span>Revenue</span><b>{inrShort(d.total_revenue)}</b></div>
        <div><span>Revenue per unit</span><b>{inr(d.avg_revenue_per_unit)}</b></div>
      </div>

      <section>
        <h2>Monthly revenue</h2>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={d.monthly_trend}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="m" {...axis} minTickGap={24} />
            <YAxis {...axis} tickFormatter={inrShort} width={70} />
            <Tooltip formatter={(v) => inrShort(v)} />
            <Line dataKey="revenue" stroke={C.blue} strokeWidth={2.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <div className="two">
        <section>
          <h2>Top products by revenue</h2>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={objRows(d.top_products)} layout="vertical" margin={{ left: 40 }}>
              <XAxis type="number" hide />
              <YAxis type="category" dataKey="name" {...axis} width={130} />
              <Tooltip formatter={(v) => inrShort(v)} />
              <Bar dataKey="value" fill={C.blue} radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </section>
        <section>
          <h2>Average daily units by discount</h2>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={objRows(d.discount_vs_sales)}>
              <CartesianGrid stroke={C.grid} vertical={false} />
              <XAxis dataKey="name" {...axis} />
              <YAxis {...axis} width={40} />
              <Tooltip />
              <Bar dataKey="value" fill={C.amber} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </section>
      </div>

      {ins.data && (
        <section>
          <h2>What needs attention</h2>
          <p>Best sellers (last 4 weeks): {Object.keys(ins.data.best_sellers).join(", ")}.</p>
          <p>Slowest movers: {Object.keys(ins.data.low_performers).join(", ")}.</p>
          <p>Top category by revenue: {ins.data.top_category}.</p>
          {ins.data.alerts.length === 0 ? <p className="muted">No products are losing demand right now.</p>
            : ins.data.alerts.map((a) => <p key={a} className="error">{a}</p>)}
        </section>
      )}
    </>
  );
}

function ProductSelect({ products, value, onChange }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {products.map((p) => <option key={p.product_id} value={p.product_id}>{p.name}</option>)}
    </select>
  );
}

function Predict({ products }) {
  const [f, setF] = useState({ product_id: products[0].product_id, discount: 15, marketing_spend: 15000, holiday: "auto", current_stock: 120 });
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const set = (k) => (v) => setF((p) => ({ ...p, [k]: v }));

  async function submit() {
    setBusy(true); setErr(null);
    try {
      const body = { product_id: f.product_id, discount: +f.discount, marketing_spend: +f.marketing_spend,
                     current_stock: +f.current_stock };
      if (f.holiday !== "auto") body.is_holiday = +f.holiday;
      setRes(await api.predict(body));
    } catch (e) { setErr(e.message); setRes(null); }
    setBusy(false);
  }

  return (
    <div className="two">
      <section>
        <h2>Predict tomorrow's sales</h2>
        <label>Product<ProductSelect products={products} value={f.product_id} onChange={set("product_id")} /></label>
        <label>Discount (%)<input type="number" min="0" max="90" value={f.discount} onChange={(e) => set("discount")(e.target.value)} /></label>
        <label>Marketing spend (₹)<input type="number" min="0" value={f.marketing_spend} onChange={(e) => set("marketing_spend")(e.target.value)} /></label>
        <label>Holiday or festival
          <select value={f.holiday} onChange={(e) => set("holiday")(e.target.value)}>
            <option value="auto">Detect from date</option><option value="1">Yes</option><option value="0">No</option>
          </select>
        </label>
        <label>Current stock (units)<input type="number" min="0" value={f.current_stock} onChange={(e) => set("current_stock")(e.target.value)} /></label>
        <button onClick={submit} disabled={busy}>{busy ? "Predicting…" : "Predict sales"}</button>
        {err && <p className="error">{err}</p>}
      </section>

      <section>
        <h2>Result</h2>
        {!res ? <p className="muted">Fill in the details and choose Predict sales.</p> : (
          <>
            <p className="big">{res.predicted_units}<small> units</small></p>
            <p>Expected revenue <b>{inr(res.expected_revenue)}</b></p>
            <p className={res.change_vs_last_week_pct >= 0 ? "good" : "error"}>
              {res.change_vs_last_week_pct >= 0 ? "+" : ""}{res.change_vs_last_week_pct}% vs same day last week
            </p>
            <ul>{res.insights.map((i) => <li key={i}>{i}</li>)}</ul>
            {res.inventory && (
              <div className={res.inventory.alert ? "alert" : "ok"}>
                {res.inventory.alert
                  ? `Restock: you have ${res.inventory.current_stock} units but a week of demand needs about ${res.inventory.recommended_stock}.`
                  : `Stock is enough: ${res.inventory.current_stock} units covers the predicted week (need about ${res.inventory.recommended_stock}).`}
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}

function Forecast({ products }) {
  const [pid, setPid] = useState(products[0].product_id);
  const [days, setDays] = useState(30);
  const fc = useLoad(() => api.forecast(pid, days), [pid, days]);
  const m = useLoad(api.metrics);
  return (
    <>
      <section>
        <h2>Demand forecast</h2>
        <div className="row">
          <label>Product<ProductSelect products={products} value={pid} onChange={setPid} /></label>
          <label>Horizon
            <select value={days} onChange={(e) => setDays(+e.target.value)}>
              <option value={7}>Next 7 days</option><option value={30}>Next 30 days</option><option value={90}>Next 90 days</option>
            </select>
          </label>
        </div>
        {fc.data ? (
          <>
            <p className="big">{fc.data.total_units.toLocaleString("en-IN")}<small> units expected over {days} days</small></p>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={fc.data.forecast}>
                <CartesianGrid stroke={C.grid} vertical={false} />
                <XAxis dataKey="date" {...axis} minTickGap={30} />
                <YAxis {...axis} width={40} />
                <Tooltip />
                <Line dataKey="predicted_units" name="Units" stroke={C.amber} strokeWidth={2.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </>
        ) : <Status s={fc} />}
      </section>
      {m.data && (
        <section>
          <h2>Model comparison</h2>
          <table>
            <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>
            <tbody>{m.data.results.map((r) => (
              <tr key={r.model} className={r.model === m.data.best_model ? "best" : ""}>
                <td>{r.model}{r.model === m.data.best_model && " (in use)"}</td><td>{r.MAE}</td><td>{r.RMSE}</td><td>{r.R2}</td>
              </tr>))}</tbody>
          </table>
          <p className="muted">Lower MAE and RMSE mean smaller errors. R² closer to 1 means a better fit.</p>
        </section>
      )}
    </>
  );
}

const TABS = ["Dashboard", "Predict", "Forecast"];

export default function App() {
  const [tab, setTab] = useState("Dashboard");
  const products = useLoad(api.products);
  return (
    <div className="shell">
      <nav>
        <h1>Sales Predictor</h1>
        {TABS.map((t) => (
          <button key={t} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>{t}</button>
        ))}
      </nav>
      <main>
        <h1 className="page">{tab}</h1>
        {!products.data ? <Status s={products} /> :
          tab === "Dashboard" ? <Dashboard /> :
          tab === "Predict" ? <Predict products={products.data} /> : <Forecast products={products.data} />}
      </main>
    </div>
  );
}
