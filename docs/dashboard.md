# BTC Price Predictor — Dashboard

A full-screen single-page dashboard built with **Next.js 15**, **Recharts**, and **Tailwind CSS**, inspired by the BrixoAI design system. Live at the Vercel deployment linked from the repository.

---

## Stack

| Technology | Version | Role |
|---|---|---|
| Next.js (App Router) | 15.3.6 | Framework — SSR + static generation |
| Recharts | 2.x | Area chart with actual vs predicted series |
| Tailwind CSS | 3.x | Utility classes for layout and spacing |
| Google Fonts — Rubik | — | Primary typeface |

---

## Project Structure

```
dashboard/
├── app/
│   ├── globals.css        # Design tokens and component styles
│   ├── icon.png           # Favicon (Bitcoin ₿ logo)
│   ├── layout.tsx         # Root layout — loads Rubik font, sets metadata
│   └── page.tsx           # Server component — data fetching
└── components/
    └── Dashboard.tsx      # Client component — all widgets and chart
```

### File Responsibilities

| File | Purpose |
|---|---|
| `app/page.tsx` | Server component. Fetches `/predict/latest`, `/drift-report`, and CoinGecko 90-day price history **in parallel** at build/revalidation time (ISR: 30-minute window). Passes data as props to `Dashboard`. |
| `components/Dashboard.tsx` | `"use client"` component. Renders all widgets, builds the chart dataset (60-day history + tomorrow's prediction bridge), and handles null/loading states when the API is unreachable. |
| `app/globals.css` | BrixoAI-inspired CSS design tokens and all component class definitions. No external CSS framework beyond Tailwind resets. |

---

## Design System

Colours, typography, and spacing mirror the BrixoAI client portal:

| Token | Value | Usage |
|---|---|---|
| `--first-color` | `hsl(217, 91%, 60%)` | Primary blue — predicted price, links |
| `--body-color` | `hsl(258, 60%, 98%)` | Page background (light lavender) |
| `--container-color` | `#fff` | Card backgrounds |
| `--title-color` | `hsl(244, 24%, 26%)` | Headings and large values |
| `--text-color` | `hsl(244, 16%, 43%)` | Labels, metadata, muted text |
| `--shadow` | `0px 5px 20px 0px rgb(69 67 96 / 10%)` | Card elevation |
| `--border-radius` | `20px` | Card corners |
| Font | Rubik 400/500/600/700 | All text |

---

## Page Layout (top → bottom)

1. **Header** — Bitcoin logo, title, subtitle, live badge, current timestamp
2. **KPI stat cards** (4-column grid)
3. **60-day area chart**
4. **Info row** — Model Architecture card + Drift Monitoring card (2 columns)
5. **Footer** — data source attributions

---

## Widgets

### KPI Stat Cards

| Card | Data source | Notes |
|---|---|---|
| Tomorrow's Prediction | `GET /predict/latest` → `predicted_price` | Large primary-blue value |
| Previous Close | `GET /predict/latest` → `previous_close`, `last_data_date` | Shows the date of the last known close |
| Predicted Change | Derived: `(predicted − previous) / previous × 100` | Green badge (▲) or red badge (▼) |
| Training Data Points | `GET /predict/latest` → `data_points` | Total daily closes used (from 2014) |

### 60-Day Area Chart

Built with Recharts `AreaChart`:

- **Actual price series** (blue, filled gradient) — last 60 days of CoinGecko daily closes
- **Predicted extension** (amber, dashed stroke, no fill) — bridges from the last actual close to tomorrow's predicted price
- **Prediction dot** — glowing amber circle on the tomorrow point
- **Reference line** — faint horizontal guide at the predicted price level
- **Custom tooltip** — shows date, actual price, and predicted price on hover
- Data revalidates every 30 minutes (Next.js ISR)

### Model Architecture Card

Static display of model metadata read from `models/selection.json` at deploy time:

- Type: `RandomForestRegressor(n_estimators=300)`
- Feature window: 20-day lags + RSI-14 + MACD + std(30) + log-return
- Feature count: 25
- Test RMSE: ≈ $622
- Target variable: Log-return (reconstructed to price)
- Deployed on: Oracle Cloud VM · eu-frankfurt-1 · Always Free

### Drift Monitoring Card

Live data from `GET /drift-report`:

| Field | Description |
|---|---|
| Last checked | Timestamp of the most recent GitHub Actions drift evaluation |
| Recent MAE (30d) | Mean absolute error over the last 30 days of predictions |
| Baseline RMSE | RMSE from the training run saved in `selection.json` |
| Drift threshold | `1.5 × baseline RMSE` — triggers retraining if exceeded |
| Retrained this run | Whether the model was retrained and the new RMSE if so |

If no drift report exists yet (before the first scheduled GitHub Actions run), the card shows the pipeline configuration instead.

---

## API Endpoints Used

| Endpoint | Method | Purpose |
|---|---|---|
| `http://138.2.180.250:8080/predict/latest` | `GET` | Autonomous next-day prediction — no input required |
| `http://138.2.180.250:8080/drift-report` | `GET` | Latest drift detection report from `models/drift_report.json` |
| `https://api.coingecko.com/api/v3/coins/bitcoin/market_chart` | `GET` | 90-day BTC/USD daily price history for the chart |

CORS is enabled on the FastAPI backend (`allow_origins=["*"]`) so the dashboard can call the Oracle Cloud API directly from the browser if needed.

---

## Running Locally

```bash
cd dashboard
npm install
npm run dev     # → http://localhost:3001
```

Production build check:

```bash
npm run build
```
