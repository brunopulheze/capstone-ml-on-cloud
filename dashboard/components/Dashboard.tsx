"use client";

import { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import LivePriceCard from "./LivePriceCard";

// ── Types ──────────────────────────────────────────────────────────────

interface PredictionData {
  predicted_price: number;
  previous_close: number;
  last_data_date: string;
  data_points: number;
}

interface DriftReport {
  available: boolean;
  message?: string;
  timestamp?: string;
  recent_mae?: number;
  baseline_rmse?: number;
  drift_threshold?: number;
  drift_detected?: boolean;
  retrained?: boolean;
  new_rmse?: number;
}

interface PricePoint {
  date: string;
  price: number;
}

interface DashboardProps {
  prediction: PredictionData | null;
  driftReport: DriftReport | null;
  priceHistory: PricePoint[] | null;
  currentPrice: number | null;
}

// ── Formatters ─────────────────────────────────────────────────────────

function fmtUSD(n: number) {
  return "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPct(n: number) {
  return (n > 0 ? "+" : "") + n.toFixed(2) + "%";
}

function fmtDate(iso: string) {
  const d = new Date(iso + "T00:00:00Z");
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

// ── Custom chart pieces ────────────────────────────────────────────────

interface TooltipEntry {
  date: string;
  price?: number | null;
  predicted?: number | null;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: { payload: TooltipEntry }[];
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip__date">{label}</p>
      {d.price != null && (
        <p className="chart-tooltip__row chart-tooltip__row--actual">
          <span>Actual</span>
          <span>{fmtUSD(d.price)}</span>
        </p>
      )}
      {d.predicted != null && (
        <p className="chart-tooltip__row chart-tooltip__row--predicted">
          <span>Predicted</span>
          <span>{fmtUSD(d.predicted)}</span>
        </p>
      )}
    </div>
  );
}

interface DotProps {
  cx?: number;
  cy?: number;
  payload?: TooltipEntry;
  index?: number;
}

function PredictionDot({ cx, cy, payload }: DotProps) {
  // Only render the dot for the "tomorrow" point (where price is null/undefined)
  if (!payload || payload.price != null || payload.predicted == null || cx == null || cy == null) {
    return <g />;
  }
  return (
    <g>
      <circle cx={cx} cy={cy} r={18} fill="hsl(38, 92%, 50%)" fillOpacity={0.12} />
      <circle cx={cx} cy={cy} r={8} fill="hsl(38, 92%, 50%)" stroke="white" strokeWidth={2.5} />
    </g>
  );
}

// ── Main component ─────────────────────────────────────────────────────

export default function Dashboard({ prediction, driftReport, priceHistory, currentPrice }: DashboardProps) {
  const pricePct =
    prediction
      ? ((prediction.predicted_price - prediction.previous_close) / prediction.previous_close) * 100
      : null;
  const isPositive = pricePct !== null && pricePct >= 0;

  // Build chart dataset: 60-day history + prediction bridge + tomorrow
  const chartData = useMemo(() => {
    if (!priceHistory) return [];
    const history = priceHistory.slice(-60);

    if (!prediction) {
      return history.map((h) => ({ date: h.date, price: h.price, predicted: null as number | null }));
    }

    // Tomorrow's date
    const lastDate = new Date(prediction.last_data_date + "T00:00:00Z");
    lastDate.setUTCDate(lastDate.getUTCDate() + 1);
    const tomorrowStr = lastDate.toISOString().split("T")[0];

    const result = history.map((h, i) => ({
      date: h.date,
      price: h.price,
      // Bridge: last history point also starts the predicted series
      predicted: i === history.length - 1 ? (h.price as number | null) : (null as number | null),
    }));

    // Append tomorrow's prediction
    result.push({
      date: tomorrowStr,
      price: null as unknown as number,
      predicted: Math.round(prediction.predicted_price),
    });

    return result;
  }, [priceHistory, prediction]);

  const [minPrice, maxPrice] = useMemo(() => {
    if (!chartData.length) return [0, "auto"] as [number, string];
    const vals = chartData.flatMap((d) =>
      [d.price, d.predicted].filter((v): v is number => v != null)
    );
    const min = Math.min(...vals) * 0.96;
    const max = Math.max(...vals) * 1.02;
    return [min, max] as [number, number];
  }, [chartData]);

  const now = new Date().toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });

  const driftStatusBadge = driftReport?.available
    ? driftReport.drift_detected
      ? "btc-badge--warn"
      : "btc-badge--ok"
    : "btc-badge--live";

  const driftStatusLabel = driftReport?.available
    ? driftReport.drift_detected
      ? "DRIFT DETECTED"
      : "HEALTHY"
    : "MONITORING ACTIVE";

  return (
    <div className="btc-dashboard">
      {/* ── Header ── */}
      <header className="btc-header">
        <div className="btc-header__left">
          <div className="btc-logo">
            <span className="btc-logo__symbol">₿</span>
          </div>
          <div>
            <h1 className="btc-header__title">BTC Price Predictor</h1>
            <p className="btc-header__subtitle">
              GRU Neural Network · Oracle Cloud · Daily Next-Day Forecast
            </p>
          </div>
        </div>
        <div className="btc-header__right">
          <span className="btc-badge btc-badge--live">
            <span className="btc-live-dot" />
            Live
          </span>
          <span className="btc-timestamp">{now}</span>
        </div>
      </header>

      {/* ── API error banner ── */}
      {!prediction && (
        <div className="btc-error">
          ⚠ Could not reach the prediction API (http://138.2.180.250:8080). Check that the Oracle
          Cloud VM is running.
        </div>
      )}

      {/* ── Stat cards ── */}
      <div className="btc-stats">
        <div className="btc-card btc-card--stat">
          <p className="btc-card__label">Tomorrow&apos;s Prediction</p>
          <p className="btc-card__value btc-card__value--lg btc-card__value--primary">
            {prediction ? fmtUSD(prediction.predicted_price) : "—"}
          </p>
          {pricePct !== null && (
            <span className={`btc-badge ${isPositive ? "btc-badge--up" : "btc-badge--down"}`}>
              {isPositive ? "▲" : "▼"} {Math.abs(pricePct).toFixed(2)}%
            </span>
          )}
        </div>

        <LivePriceCard
          initialPrice={currentPrice}
          previousClose={prediction?.previous_close ?? null}
        />

        <div className="btc-card btc-card--stat">
          <p className="btc-card__label">Previous Close</p>
          <p className="btc-card__value">
            {prediction ? fmtUSD(prediction.previous_close) : "—"}
          </p>
          <p className="btc-card__meta">
            {prediction ? `As of ${prediction.last_data_date}` : "—"}
          </p>
        </div>

        <div className="btc-card btc-card--stat">
          <p className="btc-card__label">Predicted Change</p>
          <p
            className={`btc-card__value ${pricePct == null ? "" : isPositive ? "btc-card__value--up" : "btc-card__value--down"
              }`}
          >
            {pricePct !== null ? fmtPct(pricePct) : "—"}
          </p>
          <p className="btc-card__meta">
            {prediction
              ? fmtUSD(Math.abs(prediction.predicted_price - prediction.previous_close)) + " USD"
              : ""}
          </p>
        </div>
      </div>

      {/* ── Price chart ── */}
      <div className="btc-card btc-card--chart">
        <div className="btc-chart-header">
          <h2 className="btc-card__title" style={{ marginBottom: 0 }}>
            BTC/USD — 60-Day History &amp; Next-Day Prediction
          </h2>
          <div className="btc-chart-legend">
            <span className="btc-legend-item btc-legend-item--actual">
              <svg width="20" height="3" viewBox="0 0 20 3" aria-hidden="true">
                <line x1="0" y1="1.5" x2="20" y2="1.5" stroke="hsl(217,91%,60%)" strokeWidth="2" />
              </svg>
              Actual Price
            </span>
            <span className="btc-legend-item btc-legend-item--predicted">
              <svg width="20" height="3" viewBox="0 0 20 3" aria-hidden="true">
                <line
                  x1="0" y1="1.5" x2="20" y2="1.5"
                  stroke="hsl(38,92%,50%)" strokeWidth="2" strokeDasharray="5 3"
                />
              </svg>
              Predicted (Tomorrow)
            </span>
          </div>
        </div>

        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={340}>
            <AreaChart
              data={chartData}
              margin={{ top: 12, right: 16, left: 4, bottom: 0 }}
            >
              <defs>
                <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(217,91%,60%)" stopOpacity={0.22} />
                  <stop offset="95%" stopColor="hsl(217,91%,60%)" stopOpacity={0} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="hsl(258,30%,93%)" vertical={false} />

              <XAxis
                dataKey="date"
                tick={{ fontSize: 11, fill: "hsl(244,16%,52%)" }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v: string) => fmtDate(v)}
                interval={Math.floor(chartData.length / 7)}
              />

              <YAxis
                domain={[minPrice, maxPrice]}
                tick={{ fontSize: 11, fill: "hsl(244,16%,52%)" }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
                width={52}
              />

              <Tooltip content={<CustomTooltip />} />

              {/* Actual price area */}
              <Area
                type="monotone"
                dataKey="price"
                stroke="hsl(217,91%,60%)"
                strokeWidth={2}
                fill="url(#priceGrad)"
                dot={false}
                activeDot={{ r: 4, fill: "hsl(217,91%,60%)", stroke: "white", strokeWidth: 2 }}
                connectNulls={false}
                isAnimationActive={true}
              />

              {/* Predicted extension */}
              <Area
                type="monotone"
                dataKey="predicted"
                stroke="hsl(38,92%,50%)"
                strokeWidth={2.5}
                strokeDasharray="7 4"
                fill="none"
                fillOpacity={0}
                dot={<PredictionDot />}
                activeDot={false}
                connectNulls={false}
                isAnimationActive={true}
              />

              {/* Horizontal guide at predicted price */}
              {prediction && (
                <ReferenceLine
                  y={prediction.predicted_price}
                  stroke="hsl(38,92%,50%)"
                  strokeDasharray="4 4"
                  strokeOpacity={0.35}
                  strokeWidth={1}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="btc-chart-empty">
            {priceHistory === null
              ? "Price history unavailable — CoinGecko API may be rate-limited. Refresh in a moment."
              : "No chart data to display."}
          </div>
        )}
      </div>

      {/* ── Info row ── */}
      <div className="btc-info-row">
        {/* Model architecture */}
        <div className="btc-card">
          <h2 className="btc-card__title">Model Architecture</h2>
          <div className="btc-model-grid">
            {[
              ["Type", "GRU(64) → Dense(1)"],
              ["Lookback window", "20 days"],
              ["Feature set", "100 price lags · RSI-14 · MACD · std(30) · log-return"],
              ["Test RMSE", "≈ $1,901"],
              ["Target variable", "Log-return (reconstructed to price)"],
              ["Training data", prediction ? `${prediction.data_points.toLocaleString()} daily closes since 2014` : "Daily closes since 2014"],
              ["Deployed on", "Oracle Cloud VM · eu-frankfurt-1 · Always Free"],
            ].map(([label, value]) => (
              <div className="btc-model-item" key={label}>
                <span className="btc-model-item__label">{label}</span>
                <span className="btc-model-item__value">{value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Drift monitoring */}
        <div className="btc-card">
          <h2 className="btc-card__title">
            Drift Monitoring
            <span className={`btc-badge btc-badge--sm ${driftStatusBadge}`}>
              {driftStatusLabel}
            </span>
          </h2>

          {driftReport?.available ? (
            <div className="btc-model-grid">
              {[
                [
                  "Last checked",
                  driftReport.timestamp
                    ? new Date(driftReport.timestamp).toLocaleString()
                    : "—",
                ],
                ["Recent MAE (30d)", driftReport.recent_mae != null ? fmtUSD(driftReport.recent_mae) : "—"],
                ["Baseline RMSE", driftReport.baseline_rmse != null ? fmtUSD(driftReport.baseline_rmse) : "—"],
                ["Drift threshold", driftReport.drift_threshold != null ? fmtUSD(driftReport.drift_threshold) : "—"],
                [
                  "Retrained this run",
                  driftReport.retrained
                    ? `Yes${driftReport.new_rmse != null ? ` (new RMSE: ${fmtUSD(driftReport.new_rmse)})` : ""}`
                    : "No",
                ],
              ].map(([label, value]) => (
                <div className="btc-model-item" key={label}>
                  <span className="btc-model-item__label">{label}</span>
                  <span className="btc-model-item__value">{value}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="btc-model-grid">
              {[
                ["Evaluation window", "Last 30 days"],
                ["Drift threshold", "1.5 × baseline RMSE (~$2,850)"],
                ["Schedule", "Daily at 06:00 UTC — GitHub Actions"],
                ["Auto-redeploy", "Docker Hub → Oracle Cloud VM"],
                ["Status", driftReport?.message ?? "Awaiting first scheduled run"],
              ].map(([label, value]) => (
                <div className="btc-model-item" key={label}>
                  <span className="btc-model-item__label">{label}</span>
                  <span className="btc-model-item__value">{value}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Footer ── */}
      <footer className="btc-footer">
        <span>Price history: CoinGecko API</span>
        <span>Predictions: GRU model · Oracle Cloud Infrastructure</span>
        <span>Pipeline: GitHub Actions · Docker Hub · yfinance</span>
      </footer>
    </div>
  );
}
