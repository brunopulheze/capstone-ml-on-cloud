import Dashboard from "@/components/Dashboard";

export const revalidate = 1800; // revalidate every 30 minutes

const API_BASE = "http://138.2.180.250:8080";

// ── Data fetchers ──────────────────────────────────────────────────────

async function getPrediction() {
  try {
    const res = await fetch(`${API_BASE}/predict/latest`, {
      next: { revalidate: 1800 },
    });
    if (!res.ok) return null;
    return res.json() as Promise<{
      predicted_price: number;
      previous_close: number;
      last_data_date: string;
      data_points: number;
    }>;
  } catch {
    return null;
  }
}

async function getDriftReport() {
  try {
    const res = await fetch(`${API_BASE}/drift-report`, {
      next: { revalidate: 1800 },
    });
    if (!res.ok) return null;
    return res.json() as Promise<{
      available: boolean;
      message?: string;
      timestamp?: string;
      recent_mae?: number;
      baseline_rmse?: number;
      drift_threshold?: number;
      drift_detected?: boolean;
      retrained?: boolean;
      new_rmse?: number;
    }>;
  } catch {
    return null;
  }
}

async function getCurrentPrice() {
  try {
    const res = await fetch(
      "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
      { next: { revalidate: 300 } }
    );
    if (!res.ok) return null;
    const data = await res.json();
    return (data.bitcoin?.usd as number) ?? null;
  } catch {
    return null;
  }
}

async function getPriceHistory() {
  try {
    const res = await fetch(
      "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90&interval=daily",
      { next: { revalidate: 3600 } }
    );
    if (!res.ok) return null;
    const data = await res.json();
    return (data.prices as [number, number][]).map(([ts, price]) => ({
      date: new Date(ts).toISOString().split("T")[0],
      price: Math.round(price),
    }));
  } catch {
    return null;
  }
}

async function getFullPriceHistory() {
  try {
    const res = await fetch(`${API_BASE}/history`, {
      next: { revalidate: 86400 },
    });
    if (!res.ok) return null;
    return res.json() as Promise<{ date: string; price: number }[]>;
  } catch {
    return null;
  }
}

// ── Page ───────────────────────────────────────────────────────────────

export default async function Page() {
  const [prediction, driftReport, priceHistory, fullPriceHistory, currentPrice] = await Promise.all([
    getPrediction(),
    getDriftReport(),
    getPriceHistory(),
    getFullPriceHistory(),
    getCurrentPrice(),
  ]);

  return (
    <Dashboard
      prediction={prediction}
      driftReport={driftReport}
      priceHistory={priceHistory}
      fullPriceHistory={fullPriceHistory}
      currentPrice={currentPrice}
    />
  );
}
