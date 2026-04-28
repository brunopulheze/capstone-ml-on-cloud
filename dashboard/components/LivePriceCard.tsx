"use client";

import { useState, useCallback } from "react";

interface LivePriceCardProps {
    initialPrice: number | null;
    previousClose: number | null;
}

function fmtUSD(n: number) {
    return "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export default function LivePriceCard({ initialPrice, previousClose }: LivePriceCardProps) {
    const [price, setPrice] = useState<number | null>(initialPrice);
    const [loading, setLoading] = useState(false);

    const refresh = useCallback(async () => {
        setLoading(true);
        try {
            const res = await fetch(
                "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
                { cache: "no-store" }
            );
            if (res.ok) {
                const data = await res.json();
                setPrice(data.bitcoin?.usd ?? null);
            }
        } finally {
            setLoading(false);
        }
    }, []);

    const pct =
        price != null && previousClose != null
            ? ((price - previousClose) / previousClose) * 100
            : null;
    const up = pct !== null && pct >= 0;

    return (
        <div className="btc-card btc-card--stat">
            <p className="btc-card__label">
                Today&apos;s Price
                <button
                    className={`btc-refresh-btn${loading ? " btc-refresh-btn--spinning" : ""}`}
                    onClick={refresh}
                    disabled={loading}
                    aria-label="Refresh live price"
                    title="Refresh live price"
                >
                    ↻
                </button>
            </p>
            <p className="btc-card__value">
                {price ? fmtUSD(price) : "—"}
            </p>
            {pct !== null ? (
                <span className={`btc-badge ${up ? "btc-badge--up" : "btc-badge--down"}`}>
                    {up ? "▲" : "▼"} {Math.abs(pct).toFixed(2)}% vs prev close
                </span>
            ) : (
                <p className="btc-card__meta">Live · ~5 min delay</p>
            )}
        </div>
    );
}
