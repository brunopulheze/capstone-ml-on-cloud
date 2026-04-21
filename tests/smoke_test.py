"""
Smoke test for the btc-predictor API.

Requires the container (or uvicorn) to be running before executing.

  Live Render: python tests/smoke_test.py --url https://btc-predictor-ngio.onrender.com
  Docker:      python tests/smoke_test.py --port 8081
  Local:       python tests/smoke_test.py --port 8080

Use --url for any full base URL, or --port to target localhost:<port>.
"""

import argparse
import sys
import requests
import yfinance as yf


def run(base_url: str) -> None:
    # --- health / root ---
    print(f"Target: {base_url}")

    r = requests.get(f"{base_url}/", timeout=60)
    r.raise_for_status()
    print(f"GET /        → {r.status_code} {r.json()}")

    r = requests.get(f"{base_url}/health", timeout=60)
    r.raise_for_status()
    print(f"GET /health  → {r.status_code} {r.json()}")

    # --- predict ---
    print("Downloading BTC-USD history …")
    df = yf.download("BTC-USD", start="2014-01-01", progress=False)
    df.columns = df.columns.get_level_values(0)
    prices = df["Close"].dropna().tolist()
    print(f"Sending {len(prices)} price points …")

    r = requests.post(
        f"{base_url}/predict",
        json={"prices": prices},
        timeout=120,
    )
    r.raise_for_status()
    payload = r.json()
    print(f"POST /predict → {r.status_code} {payload}")

    predicted = payload["predicted_price"]
    previous  = payload["previous_close"]
    assert isinstance(predicted, float), "predicted_price must be a float"
    assert isinstance(previous,  float), "previous_close must be a float"
    assert predicted > 0,                "predicted_price must be positive"

    print("\nAll assertions passed — smoke test OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API smoke test")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--url",  type=str,
                       help="Full base URL to test (e.g. https://btc-predictor-ngio.onrender.com)")
    group.add_argument("--port", type=int, default=8081,
                       help="Localhost port to test (default: 8081)")
    args = parser.parse_args()

    base_url = args.url if args.url else f"http://localhost:{args.port}"

    try:
        run(base_url)
    except Exception as exc:
        print(f"\nSmoke test FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
