#!/usr/bin/env python3
"""
Train the universal prediction checkpoint.

This trains ONE model on a diverse cross-section of ~40 stocks (NSE + US)
and saves it to models/. Every page of the app then loads this checkpoint
instantly instead of training a fresh model per request.

Requires internet access to Yahoo Finance — run this on your local machine
or anywhere with normal internet access (it will NOT work in network-
restricted sandboxes). Takes roughly 2-5 minutes depending on connection
speed and universe size.

Usage:
    python scripts/train_universal_model.py
    python scripts/train_universal_model.py --universe AAPL,MSFT,TCS.NS --period 2y

After training, commit the generated files to your GitHub repo so the
checkpoint persists across Streamlit Cloud redeploys:
    git add models/
    git commit -m "Train universal prediction model"
    git push
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.models import train_universal_model, DEFAULT_TRAIN_UNIVERSE


def cli_progress(pct: float, msg: str):
    bar_len = 40
    filled = int(bar_len * pct)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] {pct*100:5.1f}%  {msg:<50}", end="", flush=True)
    if pct >= 1.0:
        print()


def main():
    parser = argparse.ArgumentParser(description="Train the universal stock prediction checkpoint.")
    parser.add_argument("--universe", type=str, default=None,
                        help="Comma-separated tickers, e.g. AAPL,MSFT,TCS.NS. "
                             "Defaults to a built-in diverse 40-stock universe.")
    parser.add_argument("--period", type=str, default="5y",
                        help="History length to fetch per ticker (default: 5y)")
    args = parser.parse_args()

    universe = (args.universe.split(",") if args.universe
                else DEFAULT_TRAIN_UNIVERSE)
    universe = [u.strip().upper() for u in universe if u.strip()]

    print("=" * 70)
    print("  StockPro Universal Model Trainer")
    print("=" * 70)
    print(f"  Universe : {len(universe)} tickers")
    print(f"  Period   : {args.period}")
    print(f"  Output   : models/universal_xgb.json, universal_lgb.txt, universal_meta.json")
    print("=" * 70)
    print()

    try:
        meta = train_universal_model(
            universe=universe, period=args.period, progress_callback=cli_progress
        )
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        print("\nMost common cause: no internet access to Yahoo Finance from this")
        print("environment. Run this script somewhere with normal internet access")
        print("(your local machine, a CI runner, etc.) — NOT inside a network-")
        print("restricted sandbox.")
        sys.exit(1)

    print()
    print("=" * 70)
    print("  ✅ Training complete")
    print("=" * 70)
    print(f"  Tickers used        : {meta['n_tickers_used']}")
    print(f"  Training rows        : {meta['n_train_rows']:,}")
    print(f"  Test rows             : {meta['n_test_rows']:,}")
    print(f"  Test MAE (log return) : {meta['test_mae']:.6f}")
    print(f"  Test directional acc. : {meta['test_directional_accuracy']:.1f}%")
    print(f"  Models trained        : {', '.join(meta['models_trained'])}")
    print()
    print("  Per-ticker directional accuracy on held-out test data:")
    for t, acc in sorted(meta["per_ticker_directional_accuracy"].items(),
                         key=lambda x: -x[1])[:10]:
        print(f"    {t:<14} {acc:5.1f}%")
    print()
    print("  Next step — commit the checkpoint so it persists on Streamlit Cloud:")
    print("    git add models/")
    print("    git commit -m \"Train universal prediction model\"")
    print("    git push")
    print("=" * 70)


if __name__ == "__main__":
    main()
