# models/

This folder is intentionally **empty until you train the universal model**.

## Why it's empty

This repo does not ship a pre-trained checkpoint because training requires
fetching real historical data from Yahoo Finance, which needs genuine
internet access (not available in every build/CI environment). You train
it once, yourself, then it works for every ticker forever after.

## How to train

**Option A — In the app (easiest):**
Go to the **🔮 Price Prediction** page → expand **🔧 Train / Retrain
Universal Model** → click **🚀 Train Now**. Takes ~2–5 minutes.

**Option B — Command line:**
```bash
python scripts/train_universal_model.py
```
Optional flags:
```bash
python scripts/train_universal_model.py --universe AAPL,MSFT,TCS.NS,RELIANCE.NS --period 5y
```

## What gets created here

| File | Contents |
|---|---|
| `universal_xgb.json` | Trained XGBoost booster |
| `universal_lgb.txt` | Trained LightGBM booster |
| `universal_scaler.json` | Feature scaling parameters (RobustScaler) |
| `universal_meta.json` | Training metadata: universe, row counts, test accuracy, per-ticker breakdown |

## Making it permanent

Streamlit Cloud's filesystem is **ephemeral** — if you train via the in-app
button, the checkpoint is lost on the next redeploy/restart. To keep it:

```bash
git add models/
git commit -m "Train universal prediction model"
git push
```

Once committed, every deploy loads the same checkpoint instantly — no
retraining needed, and it works on **any ticker**, not just the ones it was
trained on (see `core/indicators.build_ml_features` — every feature is a
ratio or bounded oscillator, never a raw price level, which is what makes
this generalise across companies of any price scale).

## Retraining later

Re-run either option above whenever you want to refresh the model on more
recent data. The new checkpoint simply overwrites these files.
