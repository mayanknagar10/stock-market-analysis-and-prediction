# StockPro self-use integration

Status: temporary development connection. The Vercel preview serves Next.js; a Cloudflare Quick Tunnel forwards HTTPS requests to one FastAPI process bound to `127.0.0.1:8000` on the developer PC. The PC and tunnel must stay running while the remote frontend is used. There is no permanent backend, cloud collector, or production model.

## Start and reconnect

From the repository root, use Python 3.12 with the local dependencies installed. Start FastAPI with the following environment in a persistent local session (or launch it hidden with equivalent environment values):

```powershell
$env:PYTHONPATH = (Resolve-Path .deps).Path
$env:STOCKPRO_ENV = 'development'
$env:STOCKPRO_HOSTED_READ_ONLY = '1'
$env:STOCKPRO_TUNNEL_LOCAL_EVIDENCE = '1'
$env:STOCKPRO_ENABLE_LOCAL_OPERATOR = '0'
$env:STOCKPRO_CORS_ORIGINS = 'https://<current-vercel-preview-host>'
Remove-Item Env:STOCKPRO_OPERATOR_TOKEN,Env:STOCKPRO_API_TOKEN -ErrorAction SilentlyContinue
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Verify `http://127.0.0.1:8000/api/v1/ready` reports `READY`, `research-20261008T190859`, and `BLOCKED`. Obtain the current signed Windows `cloudflared` binary from [Cloudflare's download page](https://developers.cloudflare.com/tunnel/downloads/) and start a free Quick Tunnel in another persistent local session:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate
```

The generated `https://…trycloudflare.com` URL is temporary. Open the Vercel preview's **Research status → Preview backend connection**, paste that URL, select **Save backend URL**, then **Reload to connect**. The override stays only in that browser's local storage; it contains no credential. Other browsers can use their own override. **Use deployment default** removes the override. `NEXT_PUBLIC_API_BASE_URL` remains the Vercel Preview environment default and is a public URL, never a token. A tunnel URL change therefore needs no source edit or Vercel redeployment. The FastAPI CORS origin must still match the Vercel page origin exactly; its value is independent of the changing tunnel URL.

Quick Tunnels are for development, have no uptime guarantee, and anyone with the URL can reach the allowed public research API. Do not open port 8000 on the home router. Stop both local processes when remote use ends. A stopped tunnel produces one page-level disconnected notice and quieter section-level unavailable states; no substitute market data are generated.

## Safety boundary

`STOCKPRO_HOSTED_READ_ONLY=1` blocks `/api/v1/operations/*` and internal performance routes, disables operator jobs, caps analysis request bodies, rate-limits provider-heavy routes, and queues at most two provider-backed requests at a time. `STOCKPRO_TUNNEL_LOCAL_EVIDENCE=1` applies only when `STOCKPRO_ENV=development`; it verifies the authoritative local ledgers and frozen model bundles without requiring a separately copied evidence mirror. The local operator console can still be run separately without these tunnel settings, bound only to loopback. Never send an operator token through Vercel or a `NEXT_PUBLIC_*` variable. The tunneled browser exposes only read-only research and bounded stateless analysis; portfolio/watchlist inputs stay in browser local storage unless explicitly submitted for a calculation.

The backend returns completed-session data, not synchronized live quotes. Provider-backed fundamentals, factors, news and market context can be slow or partially unavailable. Forecasts remain RESEARCH ONLY, LOW confidence, and abstained. The V5 final gate remains FAILED; canonical prospective evidence is distinct from the preserved manual stream. No model, feature, calibration, threshold, ledger, or research result is changed by this connection.

## Current review and limits

The self-use browser checks cover Vercel → Quick Tunnel → local FastAPI for Dashboard, Markets, TCS/INFY stock sections and ticker switching, legacy Forecast, Validate, Screener, Compare, Portfolio, Watchlist, Risk, Factors, News, saved Scenarios, and the blocked internal backtesting route. Backend news can legitimately be unavailable, fundamentals can be partial, and strategy backtest execution is still only in the local Streamlit console. The Vercel frontend is a preview; its URL and tunnel URL are not permanent hosting. The developer PC must remain online.
