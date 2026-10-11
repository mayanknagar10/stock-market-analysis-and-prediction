# Prospective scheduling
Standalone jobs run without Streamlit or HTTP requests. First canonical origin session: 2026-10-09. Cutoff is 06:30 Asia/Kolkata on calendar D+1 (01:00 UTC); acquisition starts 06:00 IST (00:30 UTC), finalization runs 06:30–07:00 IST. Run every calendar day: the registered NSE calendar decides eligibility, including special sessions.

## Commands
Run from the repository root in Python 3.12:
    python -m jobs.capture_prospective_forecasts --stage capture
    python -m jobs.capture_prospective_forecasts --stage finalize
    python -m jobs.resolve_matured_outcomes
    python -m jobs.scheduled_tick

Explicit --session YYYY-MM-DD never bypasses time gates. Late acquisition records an origin unavailable. Set STOCKPRO_DATA_ROOT to durable private storage (default data/prospective_automation). Retain collection.sqlite, archive/, writer.lock, resolver markers and logs. Deploy frozen models/universe/calendar with the application. STOCKPRO_PERFORMANCE_LOG optionally enables private JSONL latency logging. Future provider credentials belong in environment/secrets.

Local workspace dependencies use PYTHONPATH=<repo>/.deps. A normal virtual environment installs requirements.txt.

## Cron / external scheduler
Use UTC scheduler time and a persistent volume:
    30 0 * * * cd /srv/stockpro && /srv/venv/bin/python -m jobs.capture_prospective_forecasts --stage capture
    0 1 * * * cd /srv/stockpro && /srv/venv/bin/python -m jobs.capture_prospective_forecasts --stage finalize
    0 15 * * * cd /srv/stockpro && /srv/venv/bin/python -m jobs.resolve_matured_outcomes

Alternatively invoke scheduled_tick every five minutes. Its UTC/IST guards are independent of host timezone/daylight saving. It records missed eligible origins after downtime as unavailable, never backfilled. Its independent resolver attempts once daily at/after20:30IST; explicit resolver reruns can retry pending source failures.

Use one durable writer. Alert on process failures and FAILED/DEGRADED batches. Exit0 does not mean HEALTHY. Keep clocks synchronized. OS locks release after crashes. Partial generation resumes only inside the original finalization window using already archived sources. Never recapture after cutoff.

## Windows
Register the included hidden five-minute dispatcher for the signed-in user:
    powershell -NoProfile -File scripts/install_local_schedule.ps1 -Python C:\path\to\python.exe

It refuses to replace an existing named task. Inspect:
    Get-ScheduledTask -TaskName StockPro-Prospective-Research-v1
    Get-ScheduledTaskInfo -TaskName StockPro-Prospective-Research-v1

The runner sets working directory and optional .deps. The PC must be awake and the user signed in; registration does not establish always-on operation. No password is stored. Unattended service-principal hosting is a separate decision. To explicitly stop:
    Disable-ScheduledTask -TaskName StockPro-Prospective-Research-v1

## Cloud collection — deferred by user choice

The approved web-platform mode uses the existing local Python backend and durable local files. No GitHub access, workflow draft, upload, secret configuration, push or activation is authorized. No cloud scheduler or paid service has been enabled. A hosted runner's temporary disk would lose the snapshots and ledger between runs; clicking a manual cloud trigger does not remove that persistence requirement. Cloud independence remains deferred until a separately approved durable-storage/scheduler plan exists. No service is promised to remain free forever.

## Protected manual frontend operations

The new Admin / Research Status page submits authenticated operations to the same local Python backend. It invokes the existing capture/finalize/resolver/verification commands asynchronously; financial or collection logic is not duplicated in React. The backend and PC must remain available. Opening an ordinary page never generates forecasts. Capture/finalize require an explicit origin and retain the canonical windows above; a late click cannot backdate data. Non-session origins return an auditable skip.

Local operator provisioning requires STOCKPRO_ENABLE_LOCAL_OPERATOR=1, STOCKPRO_ENV=development and an exact configured STOCKPRO_CORS_ORIGINS entry. Bind the API to loopback and use one worker for memory-only operator sessions. Outside this local mode a separate STOCKPRO_OPERATOR_TOKEN is required; never place it in NEXT_PUBLIC variables or browser storage. Operator records are private, append-only files under data/web_platform/jobs.

## Window-free development launcher

StockPro Background.pyw is a user-started local development helper. It verifies that its parent/probe have no Windows console and uses CREATE_NO_WINDOW with shell=False for a fixed allowlist of checks and development services. It does not run collection, outcome, training or GitHub tasks. The nested application provider/operator workers use the same Windows no-console flag. Private gateway state/logs stay under ignored data/web_platform/background and are not web API data. This helper does not replace the existing production research CLI or local scheduler.

## Recovery
Read /api/v1/collection/status for immutable batches, member failures and freshness. Exceptions remain in private logs; public responses use sanitized codes. Verify hashes before reuse. Back up the complete data directory while idle, or use SQLite online backup plus preserved snapshot files. Unknown calendar years/unconfirmed special hours fail closed until separately versioned official registrations. Scheduling cannot repair provider outages or insufficient evidence.

The standalone resolver also matures the388preserved manual forecasts in their separate legacy ledger. New raw outcome snapshots live in the automation archive; original forecast payloads do not change. The current local task permits battery operation but still requires an awake machine and user login.
