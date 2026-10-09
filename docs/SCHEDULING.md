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

## GitHub Actions
A self-hosted runner with persistent private storage can run the same commands. Use concurrency controls, secrets and pinned dependencies. Ephemeral hosted-runner filesystems cannot be the durable ledger. Never upload private raw snapshots as public artifacts.

Scheduled Actions may be delayed or dropped; late execution must still fail the canonical window. It is an optional trigger, not the preferred narrow-window clock. [GitHub schedule limitations](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule). No workflow is activated here.

## Recovery
Read /api/v1/collection/status for immutable batches, member failures and freshness. Exceptions remain in private logs; public responses use sanitized codes. Verify hashes before reuse. Back up the complete data directory while idle, or use SQLite online backup plus preserved snapshot files. Unknown calendar years/unconfirmed special hours fail closed until separately versioned official registrations. Scheduling cannot repair provider outages or insufficient evidence.

The standalone resolver also matures the388preserved manual forecasts in their separate legacy ledger. New raw outcome snapshots live in the automation archive; original forecast payloads do not change. The current local task permits battery operation but still requires an awake machine and user login.
