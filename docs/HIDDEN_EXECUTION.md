# Hidden local execution
The user prohibited terminal windows opening/closing. The command tool does not reliably honor a no-console shell override, so shell launches are paused.

Step1: double-click StockPro Background.pyw in this repository folder. Do not move it to Downloads. Windows .pyw uses the no-console Python launcher. The file aborts if attached to a console, validates a hidden child process, and writes proof to ignored data/web_platform/background/session.json. No window is displayed.

The launcher binds127.0.0.1:8765 only. Requests require a random local bearer credential stored in the private session file; Host is loopback allowlisted and all browser Origin/forwarding headers are rejected. No arbitrary command, argument, working-directory or file path is accepted. No GitHub, paid service, model training or real forecast/outcome collection task is exposed.

It initially runs the new launcher-policy test. That test is expected to fail while scripts/hidden_runtime.py is not yet implemented, providing the red test before implementation. Further fixed local development tasks become available only through the reviewed module. stdout/stderr are private logs, not browser/public responses.

The launcher is a development aid, not a production auth/deployment mechanism or substitute for prospective scheduling. Frontend and backend remain local until separately deployed with permission. No claim of no-console execution or final test success is made until the recorded proof and test results are checked.
