"""Fixed local development tasks; no arbitrary commands, remote actions or research jobs."""
import ctypes
import os
from pathlib import Path
import subprocess

TASKS=frozenset({
    "backend-tests","backend-web-tests","backend-preservation","backend-compile","export-api",
    "frontend-install","frontend-types","frontend-test","frontend-lint","frontend-build",
    "frontend-e2e","frontend-typecheck","frontend-dev","api-dev","git-status","web-qa-fixtures","git-diff"
})
def validate_task(name):
    if type(name) is not str or name not in TASKS:raise ValueError("Unsupported local development task")
    return name
def validate_host(value):
    return value in {"127.0.0.1:8765","localhost:8765"}
def get_console_window():
    return int(ctypes.windll.kernel32.GetConsoleWindow()) if os.name=="nt" else 0
def launch_options(platform=None):
    platform=platform or os.name
    return {"shell":False,"creationflags":0x08000000 if platform=="nt" else 0,"stdin":subprocess.DEVNULL}
def task_spec(name,root,python):
    validate_task(name);root=Path(root).resolve();python=Path(python)
    front=root/"frontend"
    npm=Path(os.environ.get("ProgramFiles","C:/Program Files"))/"nodejs/node_modules/npm/bin/npm-cli.js"
    node=Path(os.environ.get("ProgramFiles","C:/Program Files"))/"nodejs/node.exe"
    npm_args=[str(node),str(npm)]
    specs={
        "backend-tests":([str(python),"-m","pytest","-q"],root,1800),
        "backend-web-tests":([str(python),"-m","pytest","tests/test_web_analytics.py","tests/test_operator_jobs.py","tests/test_web_evidence.py","tests/test_web_api.py","tests/test_hidden_runtime.py","-q"],root,600),
        "backend-preservation":([str(python),"scripts/verify_automation_preservation.py"],root,180),
        "backend-compile":([str(python),"-m","compileall","-q","api","application","jobs","scripts/hidden_runtime.py"],root,180),
        "export-api":([str(python),"scripts/export_openapi.py"],root,180),
        "web-qa-fixtures":([str(python),"scripts/export_web_qa_fixtures.py"],root,180),
        "frontend-install":(npm_args+["install","--registry","https://registry.npmjs.org","--no-audit","--no-fund"],front,900),
        "frontend-types":([str(node),str(front/"node_modules/openapi-typescript/bin/cli.js"),"../docs/api/openapi.json","-o","src/lib/generated/api.d.ts"],front,300),
        "frontend-test":([str(node),str(front/"node_modules/vitest/vitest.mjs"),"run"],front,900),
        "frontend-lint":([str(node),str(front/"node_modules/eslint/bin/eslint.js"),"."],front,600),
        "frontend-build":([str(node),str(front/"node_modules/next/dist/bin/next"),"build","--webpack"],front,1800),
        "frontend-e2e":([str(node),str(front/"node_modules/@playwright/test/cli.js"),"test"],front,1800),
        "frontend-typecheck":([str(node),str(front/"node_modules/typescript/bin/tsc"),"--noEmit"],front,600),
        "frontend-dev":([str(node),str(front/"node_modules/next/dist/bin/next"),"dev","--hostname","127.0.0.1","--port","3000"],front,1800),
        "api-dev":([str(python),"-m","uvicorn","api.main:app","--host","127.0.0.1","--port","8001","--workers","1"],root,1800),
        "git-diff":(["C:/Program Files/Git/cmd/git.exe","-c","core.longpaths=true","-c","safe.directory="+root.as_posix(),"diff","--no-ext-diff","--no-textconv","-U10","44121a6","--",".gitignore","api","application","docs","scripts","tests"],root,120),
        "git-status":(["C:/Program Files/Git/cmd/git.exe","-c","core.longpaths=true","-c","safe.directory="+root.as_posix(),"status","--short"],root,120)
    }
    argv,cwd,timeout=specs[name]
    result={"argv":argv,"cwd":cwd,"timeout":timeout}
    if name in {"frontend-dev","api-dev"}:result["service"]=True
    if name=="api-dev":
        result["env"]={"STOCKPRO_ENABLE_LOCAL_OPERATOR":"1","STOCKPRO_CORS_ORIGINS":"http://127.0.0.1:3000,http://localhost:3000","STOCKPRO_ENV":"development"}
    if name=="frontend-build":
        result["env"]={"NEXT_PUBLIC_API_BASE_URL":"https://stockpro-backend.invalid"}
    if name=="frontend-dev":
        result["env"]={"NEXT_PUBLIC_API_BASE_URL":"http://127.0.0.1:8001"}
    return result
