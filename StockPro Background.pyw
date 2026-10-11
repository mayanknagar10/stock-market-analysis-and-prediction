"""Local no-console development launcher. Double-click; no GitHub or market jobs."""
import ctypes
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import importlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from threading import Lock,Thread
import uuid

ROOT=Path(__file__).resolve().parent
if not (ROOT/"api/main.py").is_file():raise RuntimeError("Keep this launcher in the StockPro repository folder")
STATE=ROOT/"data/web_platform/background"
STATE.mkdir(parents=True,exist_ok=True)
PYTHON=Path(os.environ.get("STOCKPRO_PYTHON",str(Path(os.environ.get("LOCALAPPDATA",""))/"Programs/Python/Python312/python.exe")))
if not PYTHON.is_file():PYTHON=Path(sys.executable).with_name("python.exe")
TOKEN=secrets.token_urlsafe(48)
TASKS={};LOCK=Lock()
FLAGS=0x08000000 if os.name=="nt" else 0
ENV=os.environ.copy()
ENV["PYTHONPATH"]=str(ROOT/".deps")
ENV["PLAYWRIGHT_BROWSERS_PATH"]=str(ROOT/"frontend/.cache/playwright")
ENV["npm_config_cache"]=str(ROOT/".npm-cache")
ENV["NEXT_PUBLIC_API_BASE_URL"]="http://127.0.0.1:8000"
ENV["NEXT_TELEMETRY_DISABLED"]="1"
def now():return datetime.now(timezone.utc).isoformat()
def save(name,value):
    (STATE/name).write_text(json.dumps(value,indent=2),encoding="utf-8")
def console_present():
    return bool(ctypes.windll.kernel32.GetConsoleWindow()) if os.name=="nt" else False
def check_child_console():
    result=subprocess.run([str(PYTHON),"-c","import ctypes,json;print(json.dumps({'console_window':bool(ctypes.windll.kernel32.GetConsoleWindow())}))"],
        cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        creationflags=FLAGS,shell=False,timeout=15)
    child=json.loads(result.stdout.decode())
    if result.returncode or child["console_window"]:raise RuntimeError("Child console suppression failed")
    return child
def worker(identity,name):
    try:
        if name=="runner-self-test":
            spec={"argv":[str(PYTHON),"-m","pytest","tests/test_hidden_runtime.py","-q"],"cwd":ROOT,"timeout":120}
        else:
            if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
            module=importlib.import_module("scripts.hidden_runtime")
            module=importlib.reload(module)
            spec=module.task_spec(name,ROOT,PYTHON)
        with LOCK:TASKS[identity].update(status="RUNNING",started_at=now())
        save(identity+".json",TASKS[identity])
        with (STATE/(identity+".stdout.log")).open("wb") as out,(STATE/(identity+".stderr.log")).open("wb") as error:
            task_env={**ENV,**spec.get("env",{})}
            process=subprocess.Popen(spec["argv"],cwd=spec["cwd"],env=task_env,stdin=subprocess.DEVNULL,
                stdout=out,stderr=error,shell=False,creationflags=FLAGS)
            with LOCK:TASKS[identity]["pid"]=process.pid
            save(identity+".json",TASKS[identity])
            if spec.get("service"):
                with LOCK:TASKS[identity].update(status="RUNNING_SERVICE",finished_at=None)
                save(identity+".json",TASKS[identity]);return
            try:code=process.wait(timeout=spec["timeout"])
            except subprocess.TimeoutExpired:
                process.kill();process.wait();raise RuntimeError("Task timeout")
        with LOCK:TASKS[identity].update(status="PASSED" if code==0 else "FAILED",exit_code=code,finished_at=now())
    except Exception as error:
        with LOCK:TASKS[identity].update(status="FAILED",error_type=type(error).__name__,finished_at=now())
    save(identity+".json",TASKS[identity])
def enqueue(name):
    if name!="runner-self-test":
        if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
        module=importlib.import_module("scripts.hidden_runtime")
        importlib.reload(module).validate_task(name)
    identity=uuid.uuid4().hex
    with LOCK:
        if any(t["status"] in {"QUEUED","RUNNING"} for t in TASKS.values()):raise RuntimeError("Task busy")
        TASKS[identity]={"run_id":identity,"task":name,"status":"QUEUED","queued_at":now(),"console_window":False}
    save(identity+".json",TASKS[identity])
    Thread(target=worker,args=(identity,name),daemon=True).start()
    return TASKS[identity]
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def respond(self,status,body):
        data=json.dumps(body).encode()
        self.send_response(status);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
    def allowed(self):
        host=self.headers.get("Host","")
        if host not in {"127.0.0.1:8765","localhost:8765"}:return False
        if any(k in self.headers for k in ("Origin","Forwarded","X-Forwarded-For","X-Real-IP")):return False
        value=self.headers.get("Authorization","")
        return secrets.compare_digest(value,"Bearer "+TOKEN)
    def do_GET(self):
        if not self.allowed():return self.respond(403,{"error":"FORBIDDEN"})
        if self.path=="/health":return self.respond(200,{"status":"READY","console_window":False})
        if self.path=="/tasks":
            with LOCK:return self.respond(200,{"items":list(TASKS.values())})
        if self.path.startswith("/tasks/"):
            with LOCK:task=TASKS.get(self.path.split("/")[-1])
            return self.respond(200 if task else 404,task or {"error":"NOT_FOUND"})
        return self.respond(404,{"error":"NOT_FOUND"})
    def do_POST(self):
        if not self.allowed():return self.respond(403,{"error":"FORBIDDEN"})
        if self.path!="/tasks":return self.respond(404,{"error":"NOT_FOUND"})
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 0<length<=2048:raise ValueError()
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict) or set(body)!={"task"} or not isinstance(body["task"],str):raise ValueError()
            task=enqueue(body["task"])
            return self.respond(202,task)
        except (ValueError,ImportError):return self.respond(422,{"error":"INVALID_TASK"})
        except RuntimeError:return self.respond(409,{"error":"BUSY"})
def main():
    if console_present():raise RuntimeError("Use the .pyw no-console launcher")
    proof=check_child_console()
    server=ThreadingHTTPServer(("127.0.0.1",8765),Handler)
    save("session.json",{"pid":os.getpid(),"started_at":now(),"address":"http://127.0.0.1:8765","token":TOKEN,
        "parent_console_window":False,"child_console_window":proof["console_window"]})
    enqueue("runner-self-test")
    server.serve_forever()
try:main()
except Exception as error:save("startup-error.json",{"timestamp":now(),"error_type":type(error).__name__})
