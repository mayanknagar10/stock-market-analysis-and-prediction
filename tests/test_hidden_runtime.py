"""No-console policy and local-only development task contracts; no real subprocesses."""
import subprocess
from pathlib import Path
import pytest
from scripts.hidden_runtime import launch_options, task_spec, validate_host, validate_task, get_console_window

def test_windows_children_never_allocate_console():
    options=launch_options("nt")
    assert options["shell"] is False
    assert options["creationflags"] & 0x08000000
    assert options["stdin"] == subprocess.DEVNULL

def test_other_platforms_remain_shell_free():
    options=launch_options("posix")
    assert options["shell"] is False and options["creationflags"]==0

@pytest.mark.parametrize("name",["shell","github","push","train","capture","resolve","../../secrets"])
def test_arbitrary_or_research_tasks_are_rejected(name):
    with pytest.raises(ValueError):validate_task(name)

def test_fixed_task_commands_have_no_user_args():
    root=Path(__file__).resolve().parents[1]
    spec=task_spec("backend-tests",root,Path("C:/Python/python.exe"))
    assert spec["argv"][1:]==["-m","pytest","-q"]
    assert spec["cwd"]==root
    assert spec["timeout"]<=1800

def test_only_loopback_host_is_accepted():
    assert validate_host("127.0.0.1:8765")
    assert validate_host("localhost:8765")
    assert not validate_host("attacker.example:8765")
    assert not validate_host("0.0.0.0:8765")


@pytest.mark.skipif(__import__("os").name!="nt",reason="Windows console policy")
@pytest.mark.parametrize("worker",["operator","provider"])
def test_nested_application_workers_are_console_free(monkeypatch,worker):
    captured={}
    def fake(argv,**kwargs):
        captured.update(kwargs)
        kwargs["stdout"].write(b"{}")
        return subprocess.CompletedProcess(argv,0)
    monkeypatch.setattr(subprocess,"run",fake)
    if worker=="operator":
        from application.services.operator_jobs import execute
        execute(["python.exe","-m","allowed.module"],10)
    else:
        from application.services.web_analytics import metadata_provider
        metadata_provider("fundamentals","TCS.NS",20)
    assert captured.get("creationflags",0) & 0x08000000
    assert captured["shell"] is False

@pytest.mark.parametrize("task",["frontend-types","frontend-test","frontend-lint","frontend-build","frontend-typecheck","frontend-e2e","frontend-dev"])
def test_frontend_checks_launch_direct_clis_without_npm_script_shell(task,tmp_path):
    from scripts.hidden_runtime import task_spec
    argv=task_spec(task,tmp_path,"python.exe")["argv"]
    assert argv[0].endswith("node.exe")
    assert "npm-cli.js" not in argv[1]
