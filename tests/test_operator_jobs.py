"""Operator boundary and append-only queue: no real subprocess or research writes."""
import json
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
import pytest

def app_client(tmp_path, monkeypatch, **env):
    from api.main import create_app
    from application.services.operator_jobs import OperatorJobs
    for key in ('STOCKPRO_OPERATOR_TOKEN', 'STOCKPRO_ENABLE_LOCAL_OPERATOR', 'STOCKPRO_API_TOKEN', 'STOCKPRO_CORS_ORIGINS'):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items(): monkeypatch.setenv(key, value)
    tasks=[]
    jobs=OperatorJobs(tmp_path, executor=lambda argv, timeout: {'status':'SKIPPED','reason':'NON_SESSION'}, dispatch=tasks.append)
    client=TestClient(create_app(operator_jobs=jobs), client=('127.0.0.1', 4321))
    return client, jobs, tasks

def test_operations_fail_closed_and_read_token_is_not_operator(tmp_path,monkeypatch):
    client, jobs, tasks=app_client(tmp_path,monkeypatch)
    assert client.post('/api/v1/operations/jobs',json={'job_type':'verify'}).json()['error']['code']=='OPERATOR_AUTH_UNAVAILABLE'
    monkeypatch.setenv('STOCKPRO_API_TOKEN','public-read')
    monkeypatch.setenv('STOCKPRO_OPERATOR_TOKEN','private-op')
    result=client.post('/api/v1/operations/jobs',headers={'Authorization':'Bearer public-read'},json={'job_type':'verify'})
    assert result.status_code==401 and not tasks
    assert client.get('/api/v1/system/status',headers={'Authorization':'Bearer public-read'}).status_code==200
    assert client.get('/api/v1/system/status',headers={'Authorization':'Bearer private-op'}).status_code==401

def test_local_cookie_exact_origin_loopback_expiry_and_csrf(tmp_path,monkeypatch):
    client,jobs,tasks=app_client(tmp_path,monkeypatch,STOCKPRO_ENABLE_LOCAL_OPERATOR='1',STOCKPRO_CORS_ORIGINS='http://localhost:3000')
    endpoint='/api/v1/operations/session'
    assert client.post(endpoint).status_code==403
    assert client.post(endpoint,headers={'Origin':'http://localhost:3000.evil'}).status_code==403
    result=client.post(endpoint,headers={'Origin':'http://localhost:3000'})
    assert result.status_code==200
    cookie=result.headers['set-cookie'].lower()
    assert 'httponly' in cookie and 'samesite=strict' in cookie and 'max-age=' in cookie
    assert 'token' not in result.json() and 'secret' not in result.text
    assert client.post('/api/v1/operations/jobs',json={'job_type':'verify'}).status_code==403
    assert client.post('/api/v1/operations/jobs',headers={'Origin':'http://localhost:3000'},json={'job_type':'verify'}).status_code==202
    client.app.state.operator_auth.clock=lambda: datetime.now(timezone.utc)+timedelta(days=2)
    assert client.get('/api/v1/operations/status',headers={'Origin':'http://localhost:3000'}).status_code==401
    remote=TestClient(client.app,client=('203.0.113.20',4000))
    assert remote.post(endpoint,headers={'Origin':'http://localhost:3000'}).status_code==403

def test_queue_allowlist_dedup_execution_and_immutable_terminal(tmp_path,monkeypatch):
    client,jobs,tasks=app_client(tmp_path,monkeypatch,STOCKPRO_OPERATOR_TOKEN='private-op')
    headers={'Authorization':'Bearer private-op'}
    assert client.post('/api/v1/operations/jobs',headers=headers,json={'job_type':'train'}).status_code==422
    assert client.post('/api/v1/operations/jobs',headers=headers,json={'job_type':'capture','origin_session':'2026-10-10;evil'}).status_code==422
    result=client.post('/api/v1/operations/jobs',headers=headers,json={'job_type':'capture','origin_session':'2026-10-10'})
    assert result.status_code==202 and len(tasks)==1
    record=result.json(); assert record['status']=='QUEUED' and record['universe_size']==100
    duplicate=client.post('/api/v1/operations/jobs',headers=headers,json={'job_type':'capture','origin_session':'2026-10-10'})
    assert duplicate.json()['run_id']==record['run_id'] and len(tasks)==1
    assert client.post('/api/v1/operations/jobs',headers=headers,json={'job_type':'verify'}).status_code==409
    tasks.pop()()
    final=client.get('/api/v1/operations/jobs/'+record['run_id'],headers=headers).json()
    assert final['status']=='SKIPPED_NON_SESSION' and final['actual_started_at'] and final['finished_at']
    before=list(tmp_path.glob('*.jsonl'))[0].read_bytes()
    jobs.run(record['run_id'])
    assert list(tmp_path.glob('*.jsonl'))[0].read_bytes()==before
    assert 'C:' not in json.dumps(final)
    assert jobs.command('capture','2026-10-10')[1:]==['-m','jobs.capture_prospective_forecasts','--stage','capture','--session','2026-10-10']
    assert jobs.command('verify',None)[1:]==['-m','scripts.verify_automation_preservation']
    assert jobs.command('resolve',None)[1:]==['-m','jobs.resolve_matured_outcomes']

def test_job_executor_failure_is_structured_and_recovers_without_reexecution(tmp_path):
    from application.services.operator_jobs import OperatorJobs
    tasks=[]
    def fail(argv,timeout): raise TimeoutError('C:/private/token')
    jobs=OperatorJobs(tmp_path,executor=fail,dispatch=tasks.append)
    record=jobs.enqueue('verify')
    tasks.pop()()
    final=jobs.get(record['run_id'])
    assert final['status']=='FAILED' and final['error_code']=='JOB_TIMEOUT'
    restarted=OperatorJobs(tmp_path,executor=fail,dispatch=tasks.append)
    assert restarted.get(record['run_id'])==final and not tasks

def test_real_cli_non_session_shape_and_failed_guard_status(tmp_path):
    from application.services.operator_jobs import OperatorJobs
    tasks=[]
    jobs=OperatorJobs(tmp_path,executor=lambda *args:{'status':'SKIPPED','health':'SKIPPED','members':{'TCS.NS':{'code':'NON_SESSION'}},'successful_forecasts':0,'abstained_forecasts':0,'failed_forecasts':0},dispatch=tasks.append)
    first=jobs.enqueue('capture','2026-10-10');tasks.pop()()
    assert jobs.get(first['run_id'])['status']=='SKIPPED_NON_SESSION'
    jobs.executor=lambda *args:{'status':'COMPLETED','health':'FAILED','members':{'TCS.NS':{'code':'MISSED_WINDOW'}}}
    second=jobs.enqueue('finalize','2026-10-09');tasks.pop()()
    assert jobs.get(second['run_id'])['status']=='FAILED'

def test_local_sessions_reject_forwarding_and_production(tmp_path,monkeypatch):
    client,_,_=app_client(tmp_path,monkeypatch,STOCKPRO_ENABLE_LOCAL_OPERATOR='1',STOCKPRO_CORS_ORIGINS='http://localhost:3000')
    headers={'Origin':'http://localhost:3000','X-Forwarded-For':'203.0.113.2'}
    assert client.post('/api/v1/operations/session',headers=headers).status_code==403
    monkeypatch.setenv('STOCKPRO_ENV','production')
    assert client.post('/api/v1/operations/session',headers={'Origin':'http://localhost:3000'}).status_code==403

def test_operator_token_must_differ_from_shared_read_token(tmp_path,monkeypatch):
    client,_,tasks=app_client(tmp_path,monkeypatch,STOCKPRO_OPERATOR_TOKEN='same',STOCKPRO_API_TOKEN='same')
    response=client.post('/api/v1/operations/jobs',headers={'Authorization':'Bearer same'},json={'job_type':'verify'})
    assert response.status_code==503 and response.json()['error']['code']=='OPERATOR_AUTH_UNAVAILABLE' and not tasks

def test_queued_job_serializes_across_service_instances(tmp_path):
    from application.services.operator_jobs import OperatorJobs
    tasks=[]
    first=OperatorJobs(tmp_path,dispatch=tasks.append)
    row=first.enqueue('verify')
    second=OperatorJobs(tmp_path,dispatch=tasks.append)
    assert second.get(row['run_id'])['status']=='QUEUED'
    with pytest.raises(Exception) as error:second.enqueue('resolve')
    assert error.value.code=='OPERATOR_JOB_BUSY'

def test_capture_partial_source_counts_and_resolve_provider_failure(tmp_path):
    from application.services.operator_jobs import OperatorJobs
    tasks=[]
    jobs=OperatorJobs(tmp_path,executor=lambda *args:{'status':'INPUTS_FROZEN','captured_sources':115,'failed_sources':4},dispatch=tasks.append,capture_inventory=lambda payload:{'captured_sources':115,'failed_sources':4,'required_stock_sources':96})
    first=jobs.enqueue('capture','2026-10-09');tasks.pop()()
    result=jobs.get(first['run_id'])
    assert result['status']=='SUCCESS_WITH_DEGRADATION' and result['result_status']=='INPUTS_FROZEN'
    assert result['captured_source_count']==115 and result['failed_source_count']==4
    jobs.executor=lambda *args:{'status':'RESOLVED','matured':0,'pending':400,'source_failures':{'TCS.NS':{'code':'PROVIDER_FAILURE'}}}
    second=jobs.enqueue('resolve');tasks.pop()()
    result=jobs.get(second['run_id'])
    assert result['status']=='SUCCESS_WITH_DEGRADATION' and result['provider_failure_count']==1

@pytest.mark.parametrize('kind,payload',[
    ('verify',{}),('capture',{'status':'ALIEN','captured_sources':100,'failed_sources':0}),
    ('resolve',{'status':'RESOLVED'}),('finalize',{'status':'COMPLETED','health':'HEALTHY'}),
    ('capture',{'status':'INPUTS_FROZEN','captured_sources':-1,'failed_sources':0}),
    ('verify',{'original_forecasts_verified':388}),
])
def test_unknown_or_malformed_job_payload_is_never_success(tmp_path,kind,payload):
    from application.services.operator_jobs import OperatorJobs
    tasks=[]
    jobs=OperatorJobs(tmp_path,executor=lambda *args:payload,dispatch=tasks.append)
    row=jobs.enqueue(kind,'2026-10-09' if kind in {'capture','finalize'} else None);tasks.pop()()
    result=jobs.get(row['run_id'])
    assert result['status']=='FAILED' and result['error_code']=='JOB_OUTPUT_INVALID'

def test_no_captured_inputs_is_failed_not_success_with_degradation(tmp_path):
    from application.services.operator_jobs import OperatorJobs
    tasks=[]
    jobs=OperatorJobs(tmp_path,executor=lambda *args:{'status':'INPUTS_FROZEN','captured_sources':0,'failed_sources':119},dispatch=tasks.append)
    row=jobs.enqueue('capture','2026-10-09');tasks.pop()()
    result=jobs.get(row['run_id'])
    assert result['status']=='FAILED' and result['error_code']=='JOB_REQUIRED_INPUTS_UNAVAILABLE'

def test_nonzero_cli_exit_preserves_structured_failure_counts(tmp_path,monkeypatch):
    from application.services.operator_jobs import OperatorJobs,execute
    import subprocess
    payload={'status':'COMPLETED','health':'FAILED','successful_forecasts':0,'failed_forecasts':400,'abstained_forecasts':0,'provider_failures':119,'data_quality_failures':5}
    calls=[]
    def subprocess_fixture(argv,**kwargs):
        calls.append((argv,kwargs['shell'],kwargs['timeout']))
        kwargs['stdout'].write(json.dumps(payload).encode('utf-8'))
        kwargs['stderr'].write(b'C:/private/secret traceback')
        return subprocess.CompletedProcess(argv,2)
    monkeypatch.setattr(subprocess,'run',subprocess_fixture)
    tasks=[];jobs=OperatorJobs(tmp_path,executor=execute,dispatch=tasks.append)
    row=jobs.enqueue('finalize','2026-10-09');tasks.pop()()
    result=jobs.get(row['run_id'])
    assert result['status']=='FAILED' and result['failure_count']==400 and result['provider_failure_count']==119
    assert result['data_quality_failure_count']==5 and result['error_code']=='JOB_PROCESS_FAILED'
    assert calls[0][1:] == (False,7200)
    assert 'private' not in json.dumps(result) and 'traceback' not in json.dumps(result)

@pytest.mark.parametrize('inventory,expected',[
    ({'captured_sources':0,'failed_sources':119,'required_stock_sources':0},'FAILED'),
    ({'captured_sources':19,'failed_sources':100,'required_stock_sources':0},'FAILED'),
    ({'captured_sources':119,'failed_sources':0,'required_stock_sources':100},'SUCCESS'),
])
def test_repeat_capture_uses_read_only_frozen_inventory(tmp_path,inventory,expected):
    from application.services.operator_jobs import OperatorJobs
    tasks=[];calls=[]
    def reader(payload):calls.append(payload);return inventory
    jobs=OperatorJobs(tmp_path,executor=lambda *args:{'status':'INPUTS_ALREADY_FROZEN'},dispatch=tasks.append,capture_inventory=reader)
    row=jobs.enqueue('capture','2026-10-09');tasks.pop()()
    result=jobs.get(row['run_id'])
    assert result['status']==expected and calls
    assert result['captured_source_count']==inventory['captured_sources']

def test_frozen_inventory_reader_is_hash_checked_read_only(tmp_path):
    import hashlib
    from application.collection.store import CollectionStore
    from application.services.operator_jobs import frozen_capture_inventory
    store=CollectionStore(tmp_path);batch=store.begin('2026-10-09')
    store.freeze_sources(batch,{}, {'TCS.NS':{'code':'PROVIDER_FAILURE'}},'2026-10-10T00:59Z')
    before=hashlib.sha256(store.path.read_bytes()).hexdigest()
    inventory=frozen_capture_inventory(batch,directory=tmp_path)
    assert inventory=={'captured_sources':0,'failed_sources':1,'required_stock_sources':0}
    assert hashlib.sha256(store.path.read_bytes()).hexdigest()==before

def test_success_shapes_for_verify_and_no_due_resolve(tmp_path):
    from application.services.operator_jobs import OperatorJobs
    tasks=[];payloads=[{'v5_files_verified':1052,'additional_files_verified':2904,'original_forecasts_verified':388,'production_promotion':'BLOCKED'},{'status':'NO_MATURED_FORECASTS','matured':0,'pending':388}]
    jobs=OperatorJobs(tmp_path,executor=lambda *args:payloads.pop(0),dispatch=tasks.append)
    for kind in ('verify','resolve'):
        row=jobs.enqueue(kind);tasks.pop()()
        assert jobs.get(row['run_id'])['status']=='SUCCESS'
