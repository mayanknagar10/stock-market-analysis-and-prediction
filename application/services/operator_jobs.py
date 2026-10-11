"""Explicit operator-only local jobs; no CLI research logic is reimplemented here."""
import copy
import hashlib
from datetime import date,datetime,timezone,timedelta
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sqlite3
import sys
import tempfile
from threading import RLock,Thread
import uuid
from application.config import ROOT,PROTOCOL,data_root,universe
from application.services.prospective import ServiceError
from application.web_schemas import JobRecord

def utcnow():return datetime.now(timezone.utc)

class OperatorAuth:
    cookie='stockpro_operator_session'
    lifetime=timedelta(hours=1)
    def __init__(self,clock=utcnow):self.clock=clock;self.sessions={};self.lock=RLock()
    def origins(self):return {x.strip() for x in os.environ.get('STOCKPRO_CORS_ORIGINS','').split(',') if x.strip() and x.strip()!='*'}
    def loopback(self,request):
        try:return ipaddress.ip_address(request.client.host).is_loopback
        except (ValueError,AttributeError):return False
    def local_enabled(self):
        return os.environ.get('STOCKPRO_ENABLE_LOCAL_OPERATOR')=='1' and os.environ.get('STOCKPRO_ENV','development').lower() not in {'production','prod'}
    def local_guard(self,request):
        if not self.local_enabled() or not self.loopback(request) or request.headers.get('origin') not in self.origins() or any(k in request.headers for k in ('forwarded','x-forwarded-for','x-real-ip')):
            raise ServiceError('OPERATOR_ORIGIN_FORBIDDEN',403)
    def provision(self,request,response):
        self.local_guard(request)
        now=self.clock();expiry=now+self.lifetime;token=secrets.token_urlsafe(48)
        with self.lock:
            self.sessions={key:value for key,value in self.sessions.items() if value>now}
            if len(self.sessions)>=128:raise ServiceError('OPERATOR_SESSION_LIMIT',429)
            self.sessions[token]=expiry
        response.set_cookie(self.cookie,token,max_age=int(self.lifetime.total_seconds()),httponly=True,samesite='strict',secure=request.url.scheme=='https',path='/api/v1')
        return {'authenticated':True,'mode':'LOCAL_LOOPBACK','expires_at':expiry}
    def require(self,request):
        configured=os.environ.get('STOCKPRO_OPERATOR_TOKEN')
        if configured and configured==os.environ.get('STOCKPRO_API_TOKEN'):raise ServiceError('OPERATOR_AUTH_UNAVAILABLE',503)
        header=request.headers.get('authorization','')
        supplied=header[7:] if header.startswith('Bearer ') else ''
        if configured and secrets.compare_digest(supplied,configured):
            origin=request.headers.get('origin')
            if origin and origin not in self.origins():raise ServiceError('OPERATOR_ORIGIN_FORBIDDEN',403)
            return {'authenticated':True,'mode':'BEARER','expires_at':None}
        token=request.cookies.get(self.cookie)
        if token:
            self.local_guard(request)
            with self.lock:
                expiry=self.sessions.get(token)
                if expiry is not None and expiry>self.clock():return {'authenticated':True,'mode':'LOCAL_LOOPBACK','expires_at':expiry}
                self.sessions.pop(token,None)
            raise ServiceError('OPERATOR_UNAUTHORIZED',401)
        if not configured and not self.local_enabled():raise ServiceError('OPERATOR_AUTH_UNAVAILABLE',503)
        raise ServiceError('OPERATOR_UNAUTHORIZED',401)
    def revoke(self,request,response):
        self.require(request)
        with self.lock:self.sessions.pop(request.cookies.get(self.cookie),None)
        response.delete_cookie(self.cookie,path='/api/v1',httponly=True,samesite='strict')
        return {'authenticated':False,'mode':'UNAVAILABLE','expires_at':None}

def execute(argv,timeout):
    # Private subprocess output is never returned over HTTP. Temporary files keep
    # a provider/CLI failure from accumulating an unbounded in-memory pipe.
    with tempfile.TemporaryFile() as output,tempfile.TemporaryFile() as error:
        result=subprocess.run(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=error,timeout=timeout,shell=False,creationflags=0x08000000 if os.name=="nt" else 0)
        size=output.tell()
        if size>2_000_000:raise ServiceError('JOB_OUTPUT_INVALID',500)
        output.seek(0)
        try:
            payload=json.loads(output.read().decode('utf-8'))
            if not isinstance(payload,dict):raise ValueError('Expected object')
            payload['_process_failed']=result.returncode!=0
            return payload
        except (ValueError,UnicodeError):raise ServiceError('JOB_PROCESS_FAILED' if result.returncode else 'JOB_OUTPUT_INVALID',500)

def frozen_capture_inventory(payload,directory=None):
    """Read the exact frozen manifest without constructing a writable store."""
    root=Path(directory or data_root()).resolve();path=root/'collection.sqlite'
    try:
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as connection:
            row=connection.execute('SELECT payload,hash FROM sources WHERE batch_id=?',(payload['batch_id'],)).fetchone()
        if row is None or hashlib.sha256(row[0].encode()).hexdigest()!=row[1]:raise ValueError('Source integrity')
        manifest=json.loads(row[0])
        if datetime.fromisoformat(manifest['archived_at'].replace('Z','+00:00'))>=datetime.fromisoformat(payload['cutoff'].replace('Z','+00:00')):raise ValueError('Late source manifest')
        snapshots=manifest['snapshots'];failures=manifest['failures']
        if not isinstance(snapshots,dict) or not isinstance(failures,dict):raise ValueError('Source shape')
        from research.post_v5.archive import RunArchive
        archive=RunArchive(root/'archive')
        for identifier in snapshots.values():archive.get('snapshots',identifier)
        members={item['ticker'] for item in universe()[0]['members']}
        return {'captured_sources':len(snapshots),'failed_sources':len(failures),'required_stock_sources':len(members.intersection(snapshots))}
    except (OSError,ValueError,KeyError,sqlite3.Error):raise ServiceError('JOB_OUTPUT_INVALID',500)

def nonnegative_count(payload,key):
    value=payload.get(key)
    if type(value) is not int or value<0:raise ServiceError('JOB_OUTPUT_INVALID',500)
    return value

def validate_job_result(kind,payload):
    """Only the existing commands' known result shapes can establish success."""
    status=payload.get('status');health=payload.get('health')
    if kind=='verify':
        for key in ('v5_files_verified','additional_files_verified','original_forecasts_verified'):nonnegative_count(payload,key)
        if payload.get('production_promotion')!='BLOCKED':raise ServiceError('JOB_OUTPUT_INVALID',500)
        return
    if kind=='resolve':
        if status not in {'RESOLVED','NO_MATURED_FORECASTS'}:raise ServiceError('JOB_OUTPUT_INVALID',500)
        nonnegative_count(payload,'matured');nonnegative_count(payload,'pending')
        if status=='RESOLVED' and not isinstance(payload.get('source_failures'),dict):raise ServiceError('JOB_OUTPUT_INVALID',500)
        if status=='NO_MATURED_FORECASTS' and payload['matured']!=0:raise ServiceError('JOB_OUTPUT_INVALID',500)
        return
    if status=='NOT_DUE':return
    if status=='SKIPPED' and (health=='SKIPPED' or payload.get('reason')=='NON_SESSION'):return
    if kind=='capture' and status=='INPUTS_FROZEN':
        nonnegative_count(payload,'captured_sources');nonnegative_count(payload,'failed_sources');return
    if kind=='capture' and status=='INPUTS_ALREADY_FROZEN':return
    if status not in {'COMPLETED','ALREADY_EXISTS'} or health not in {'HEALTHY','DEGRADED','FAILED'}:raise ServiceError('JOB_OUTPUT_INVALID',500)
    if health=='FAILED':return
    success=nonnegative_count(payload,'successful_forecasts');failure=nonnegative_count(payload,'failed_forecasts');abstained=nonnegative_count(payload,'abstained_forecasts')
    if success+failure!=400 or abstained>success:raise ServiceError('JOB_OUTPUT_INVALID',500)

class OperatorJobs:
    terminal={'SUCCESS','SUCCESS_WITH_DEGRADATION','SKIPPED_NON_SESSION','FAILED'}
    def __init__(self,directory=None,executor=execute,dispatch=None,clock=utcnow,timeout=7200,capture_inventory=frozen_capture_inventory):
        self.directory=Path(directory or ROOT/'data/web_platform/jobs').resolve()
        self.executor=executor;self.clock=clock;self.timeout=min(max(int(timeout),1),7200)
        self.capture_inventory=capture_inventory
        self.dispatch=dispatch or (lambda task:Thread(target=task,daemon=True,name='stockpro-operator-job').start())
        self.lock=RLock();self.records={};self.active=None;self.queue_handle=None
        # Recover only operational records. An interrupted CLI is not blindly
        # reexecuted: the existing command's resume semantics govern a new request.
        if self.directory.exists():
            recovery_handle=self._claim()
            for path in self.directory.glob('*.jsonl'):
                if not re.fullmatch(r'[0-9a-f]{32}',path.stem):continue
                try:
                    lines=path.read_text(encoding='utf-8').splitlines()
                    item=JobRecord.model_validate(json.loads(lines[-1])).model_dump(mode='json')
                    if item['run_id']!=path.stem:continue
                    self.records[item['run_id']]=item
                    if recovery_handle is not None and item['status'] not in self.terminal:
                        self._append(dict(item,status='FAILED',error_code='JOB_INTERRUPTED',finished_at=self.clock().isoformat()))
                except (OSError,ValueError,IndexError):continue
            self._release(recovery_handle)
    def _claim(self):
        self.directory.mkdir(parents=True,exist_ok=True)
        stream=(self.directory/'operator.lock').open('a+b')
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            return stream
        except (OSError,BlockingIOError):stream.close();return None
    def _release(self,stream):
        if stream is None:return
        try:
            stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        finally:stream.close()
    def command(self,kind,origin):
        if kind in {'capture','finalize'}:
            try:session=date.fromisoformat(origin).isoformat()
            except (ValueError,TypeError):raise ServiceError('INVALID_REQUEST',422)
            return [sys.executable,'-m','jobs.capture_prospective_forecasts','--stage',kind,'--session',session]
        if origin is not None:raise ServiceError('INVALID_REQUEST',422)
        module={'resolve':'jobs.resolve_matured_outcomes','verify':'scripts.verify_automation_preservation'}.get(kind)
        if module is None:raise ServiceError('INVALID_REQUEST',422)
        return [sys.executable,'-m',module]
    def _append(self,item):
        previous=self.records.get(item['run_id'])
        if previous and previous['status'] in self.terminal:raise ServiceError('JOB_IMMUTABLE',409)
        item=JobRecord.model_validate(item).model_dump(mode='json')
        self.directory.mkdir(parents=True,exist_ok=True)
        with (self.directory/(item['run_id']+'.jsonl')).open('a',encoding='utf-8') as out:
            out.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())
        self.records[item['run_id']]=item
    def enqueue(self,kind,origin=None):
        self.command(kind,origin)
        with self.lock:
            if self.active:
                item=self.records[self.active]
                if item['job_type']==kind and item['expected_origin']==origin:return copy.deepcopy(item)
                raise ServiceError('OPERATOR_JOB_BUSY',409)
            self.queue_handle=self._claim()
            if self.queue_handle is None:
                for record in self.records.values():
                    current=self.get(record['run_id'])
                    if current['status'] not in self.terminal and current['job_type']==kind and current['expected_origin']==origin:return current
                raise ServiceError('OPERATOR_JOB_BUSY',409)
            identity=uuid.uuid4().hex
            item=JobRecord(run_id=identity,scheduled_at=self.clock(),job_type=kind,protocol_version=PROTOCOL,expected_origin=origin,status='QUEUED').model_dump(mode='json')
            try:self._append(item);self.active=identity
            except Exception:
                self._release(self.queue_handle);self.queue_handle=None;raise
        try:self.dispatch(lambda:self.run(identity))
        except Exception:
            with self.lock:
                self._append(dict(item,status='FAILED',error_code='JOB_DISPATCH_FAILED',finished_at=self.clock().isoformat()));self.active=None;self._release(self.queue_handle);self.queue_handle=None
            raise ServiceError('JOB_DISPATCH_FAILED',503)
        return copy.deepcopy(item)
    def run(self,identity):
        with self.lock:
            item=self.records.get(identity)
            if not item or item['status']!='QUEUED':return
            item=dict(item,status='RUNNING',actual_started_at=self.clock().isoformat());self._append(item)
        try:
            payload=self.executor(self.command(item['job_type'],item['expected_origin']),self.timeout)
            if not isinstance(payload,dict):raise ServiceError('JOB_OUTPUT_INVALID')
            validate_job_result(item['job_type'],payload)
            status=payload.get('status');health=payload.get('health');reason=payload.get('reason')
            no_required_inputs=False
            if item['job_type']=='capture' and status in {'INPUTS_FROZEN','INPUTS_ALREADY_FROZEN'}:
                if status=='INPUTS_FROZEN' and payload['captured_sources']==0:no_required_inputs=True
                else:
                    inventory=self.capture_inventory(payload)
                    for key in ('captured_sources','failed_sources','required_stock_sources'):nonnegative_count(inventory,key)
                    if status=='INPUTS_FROZEN' and any(payload[key]!=inventory[key] for key in ('captured_sources','failed_sources')):raise ServiceError('JOB_OUTPUT_INVALID')
                    payload.update(captured_sources=inventory['captured_sources'],failed_sources=inventory['failed_sources'])
                    no_required_inputs=inventory['required_stock_sources']==0
            failed=bool(payload.get('_process_failed')) or no_required_inputs or health=='FAILED' or status=='NOT_DUE' or reason in {'MISSED_WINDOW','INPUTS_NOT_CAPTURED','RUNTIME_IDENTITY_CHANGED'}
            final='FAILED' if failed else 'SKIPPED_NON_SESSION' if status=='SKIPPED' and (health=='SKIPPED' or reason=='NON_SESSION' or payload.get('failure_reason')=='NON_SESSION') else 'SUCCESS_WITH_DEGRADATION' if health in {'DEGRADED','PARTIAL'} or payload.get('failed_stocks',0)>0 or payload.get('failed_sources',0)>0 or payload.get('invalid',0)>0 or payload.get('source_failures') else 'SUCCESS'
            aliases={'success_count':('successful_forecasts','matured','original_forecasts_verified'),'failure_count':('failed_forecasts','invalid'), 'abstention_count':('abstained_forecasts',),'provider_failure_count':('provider_failures',),'data_quality_failure_count':('data_quality_failures',),'captured_source_count':('captured_sources',),'failed_source_count':('failed_sources',),'pending_outcome_count':('pending',)}
            for public,keys in aliases.items():
                value=next((payload[k] for k in keys if type(payload.get(k)) is int and payload[k]>=0),None)
                item[public]=value
            if isinstance(payload.get('source_failures'),dict):
                item['provider_failure_count']=sum(value.get('code')=='PROVIDER_FAILURE' for value in payload['source_failures'].values() if isinstance(value,dict))
                item['data_quality_failure_count']=sum(value.get('code')=='DATA_QUALITY_FAILURE' for value in payload['source_failures'].values() if isinstance(value,dict))
            item['duplicate_count']=int(status=='ALREADY_EXISTS') if status else payload.get('duplicate_prevention_events')
            # The status is a bounded enum, never raw reason/error strings or paths.
            safe_status={'SKIPPED','NOT_DUE','ALREADY_EXISTS','INPUTS_ALREADY_FROZEN','INPUTS_FROZEN','CAPTURED','SEALED','COMPLETED','COMPLETE','SUCCESS','FAILED','RESOLVED','NO_MATURED_FORECASTS'}
            item.update(status=final,result_status=status if status in safe_status else None,error_code='JOB_PROCESS_FAILED' if payload.get('_process_failed') else 'JOB_REQUIRED_INPUTS_UNAVAILABLE' if no_required_inputs else 'JOB_NOT_DUE' if status=='NOT_DUE' else 'JOB_RESEARCH_GUARD_FAILED' if failed else None)
        except Exception as error:
            item.update(status='FAILED',error_code='JOB_TIMEOUT' if isinstance(error,(TimeoutError,subprocess.TimeoutExpired)) else error.code if isinstance(error,ServiceError) else 'JOB_EXECUTION_FAILED')
        with self.lock:
            try:self._append(dict(item,finished_at=self.clock().isoformat()))
            finally:self.active=None;self._release(self.queue_handle);self.queue_handle=None
    def get(self,identity):
        with self.lock:
            if identity not in self.records:raise ServiceError('JOB_NOT_FOUND',404)
            try:
                path=self.directory/(identity+'.jsonl')
                self.records[identity]=JobRecord.model_validate(json.loads(path.read_text(encoding='utf-8').splitlines()[-1])).model_dump(mode='json')
            except (OSError,ValueError,IndexError):raise ServiceError('JOB_RECORD_UNAVAILABLE')
            return copy.deepcopy(self.records[identity])
    def page(self,offset=0,limit=100):
        with self.lock:
            rows=sorted(self.records.values(),key=lambda r:r['scheduled_at'],reverse=True)
            return {'items':copy.deepcopy(rows[offset:offset+limit]),'total':len(rows),'offset':offset,'limit':limit}
    def status(self):
        with self.lock:return {'running_job':self.get(self.active) if self.active else None,'timeout_seconds':self.timeout}
