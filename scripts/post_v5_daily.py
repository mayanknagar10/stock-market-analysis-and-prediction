"""One complete prospective capture/shadow/maturity cycle, with immutable operation log."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.post_v5.preservation import verify
from research.post_v5.archive import RunArchive
from research.post_v5.shadow import DIRECTORY,STUDY

def main():
    verify(ROOT,ROOT/'reports/post_v5/V5_PRESERVATION.json')
    identifier='operation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    directory=DIRECTORY/'operations'/identifier;directory.mkdir(parents=True,exist_ok=False)
    environment=os.environ.copy();environment['PYTHONPATH']=str(ROOT/'.deps')+os.pathsep+str(ROOT)+os.pathsep+environment.get('PYTHONPATH','')
    operation={'operation_id':identifier,'study_id':STUDY,'started_at':datetime.now(timezone.utc).isoformat(),
        'scope':'research_only','steps':{},'training_permitted':False,'protocol_sha256':hashlib.sha256((ROOT/'config/research'/STUDY/'PROTOCOL.json').read_bytes()).hexdigest()}
    def job(name,args):
        result=subprocess.run([sys.executable,str(ROOT/'scripts'/name),*args],cwd=ROOT,env=environment,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        payload=result.stdout.encode('utf-8')
        with (directory/(name+'.log')).open('xb') as stream:stream.write(payload)
        operation['steps'][name]={'exit_code':result.returncode,'log_sha256':hashlib.sha256(payload).hexdigest()}
        if result.returncode:raise RuntimeError(name+' failed; inspect private operation log')
        return result.stdout
    try:
        output=job('post_v5_capture.py',[])
        matches=re.findall(r'^RUN_ID ([0-9a-f]{64})$',output,flags=re.MULTILINE)
        if len(matches)!=1:raise ValueError('Capture must return one unambiguous run identity')
        run_id=matches[0];operation['run_id']=run_id
        job('post_v5_shadow.py',['--run-id',run_id]);job('post_v5_resolve.py',['--run-id',run_id])
        verify(ROOT,ROOT/'reports/post_v5/V5_PRESERVATION.json')
        operation['status']='completed; source eligibility and forecast failures remain explicit in batch reports'
    except Exception as error:
        operation['status']='failed';operation['error']=str(error)
    operation['finished_at']=datetime.now(timezone.utc).isoformat()
    RunArchive(DIRECTORY).put('attempts',operation)
    public=ROOT/'reports/post_v5/operations'/f'{identifier}.json';public.parent.mkdir(parents=True,exist_ok=True)
    with public.open('x',encoding='utf-8') as stream:json.dump(operation,stream,indent=2)
    print(json.dumps(operation,indent=2))
    if operation['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
