"""Verify consumed evidence and frozen files without training or final-test reuse."""
from datetime import datetime,timezone
import compileall
import hashlib
import json
from pathlib import Path
import re
import sys
import subprocess
from urllib.request import urlopen
import pandas as pd
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    lock=json.loads((ROOT/'reports/validation/FINAL_PROTOCOL_LOCK.json').read_text(encoding='utf-8'))
    consumed=json.loads((ROOT/'reports/validation/FINAL_TEST_CONSUMED.json').read_text(encoding='utf-8'))
    report=json.loads((ROOT/'reports/validation/FINAL_COMPARISON.json').read_text(encoding='utf-8'))
    active=json.loads((ROOT/'models/v5/ACTIVE_RESEARCH.json').read_text(encoding='utf-8'))
    assert lock['model_version']==consumed['model_version']==report['model_version']==active['model_version']
    assert active['models']==lock['models']
    frozen={**lock['source_hashes'],**lock['artifact_hashes']}
    for relative,expected in frozen.items():
        assert digest(ROOT/relative)==expected,relative+' changed after locking'
    identities=list(frozen)
    process=subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'cat-file','--batch'],
        input=(''.join(':'+name.replace(chr(92),'/')+chr(10) for name in identities)).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=ROOT,check=True)
    cursor=0
    for relative in identities:
        newline=process.stdout.index(b'\n',cursor)
        header=process.stdout[cursor:newline].split()
        assert len(header)==3 and header[1]==b'blob',relative+' missing staged blob'
        size=int(header[2]); cursor=newline+1
        content=process.stdout[cursor:cursor+size]; cursor+=size+1
        assert hashlib.sha256(content).hexdigest()==frozen[relative],relative+' staged bytes differ from lock'
    assert report['final_test_evaluated'] is True and report['point_in_time_verified'] is False
    assert report['research_metric_gate_passed'] is False and report['abstention_rate']==1.0
    rows=pd.read_csv(ROOT/'reports/validation/FINAL_COMPARISON.csv')
    v5=rows[rows.method=='v5_direct']
    assert v5.abstain.all() and not v5.duplicated(['ticker','origin_session','horizon']).any()
    for key,counts in report['sample_counts'].items():
        family,horizon=key.rsplit('_',1)
        assert len(v5[(v5.family==family)&(v5.horizon==int(horizon))])==counts['n']
    docs=['models/README.md','README.md','docs/V5_TASK_BOARD.md','docs/V5_COMPLETION_REPORT.md','docs/V5_IMPLEMENTATION_STATUS.md',
        'docs/V5_ARCHITECTURE.md','docs/V5_DEPLOYMENT.md','docs/DATA_SOURCES.md','reports/V4_VS_V5.md']
    for relative in docs:
        document=ROOT/relative
        text=document.read_text(encoding='utf-8')
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
            if ':' not in target and not target.startswith('#'):
                linked=(document.parent/target.split('#')[0]).resolve()
                assert linked==(ROOT/'reports/validation/V5_VERIFICATION.json').resolve() or linked.exists(),relative+': '+target
    pages={}
    for relative in ['app.py','pages/forecast.py','pages/validate.py']:
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
        if relative!='app.py': app.switch_page(relative).run()
        assert not app.exception,[e.message for e in app.exception]
        pages[relative]='PASS'
        if relative=='pages/validate.py':
            text=' '.join(str(e.value) for e in app.markdown)
            assert 'frozen research performance gate FAILED' in text
    for directory in ['core','pages','utils','scripts']: assert compileall.compile_dir(str(ROOT/directory),quiet=1)
    assert compileall.compile_file(str(ROOT/'app.py'),quiet=1)
    with urlopen('http://127.0.0.1:8515/_stcore/health',timeout=5) as response: health=response.read().decode().strip()
    assert health=='ok'
    test_run=json.loads((ROOT/'reports/validation/V5_TEST_RUN.json').read_text(encoding='utf-8'))
    assert test_run['exit_code']==0
    evidence={'verified_at':datetime.now(timezone.utc).isoformat(),'model_version':report['model_version'],
        'scope':'research_only','recorded_full_regression':test_run,
        'compileall':{'result':'PASS','command':'python -m compileall -q core pages utils scripts app.py'},
        'streamlit_health':{'port':8515,'result':health,'checked_at':datetime.now(timezone.utc).isoformat()},
        'functional_pages':pages,'browser_visual_accessibility_qa':'NOT DONE: browser runtime returned no available browser',
        'frozen_files_verified':len(frozen),'staged_frozen_blobs_verified':len(identities),'documentation_links':'PASS','v5_final_row_count':len(v5),
        'comparison_sha256':digest(ROOT/'reports/validation/FINAL_COMPARISON.json'),
        'predictions_sha256':digest(ROOT/'reports/validation/FINAL_COMPARISON.csv'),
        'research_metric_gate_passed':False,'production_promotion':'BLOCKED','abstention_rate':1.0}
    (ROOT/'reports/validation/V5_VERIFICATION.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(evidence,indent=2))
if __name__=='__main__': main()
