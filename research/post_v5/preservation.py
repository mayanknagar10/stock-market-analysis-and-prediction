"""Seal original V5 bytes. New experiments have no write access through this module."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
PROTECTED=['core','models/v5','reports/validation','reports/baseline','data/snapshots','data/v5/snapshots',
    'data/v5/REFERENCE_CAPTURE.json','data/v5/predictions.sqlite','data/v5/predictions.sqlite-wal','data/v5/predictions.sqlite-shm',
    'reports/V4_VS_V5.md','config/context_sources.json','config/v5_validation.json',
    'scripts/evaluate_final.py','docs/V5_COMPLETION_REPORT.md','docs/V5_TASK_BOARD.md','docs/V5_IMPLEMENTATION_STATUS.md',
    'docs/V5_ARCHITECTURE.md','docs/V5_DEPLOYMENT.md']
def files(root):
    root=Path(root);result=[]
    for name in PROTECTED:
        file=root/name
        if file.is_dir(): result.extend(p for p in file.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.log')
        elif file.is_file(): result.append(file)
    result.extend(p for p in (root/'models').glob('universal_*') if p.is_file())
    return sorted(set(result))
def seal(root,output,commit):
    root=Path(root);output=Path(output)
    record={'sealed_at':datetime.now(timezone.utc).isoformat(),'commit':commit,'model_version':'research-20261008T190859',
        'files':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files(root)}}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream: json.dump(record,stream,indent=2)
    return record

def verify(root,manifest):
    root=Path(root);doc=json.loads(Path(manifest).read_text(encoding='utf-8'))
    for relative,expected in doc['files'].items():
        file=root/relative
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
            raise ValueError('Protected V5 file modified or missing: '+relative)
    return {'verified_files':len(doc['files']),'commit':doc['commit'],'model_version':doc['model_version']}
