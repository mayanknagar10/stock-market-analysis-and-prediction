"""Read pinned historical evidence. No evaluation, fitting, or ledger constructors."""
import json
import re
from application.config import ROOT,MODEL
from application.services.prospective import ServiceError
from application.services.analytics import clean
from application.web_schemas import HistoricalEvidenceResponse

def private_path(value):
    if '\\' in value or ':/' in value:return True
    if re.search(r'(?:^|[\s\"\x27(])(?:/|~/|\.\.?/)\S+',value):return True
    if re.search(r'\b(?:data|models|reports|research|config|home|tmp|private|secrets)/\S+',value,re.IGNORECASE):return True
    if re.search(r'\b[\w.-]+/(?:[\w.-]+/)*[\w.-]+\.[a-zA-Z0-9]+\b',value):return True
    if re.fullmatch(r'[a-z0-9_.-]+(?:/[a-z0-9_.-]+)+',value) and not re.fullmatch(r'(?:mae|rmse|mse|r2|brier|log_loss)/(?:mae|rmse|mse|r2|brier|log_loss)',value,re.IGNORECASE):return True
    return False

def public_metrics(value):
    # Only saved metric/breakdown subtrees are admitted by the outer allowlist.
    if isinstance(value,dict):return {k:public_metrics(v) for k,v in value.items() if not private_path(str(k)) and k not in {'parameters','feature_values','feature_names','model','models','sources','source_paths','artifact_path','training_parameters','raw_payload'}}
    if isinstance(value,list):return [public_metrics(v) for v in value]
    if isinstance(value,str):return None if private_path(value) else value
    return clean(value)

class HistoricalEvidence:
    def __init__(self,root=ROOT):self.root=root
    def read(self,report):
        paths={'development':self.root/'reports/validation'/MODEL/'DEVELOPMENT.json','final':self.root/'reports/validation/FINAL_COMPARISON.json','quarantine':self.root/'models/v5/QUARANTINE.json'}
        if report not in paths:raise ServiceError('INVALID_REQUEST',422)
        try:document=json.loads(paths[report].read_text(encoding='utf-8'))
        except (OSError,ValueError):raise ServiceError('HISTORICAL_EVIDENCE_UNAVAILABLE',404)
        if report!='quarantine' and document.get('model_version')!=MODEL:raise ServiceError('SCHEMA_MISMATCH')
        metrics={};models=[]
        if report=='final':
            for key in ('sample_counts','relative_metrics','results','frozen_research_gate_counts','research_metric_gate_passed','abstention_rate','final_test_start','final_test_evaluated'):
                if key in document:metrics[key]=public_metrics(document[key])
            if document.get('research_metric_gate_passed') is not False:raise ServiceError('SCHEMA_MISMATCH')
        elif report=='development':
            for name,item in document.get('models',{}).items():
                metadata=item.get('metadata',{})
                models.append({'name':name,'family':metadata.get('family'),'horizon':metadata.get('horizon'),
                    'metadata':{key:public_metrics(metadata[key]) for key in ('scope','limitation','training_cutoff','calibration_state','calibration_accepted','raw_probability_metrics','relative_evidence','validation_metrics','n_train','n_validation','feature_schema_hash') if key in metadata},
                    'policy':public_metrics(item.get('policy',{})),'breakdowns':public_metrics(item.get('breakdowns',{})),'zero_return_metrics':public_metrics(item.get('zero_return_metrics',{}))})
            metrics={'final_test_evaluated':document.get('final_test_evaluated',False),'final_test_start':document.get('final_test_start')}
        else:
            models=[{key:public_metrics(v.get(key)) for key in ('model_version','reason','state','scope')} for v in document.get('versions',[])]
        limitations=[public_metrics(value) or 'Private artifact details omitted.' for value in document.get('limitations',[])]+['Retrospective revised source vintages and current membership; not certified point in time.','FAILED historical V5 gate remains frozen; prospective evidence is a separate stream.','Overlapping horizons and assets are dependent; no validated 90% interval.']
        locks={}
        for key,filename in [('protocol_lock','FINAL_PROTOCOL_LOCK.json'),('final_test_consumed','FINAL_TEST_CONSUMED.json')]:
            try:
                saved=json.loads((self.root/'reports/validation'/filename).read_text(encoding='utf-8'))
                locks[key]=public_metrics({k:v for k,v in saved.items() if k in {'model_version','scope','locked_at','consumed_at','final_test_start','final_test_end','status','state','evaluated_at','locked','protocol_version','final_test_evaluated','locked_before_evaluation','report_hash','research_only'}})
            except (OSError,ValueError):locks[key]={}
        return HistoricalEvidenceResponse(report=report,model_version=MODEL,metrics=metrics,models=models,limitations=limitations,**locks)
