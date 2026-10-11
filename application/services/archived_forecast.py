"""Saved forecast extras and explicit hypothetical inference from cloned archive inputs."""
import copy
import json
import sqlite3
import numpy as np
import pandas as pd
from application.config import ROOT,MODEL,data_root
from application.services.analytics import clean
from application.services.prospective import ProspectiveService,ServiceError,read_table,FINAL_START
from application.schemas import HistoricalForecast
from application.web_schemas import ResearchHistoryResponse,ArchivedDetail,ScenarioResponse,AnaloguesResponse,SavedDrivers,RelativeForecast
from core.data.contracts import DataSnapshot
from core.forecasting.scenarios import ScenarioEngine
from core.forecasting.inference import historical_analogues
from application.services.historical_evidence import public_metrics

def public_forecast(row,group):
    keys=('prediction_id','ticker','forecast_as_of','generated_at','information_cutoff','origin_session','horizon','current_price','central_log_return','central_price','quantiles','p_positive_calibrated','raw_p_positive','calibration_state','regime','ood_score','model_version','training_cutoff','feature_schema_hash','confidence','abstain','trust_reasons','evaluation_sealed')
    item={key:row.get(key) for key in keys}
    item['raw_p_positive']=row.get('raw_p_positive',row.get('p_positive') if row.get('p_positive_calibrated') is None else None)
    item.update(protocol_version=row.get('protocol_version','legacy-manual-v1'),origin_group=group,provenance_references=row.get('snapshot_ids',[]),recommendation_allowed=False)
    item['price_timestamp']=row.get('price_timestamp') or pd.Timestamp(row['origin_session'],tz='Asia/Kolkata' if row['market']=='india_equity' else 'America/New_York').tz_convert('UTC').isoformat()
    outcome=None if row.get('evaluation_sealed') else row.get('outcome')
    allowed=('status','actual_return','actual_log_return','actual_future_price','absolute_error','squared_error','direction_correct','interval_80_hit','interval_50_hit','benchmark_return','sector_return','alpha_market','alpha_sector','maturity_timestamp','resolved_at','return_convention','timestamp_correction')
    item['outcome']={key:outcome.get(key) for key in allowed} if outcome else None
    item['outcome_state']='SEALED' if row.get('evaluation_sealed') else outcome['status'] if outcome else 'OUTCOME_PENDING'
    try:return HistoricalForecast.model_validate(item)
    except ValueError:raise ServiceError('SCHEMA_MISMATCH')

class ArchivedForecasts:
    def __init__(self,prospective=None,input_loader=None,engine_factory=ScenarioEngine):
        self.prospective=prospective or ProspectiveService();self.input_loader=input_loader or self._load_inputs;self.engine_factory=engine_factory
    def history(self,group='canonical',ticker=None,offset=0,limit=100,horizon=None):
        if group not in {'canonical','legacy','v5'} or offset<0 or not 1<=limit<=1000:raise ServiceError('INVALID_REQUEST',422)
        rows=[row for row in self._rows(group) if (ticker is None or row['ticker']==ticker) and (horizon is None or row['horizon']==horizon)]
        rows.sort(key=lambda r:r['prediction_id']);rows.sort(key=lambda r:r['forecast_as_of'],reverse=True)
        return ResearchHistoryResponse(items=[public_forecast(r,group) for r in rows[offset:offset+limit]],total=len(rows),offset=offset,limit=limit)
    def _rows(self,group):
        if group!='v5':return self.prospective.rows(group)
        path=ROOT/'data/v5/predictions.sqlite'
        records=read_table(path,'forecasts');read_table(path,'outcomes')
        outcomes={}
        if path.exists():
            with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as connection:
                outcomes={identity:json.loads(payload) for identity,payload in connection.execute('SELECT prediction_id,payload FROM outcomes')}
        for row in records:
            outcome=outcomes.get(row['prediction_id'])
            row['evaluation_sealed']=row.get('origin_session','')>=FINAL_START or bool(outcome and outcome.get('maturity_timestamp','')[:10]>=FINAL_START)
            row['outcome']=None if row['evaluation_sealed'] else outcome
        return records
    def _record(self,identity,group):
        for row in self._rows(group):
            if row['prediction_id']==identity:return row
        raise ServiceError('FORECAST_NOT_FOUND',404)
    def detail(self,identity,group='canonical'):
        row=self._record(identity,group)
        raw_drivers=row.get('drivers');drivers=None
        if isinstance(raw_drivers,dict):
            try:drivers=SavedDrivers.model_validate(clean({key:raw_drivers[key] for key in SavedDrivers.model_fields if key in raw_drivers}))
            except ValueError:raise ServiceError('SCHEMA_MISMATCH')
        relatives={}
        for key,value in row.get('relative_forecasts',{}).items():
            if key not in {'alpha_market','alpha_sector','market','sector'}:continue
            relatives[key]=RelativeForecast.model_validate({name:public_metrics(value[name]) for name in RelativeForecast.model_fields if name in value}) if isinstance(value,dict) else None
        return ArchivedDetail(prediction=public_forecast(row,group),drivers=drivers,relative_forecasts=relatives,component_returns=clean(row.get('component_returns',{})),probability_components=clean(row.get('probability_components',{})))
    def _load_inputs(self,identity,group):
        from research.post_v5.archive import RunArchive
        from core.forecasting.service import adjusted_frame,active_research,load_research_model
        row=self._record(identity,group)
        if row.get('model_version')!=MODEL:raise ServiceError('MODEL_IDENTITY_UNAVAILABLE',404)
        pointer=active_research()
        if pointer.get('model_version')!=row['model_version']:raise ServiceError('MODEL_IDENTITY_UNAVAILABLE',404)
        archive=RunArchive(data_root()/'archive' if group=='canonical' else ROOT/'data/research/prospective-india-20261008-v1')
        try:
            settings=json.loads((ROOT/'config/context_sources.json').read_text(encoding='utf-8'))
            sources={key:archive.load_snapshot(identity) for key,identity in row['input_snapshot_ids'].items() if key==row['ticker'] or key in settings['sources']}
            stock=sources[row['ticker']]
            _,frame=adjusted_frame(stock,row['information_cutoff'])
            references={key:source for key,source in sources.items() if source.metadata.family=='research_reference'}
            bundle=load_research_model(row['market'],row['horizon'],pointer)
            if bundle.get('metadata',{}).get('model_version')!=row['model_version']:raise ValueError('Model identity')
            if bundle.get('metadata',{}).get('feature_schema_hash')!=row.get('feature_schema_hash'):raise ValueError('Feature identity')
            return {'record':row,'bundle':bundle,'stock':frame,'references':references,'sources':sources,'settings':settings}
        except (OSError,ValueError,KeyError):raise ServiceError('ARCHIVED_INPUTS_UNAVAILABLE',404)
    def _cloned_inputs(self,identity,group):
        try:
            inputs=self.input_loader(identity,group)
            cutoff=pd.Timestamp(inputs['record']['information_cutoff'])
            if cutoff.tzinfo is None:raise ValueError('Naive cutoff')
            sources=inputs.get('sources',inputs['references'])
            for source in sources.values():
                if hasattr(source,'metadata'):
                    for field in ('source_timestamp','available_at','vintage_at','fetched_at'):
                        if pd.Timestamp(getattr(source.metadata,field))>cutoff:raise ValueError('Future source')
            cloned=copy.deepcopy(inputs)
            cloned['stock']=inputs['stock'].copy(deep=True)
            cloned['references']={key:DataSnapshot(source.frame.copy(deep=True),copy.deepcopy(source.metadata)) if hasattr(source,'metadata') else source.copy(deep=True) for key,source in inputs['references'].items()}
            return cloned
        except ServiceError:raise
        except (ValueError,KeyError,TypeError):raise ServiceError('ARCHIVED_INPUTS_INELIGIBLE',422)
    def scenario(self,identity,request):
        inputs=self._cloned_inputs(identity,request.group);row=inputs['record']
        try:
            engine=self.engine_factory(inputs['bundle'],inputs['stock'],row['ticker'],row['market'],inputs['references'],inputs['settings'],row['current_price'],row['information_cutoff'])
            result=engine.run(dict(request.shocks))
            allowed=('central_return','central_price','central_log_return','quantiles','p_positive_calibrated')
            return ScenarioResponse(prediction_id=identity,shocks=request.shocks,changed_features=result.get('changed_features',[]),**clean({key:result.get(key) for key in allowed}))
        except (ValueError,KeyError):raise ServiceError('SCENARIO_UNAVAILABLE',422)
    def analogues(self,identity,group='canonical',count=20):
        if not 1<=count<=50:raise ServiceError('INVALID_REQUEST',422)
        inputs=self._cloned_inputs(identity,group);row=inputs['record'];bundle=inputs['bundle']
        try:
            frame=pd.DataFrame([row['feature_values']])[list(bundle['regression'].feature_names)]
            result=historical_analogues(bundle,frame,row['information_cutoff'],count=count)
            allowed=('n','win_rate','median_return','best_return','worst_return','distribution','timestamps','tickers','median_excess_return')
            payload={key:result[key] for key in allowed if key in result}
            return AnaloguesResponse(prediction_id=identity,status=result.get('status','AVAILABLE'),**clean(payload))
        except (ValueError,KeyError):raise ServiceError('ANALOGUES_UNAVAILABLE',404)
