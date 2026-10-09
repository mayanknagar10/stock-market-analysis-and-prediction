"""Public contracts: research semantics are explicit, private paths/payloads are absent."""
from typing import Any,Literal
from datetime import datetime
from pydantic import BaseModel,ConfigDict,Field,field_validator,AwareDatetime
class Contract(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
class ErrorDetail(Contract):
    code:str
    message:str
    request_id:str
class ErrorResponse(Contract):
    error:ErrorDetail
class ResearchState(Contract):
    research_state:Literal["RESEARCH_ONLY"]="RESEARCH_ONLY"
    recommendation_allowed:Literal[False]=False
    production_promotion:Literal["BLOCKED"]="BLOCKED"
    model_version:str
    protocol_version:str
class HorizonForecast(Contract):
    prediction_id:str
    horizon:Literal[1,5,10,20]
    predicted_log_return:float
    predicted_return:float
    predicted_price:float
    quantile_log_returns:list[float]
    quantile_prices:list[float]
    probability:float|None=None
    raw_probability:float|None=None
    calibration_state:str
    confidence:Literal["LOW"]="LOW"
    abstain:Literal[True]=True
    abstention_reasons:list[str]
    ood_score:float|None=None
    ood_status:Literal["IN_DOMAIN","OUT_OF_DOMAIN","UNKNOWN"]="UNKNOWN"
    feature_vector_hash:str|None=None
    regime:str
    data_quality_status:str
    training_cutoff:AwareDatetime
    feature_schema:str
    provenance_references:list[str]
    recommendation_allowed:Literal[False]=False
class ForecastResponse(ResearchState):
    ticker:str
    current_price:float
    price_timestamp:datetime
    prediction_origin:datetime
    information_cutoff:datetime
    origin_session:str
    origin_group:str
    calendar_version:str|None=None
    issuing_code_identity:str|None=None
    model_artifact_identity:str|None=None
    price_timestamp_kind:str="DAILY_ORIGIN_SESSION_LABEL"
    horizons:list[HorizonForecast]
    @field_validator("price_timestamp","prediction_origin","information_cutoff")
    @classmethod
    def timestamps_aware(cls,value):
        if value.tzinfo is None or value.utcoffset() is None:raise ValueError("Timezone required")
        return value
class Page(Contract):
    items:list[dict[str,Any]]
    total:int
    offset:int=0
    limit:int=100
class ValidationResponse(ResearchState):
    evidence_state:str
    origin_group:str
    counts:dict[str,Any]
    overall:dict[str,Any]
    groups:dict[str,dict[str,dict[str,Any]]]
    evidence_requirements:dict[str,Any]
    limitations:list[str]
class HealthResponse(Contract):
    status:str
    timestamp:datetime
    research_state:Literal["RESEARCH_ONLY"]="RESEARCH_ONLY"
    recommendation_allowed:Literal[False]=False
class CollectionResponse(Contract):
    protocol_version:str
    health:str
    batches:list[dict[str,Any]]
    open_batches:list[dict[str,Any]]
    duplicate_prevention_events:int
    research_state:Literal["RESEARCH_ONLY"]="RESEARCH_ONLY"
class StockResponse(Contract):
    ticker:str
    retrieved_at:datetime
    provider:str
    data_quality_status:str
    values:dict[str,Any]
class ModelResponse(ResearchState):
    health:str
    calibration_state:str
    training_lineage:str
    successor_trained:Literal[False]=False
    details:dict[str,Any]=Field(default_factory=dict)

class HistoryBar(Contract):
    timestamp:AwareDatetime
    open:float|None
    high:float|None
    low:float|None
    close:float|None
    adjusted_close:float|None
    volume:float|None
    dividends:float|None
    stock_splits:float|None
class HistoryResponse(Contract):
    items:list[HistoryBar]
    total:int
    offset:int=0
    limit:int=100
class SearchItem(Contract):
    ticker:str
    company:str
    industry:str
    isin:str
class SearchResponse(Contract):
    items:list[SearchItem]
    total:int
    offset:int=0
    limit:int=100

class RealizedOutcome(Contract):
    status:str
    actual_return:float|None
    actual_log_return:float|None
    actual_future_price:float|None
    absolute_error:float|None
    squared_error:float|None
    direction_correct:bool|None
    interval_80_hit:bool|None
    interval_50_hit:bool|None
    benchmark_return:float|None
    sector_return:float|None
    alpha_market:float|None
    alpha_sector:float|None
    maturity_timestamp:AwareDatetime|None
    resolved_at:AwareDatetime|None
    return_convention:str|None
    timestamp_correction:str|None
class HistoricalForecast(ResearchState):
    prediction_id:str
    ticker:str
    forecast_as_of:AwareDatetime
    information_cutoff:AwareDatetime
    price_timestamp:AwareDatetime
    origin_session:str
    origin_group:Literal["canonical","legacy"]
    horizon:Literal[1,5,10,20]
    current_price:float
    central_log_return:float
    central_price:float
    quantiles:list[float]
    p_positive_calibrated:float|None
    raw_p_positive:float|None
    calibration_state:str|None
    regime:str
    ood_score:float|None
    training_cutoff:AwareDatetime
    feature_schema_hash:str
    provenance_references:list[str]
    confidence:Literal["LOW"]
    abstain:Literal[True]
    trust_reasons:list[str]|None
    evaluation_sealed:bool
    outcome:RealizedOutcome|None
    outcome_state:str
class ForecastHistoryResponse(Contract):
    items:list[HistoricalForecast]
    total:int
    offset:int=0
    limit:int=100
