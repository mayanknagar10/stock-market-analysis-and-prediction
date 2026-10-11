"""Typed additive web contracts. Research invariants and null availability are explicit."""
from datetime import date
from typing import Any, Literal
import math
from pydantic import AwareDatetime, Field, field_validator, model_validator
from application.schemas import Contract, HistoryBar, HistoricalForecast

class WebState(Contract):
    research_state: Literal['RESEARCH_ONLY']='RESEARCH_ONLY'
    recommendation_allowed: Literal[False]=False
    production_promotion: Literal['BLOCKED']='BLOCKED'

class SessionBar(HistoryBar):
    exchange_session: str

class SessionHistory(Contract):
    items: list[SessionBar]
    total: int
    offset: int=0
    limit: int=100
    exchange_timezone: str
    currency: str
    ordering: Literal['ascending_exchange_session']='ascending_exchange_session'

class Company(Contract):
    name: str|None=None
    sector: str|None=None
    industry: str|None=None
    description: str|None=None
    website: str|None=None
    exchange: str|None=None
    employees: float|None=None

class FundamentalMetrics(Contract):
    market_cap: float|None=None
    pe_ttm: float|None=None
    pe_fwd: float|None=None
    eps: float|None=None
    dividend_yield: float|None=None
    dividend_yield_provider_value: float|None=None
    beta: float|None=None
    week52_high: float|None=None
    week52_low: float|None=None
    avg_volume_10d: float|None=None
    avg_volume_3m: float|None=None
    revenue_ttm: float|None=None
    gross_margin: float|None=None
    operating_margin: float|None=None
    roe: float|None=None
    debt_equity: float|None=None

class FundamentalsResponse(Contract):
    ticker: str
    currency: str
    retrieved_at: AwareDatetime
    status: Literal['AVAILABLE','PARTIAL','UNAVAILABLE']
    company: Company
    metrics: FundamentalMetrics
    error_code: str|None=None
    vintage: Literal['CURRENT_REVISED']='CURRENT_REVISED'
    point_in_time_verified: Literal[False]=False
    provider: str='yahoo/yfinance'
    dividend_yield_source_unit: Literal['fraction','percent','unverified']='unverified'
    units: dict[str,str]=Field(default_factory=lambda:{'dividend_yield':'fraction','dividend_yield_provider_value':'provider_unit_unverified','gross_margin':'fraction','operating_margin':'fraction','roe':'fraction','market_cap':'currency','revenue_ttm':'currency','debt_equity':'provider_percent'})

class NewsItem(Contract):
    title: str
    publisher: str|None=None
    url: str|None=None
    published_at: AwareDatetime|None=None
    sentiment_score: float|None=None
    sentiment_label: str|None=None

class NewsResponse(Contract):
    ticker: str
    retrieved_at: AwareDatetime
    status: Literal['AVAILABLE','UNAVAILABLE']
    items: list[NewsItem]
    overall_score: float|None=None
    error_code: str|None=None
    vintage: Literal['CURRENT_REVISED']='CURRENT_REVISED'
    sentiment_method: str='VADER_FINANCE_LEXICON'
    point_in_time_verified: Literal[False]=False

class TechnicalPoint(Contract):
    exchange_session: str
    timestamp: AwareDatetime
    values: dict[str,float|None]

class TechnicalSeries(Contract):
    ticker: str
    group: str
    items: list[TechnicalPoint]
    total: int
    offset: int=0
    limit: int=100
    exchange_timezone: str

class RiskResponse(Contract):
    ticker: str
    benchmark: str
    aligned_observations: int
    metrics: dict[str,float|int|None]
    drawdown: list[TechnicalPoint]
    benchmark_status: str
    alignment: str='INNER_JOIN_EXCHANGE_SESSION_DATE'
    return_convention: str='adjusted_close_simple_returns'
    annual_sessions: int=252
    risk_free_rate: float=.045

class FactorExposures(Contract):
    alpha_monthly_pct: float|None
    alpha_annualised_pct: float|None
    alpha_t_stat: float|None
    alpha_p_value: float|None
    betas: dict[str,float|None]
    t_stats: dict[str,float|None]
    p_values: dict[str,float|None]
    r_squared: float|None
    adj_r_squared: float|None
    n_observations: int

class FactorMonth(Contract):
    month: str
    stock_return: float
    factors: dict[str,float]

class FactorResponse(Contract):
    ticker: str
    status: Literal['AVAILABLE','UNAVAILABLE']
    signals: dict[str,float|None]
    exposures: FactorExposures|None=None
    aligned_months: list[str]=Field(default_factory=list)
    factor_series: list[FactorMonth]=Field(default_factory=list)
    error_code: str|None=None
    vintage: Literal['CURRENT_REVISED']='CURRENT_REVISED'
    limitations: list[str]=Field(default_factory=lambda:['Descriptive US Fama-French factors; India market/currency mismatch.', 'Only completed matched calendar months; not frozen model training or prospective evidence.'])

class MarketItem(Contract):
    source: str
    symbol: str
    category: str
    price: float|None=None
    previous_close: float|None=None
    change: float|None=None
    change_percent: float|None=None
    exchange_session: str|None=None
    timestamp: AwareDatetime|None=None
    retrieved_at: AwareDatetime|None=None
    status: Literal['AVAILABLE','UNAVAILABLE']
    error_code: str|None=None

class MarketsResponse(Contract):
    items: list[MarketItem]
    total: int
    timestamp_kind: str='PROVIDER_DAILY_SESSION_LABEL'
    change_units: Literal['source_price_units']='source_price_units'
    change_percent_units: Literal['percent']='percent'

class Holding(Contract):
    ticker: str=Field(max_length=30)
    weight: float=Field(ge=0,le=1e12)

class PortfolioRequest(Contract):
    holdings: list[Holding]=Field(min_length=1,max_length=10)
    initial_value: float=Field(default=100,gt=0,le=1e12)
    @model_validator(mode='after')
    def positive_total(self):
        if sum(h.weight for h in self.holdings)<=0: raise ValueError('Positive total weight required')
        if len({h.ticker.upper().strip() for h in self.holdings})!=len(self.holdings):raise ValueError('Duplicate holding')
        return self

class ValuePoint(Contract):
    exchange_session: str
    value: float

class PortfolioResponse(WebState):
    currency: str
    weights: dict[str,float]
    series: list[ValuePoint]
    metrics: dict[str,float|int|None]
    failed_holdings: dict[str,str]
    convention: str='DAILY_CONSTANT_WEIGHT_REBALANCED_ADJUSTED_RETURNS'
    alignment: str='INNER_JOIN_EXCHANGE_SESSION_DATE'
    limitations: list[str]=Field(default_factory=lambda:['No transaction costs; daily rebalancing hypothetical; failed holdings excluded and remaining weights normalized.'])

class CompareRequest(Contract):
    tickers: list[str]=Field(min_length=2,max_length=5)
    @field_validator('tickers')
    @classmethod
    def distinct(cls,value):
        if len({s.strip().upper() for s in value})!=len(value):raise ValueError('Duplicate ticker')
        return value

class ComparePoint(Contract):
    exchange_session: str
    values: dict[str,float|None]

class CompareResponse(WebState):
    series: list[ComparePoint]
    cumulative_return_spread: list[ComparePoint]
    correlation: dict[str,dict[str,float|None]]
    rolling_correlation: list[ComparePoint]
    metrics: dict[str,dict[str,float|int|None]]
    currencies: dict[str,str]
    alignment: str='INNER_JOIN_LOCAL_SESSION_DATE_CROSS_MARKET_DESCRIPTIVE'
    limitations: list[str]=Field(default_factory=lambda:['Different exchanges close at different instants; shared local session dates are descriptive, not simultaneous observations.','Normalized returns do not convert currencies.'])

class WatchPosition(Contract):
    ticker: str
    quantity: float=Field(gt=0,le=1e12)
    average_buy: float=Field(gt=0,le=1e12)
    target: float|None=Field(default=None,gt=0,le=1e12)
    stop: float|None=Field(default=None,gt=0,le=1e12)

class WatchlistRequest(Contract):
    positions: list[WatchPosition]=Field(min_length=1,max_length=30)

class WatchItem(Contract):
    ticker: str
    currency: str
    status: str
    error_code: str|None=None
    last_price: float|None=None
    exchange_session: str|None=None
    value: float|None=None
    cost: float|None=None
    pnl: float|None=None
    pnl_percent: float|None=None
    change_percent: float|None=None
    target_reached: bool|None=None
    stop_reached: bool|None=None

class CurrencyTotal(Contract):
    value: float
    cost: float
    pnl: float

class WatchlistResponse(WebState):
    items: list[WatchItem]
    totals: dict[str,CurrencyTotal]
    convention: str='STATIC_QUANTITY_UNADJUSTED_LAST_CLOSE_PER_CURRENCY'
    percent_units: Literal['percent']='percent'

class ScreenRequest(Contract):
    tickers: list[str]=Field(min_length=1,max_length=30)
    pe_min: float|None=Field(default=None,ge=0,le=1e6)
    pe_max: float|None=Field(default=None,ge=0,le=1e6)
    beta_max: float|None=Field(default=None,ge=0,le=100)
    dividend_min: float|None=Field(default=None,ge=0,le=100)
    rsi_min: float=Field(default=0,ge=0,le=100)
    rsi_max: float=Field(default=100,ge=0,le=100)
    trend: Literal['any','above_ema50','below_ema50']='any'
    @model_validator(mode='after')
    def ranges(self):
        if self.rsi_min>self.rsi_max or (self.pe_min is not None and self.pe_max is not None and self.pe_min>self.pe_max):raise ValueError('Invalid range')
        if len({s.upper().strip() for s in self.tickers})!=len(self.tickers):raise ValueError('Duplicate ticker')
        return self

class ScreenItem(Contract):
    ticker: str
    status: Literal['MATCH','EXCLUDED','UNAVAILABLE']
    reasons: list[str]
    values: dict[str,float|None]

class ScreenResponse(WebState):
    items: list[ScreenItem]
    missing_policy: str='EXCLUDE_WHEN_REQUESTED_FILTER_DATA_MISSING_V1'
    universe: str='EXPLICIT_INPUT_LIST'

class HistoricalEvidenceResponse(WebState):
    report: Literal['development','final','quarantine']
    evidence_stream: Literal['RETROSPECTIVE_RECONSTRUCTED']='RETROSPECTIVE_RECONSTRUCTED'
    model_version: str
    final_gate: Literal['FAILED']='FAILED'
    point_in_time_verified: Literal[False]=False
    metrics: dict[str,Any]=Field(default_factory=dict)
    models: list[dict[str,Any]]=Field(default_factory=list)
    limitations: list[str]=Field(default_factory=list)
    protocol_lock: dict[str,Any]=Field(default_factory=dict)
    final_test_consumed: dict[str,Any]=Field(default_factory=dict)

class ResearchHistoryResponse(Contract):
    items: list[HistoricalForecast]
    total: int
    offset: int=0
    limit: int=100
    order: str='FORECAST_AS_OF_DESC_PREDICTION_ID_ASC'

class SavedDrivers(Contract):
    features: list[str]
    values: list[float]
    base_value: float|None=None
    label: str='Model contribution in log-return units; association, not causal proof.'
    @model_validator(mode='after')
    def aligned(self):
        if len(self.features)!=len(self.values):raise ValueError('Contribution alignment')
        return self

class RelativeForecast(Contract):
    central_excess_log_return: float|None=None
    raw_p_outperform: float|None=None
    calibrated_p_outperform: float|None=None
    validation_metrics: dict[str,Any]=Field(default_factory=dict)
    scope: Literal['research_only']='research_only'

class ArchivedDetail(WebState):
    prediction: HistoricalForecast
    drivers: SavedDrivers|None=None
    relative_forecasts: dict[str,RelativeForecast|None]=Field(default_factory=dict)
    component_returns: dict[str,float|None]=Field(default_factory=dict)
    probability_components: dict[str,float|None]=Field(default_factory=dict)
    scope: str='SAVED_ARCHIVED_INPUTS'

class ScenarioRequest(Contract):
    group: Literal['canonical','legacy']='canonical'
    shocks: dict[str,float]=Field(min_length=1,max_length=8)
    @field_validator('shocks')
    @classmethod
    def bounded_shocks(cls,value):
        allowed={'stock','market','sector','nasdaq','usd_inr','brent','vix','india_vix','gold','wti','dxy','us_yield','nikkei','hang_seng'}
        if any(k not in allowed or not math.isfinite(v) or not -.4<=v<=.4 for k,v in value.items()):raise ValueError('Unsupported shock')
        return value

class ScenarioResponse(WebState):
    prediction_id: str
    scope: Literal['HYPOTHETICAL_ARCHIVED_RESEARCH']='HYPOTHETICAL_ARCHIVED_RESEARCH'
    confidence: Literal['LOW']='LOW'
    abstain: Literal[True]=True
    central_return: float|None=None
    central_price: float|None=None
    central_log_return: float|None=None
    quantiles: list[float]|None=None
    p_positive_calibrated: float|None=None
    shocks: dict[str,float]
    changed_features: list[str]
    limitations: list[str]=Field(default_factory=lambda:['Conditional model associations are not causal stress predictions.','Original saved forecasts, model artifacts and source snapshots are unchanged.'])

class AnaloguesResponse(WebState):
    prediction_id: str
    status: str
    n: int
    win_rate: float|None=None
    median_return: float|None=None
    best_return: float|None=None
    worst_return: float|None=None
    distribution: list[float]=Field(default_factory=list)
    timestamps: list[str]=Field(default_factory=list)
    tickers: list[str]=Field(default_factory=list)
    median_excess_return: float|None=None
    limitation: str='Overlapping matured historical neighbours are dependent; not forward evidence.'

class JobRequest(Contract):
    job_type: Literal['capture','finalize','resolve','verify']
    origin_session: date|None=None
    @model_validator(mode='after')
    def session(self):
        if self.job_type in {'capture','finalize'} and self.origin_session is None:raise ValueError('Origin required')
        if self.job_type in {'resolve','verify'} and self.origin_session is not None:raise ValueError('This command does not accept an origin')
        return self

class JobRecord(WebState):
    run_id: str
    scheduled_at: AwareDatetime
    actual_started_at: AwareDatetime|None=None
    finished_at: AwareDatetime|None=None
    job_type: Literal['capture','finalize','resolve','verify']
    protocol_version: str
    expected_origin: str|None=None
    universe_size: Literal[100]=100
    status: Literal['QUEUED','RUNNING','SUCCESS','SUCCESS_WITH_DEGRADATION','SKIPPED_NON_SESSION','FAILED']
    success_count: int|None=None
    failure_count: int|None=None
    abstention_count: int|None=None
    duplicate_count: int|None=None
    provider_failure_count: int|None=None
    data_quality_failure_count: int|None=None
    captured_source_count: int|None=None
    failed_source_count: int|None=None
    pending_outcome_count: int|None=None
    error_code: str|None=None
    result_status: str|None=None

class JobPage(Contract):
    items: list[JobRecord]
    total: int
    offset: int=0
    limit: int=100

class OperatorSession(Contract):
    authenticated: bool
    mode: Literal['LOCAL_LOOPBACK','BEARER','UNAVAILABLE']
    expires_at: AwareDatetime|None=None

class OperatorStatus(Contract):
    backend_available: Literal[True]=True
    execution_mode: Literal['LOCAL_PC_MANUAL']='LOCAL_PC_MANUAL'
    cloud_independence: Literal['DEFERRED']='DEFERRED'
    running_job: JobRecord|None=None
    timeout_seconds: int
