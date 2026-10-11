We are changing the implementation priority for StockPro.

The immediate priority is now:

# BUILD THE COMPLETE PROFESSIONAL STOCKPRO FRONTEND FIRST

Do not spend this phase expanding prediction research, cloud scheduling, new NLP models, new data providers, or additional infrastructure unless something is strictly required for the frontend to function.

The reason for this priority change is that we want the finished product interface to become the blueprint for the remaining backend work. Once the product is visually and functionally complete, we will identify missing API/data capabilities and fill those gaps systematically.

Use the current backend and APIs wherever available.

---

# CURRENT STATE

Preserve the existing project state.

Important existing facts:

- Existing Python analytics/research system is authoritative.
- FastAPI backend already exists.
- OpenAPI contracts already exist.
- Current research models remain frozen.
- No successor model should be trained during this task.
- Production prediction remains blocked.
- Current forecasts may be LOW confidence / abstained.
- Prospective predictions and immutable research records must remain untouched.
- Existing Streamlit implementation remains available as an internal/research fallback.
- Next.js/TypeScript toolchain has already been prepared.
- Shared frontend shell/design work may already be partially implemented.

Do not destroy or rewrite existing working infrastructure.

If unfinished work from a previous task is currently present, preserve it and bring the repository to a safe consistent state before continuing.

---

# PRODUCT VISION

Build StockPro as a serious modern financial research platform.

Use products such as Tradytics and professional financial terminals as references for:

- information density
- breadth
- research workflows
- charts
- scanners
- dashboards
- market context
- discoverability

Do NOT copy Tradytics branding, layouts, source code or proprietary design.

StockPro should have its own product identity.

The goal is:

Tradytics-like breadth

+

TradingView-quality chart experience

+

quantitative research tools

+

portfolio/risk analytics

+

transparent StockPro forecasting

+

model validation and live evidence

=

a cohesive StockPro platform.

StockPro should ultimately feel more organized and easier to understand than feature-heavy competitor platforms.

---

# CORE UX PRINCIPLE

Use:

# SIMPLE FIRST — DEPTH ON DEMAND

Do not expose every advanced feature at once.

A normal user should quickly understand:

1. What is happening?
2. What does the data say about this stock?
3. What does StockPro currently forecast?
4. How risky is it?
5. Why does the model believe that?
6. Can this model currently be trusted?

Advanced users should be able to drill into:

- SHAP
- quantiles
- calibration
- OOD
- regimes
- factor exposure
- benchmark-relative returns
- sector-relative returns
- historical analogues
- model versions
- provenance
- detailed risk statistics

without cluttering the default experience.

---

# TECHNOLOGY

Frontend:

- Next.js
- React
- TypeScript
- App Router
- Tailwind CSS
- TanStack Query
- reusable accessible UI components
- TradingView Lightweight Charts for primary financial charting where appropriate
- suitable analytics/chart library for non-price visualizations

Backend:

- existing FastAPI
- existing Python services

Do NOT rewrite financial calculations in TypeScript.

Do NOT rewrite ML models in JavaScript.

Do NOT create a MERN backend.

Python remains authoritative for:

- market calculations
- technical indicators
- forecasts
- risk
- factors
- validation
- model metadata
- calibration
- OOD
- SHAP
- backtesting
- scenario calculations
- ledger
- sentiment
- data health

React displays and interacts with these results.

---

# PRIORITY FOR THIS TASK

Focus on the complete product interface.

The sequence should now be:

1. Design system
2. Application shell
3. Global search
4. Dashboard
5. Stock Workspace
6. Forecast
7. Validate
8. Markets
9. Screener
10. Portfolio
11. Watchlist
12. Compare
13. Risk
14. Research tools
15. News/events interface
16. Backtesting interface
17. Responsive/mobile refinement
18. Loading/error/empty/degraded states
19. Accessibility
20. Frontend tests
21. Production build
22. Documentation

Remaining backend gaps should be documented rather than blocking the overall UI.

---

# INFORMATION ARCHITECTURE

Avoid putting every tool directly in the main sidebar.

Use a clean hierarchy.

Primary navigation:

Dashboard

Markets

Stocks / Research

Forecast

Validate

Screener

Portfolio

Watchlist

More

Inside More or contextual navigation:

- Compare
- Risk
- Backtesting
- Factor Analysis
- News
- Model/Research tools

Admin/engineering information should not dominate normal product navigation.

---

# GLOBAL APPLICATION SHELL

Build a polished persistent shell.

Desktop:

Left sidebar
+
top/global command/search bar
+
main workspace
+
optional contextual panel

Mobile:

responsive drawer navigation
+
compact top bar
+
mobile-safe content

The layout must feel like a real web product, not a Streamlit page recreated in React.

---

# GLOBAL SEARCH

Build one powerful global search.

Users should be able to search:

- ticker
- company
- market

Support:

- debounce
- keyboard navigation
- recent searches
- loading
- empty state
- invalid ticker
- error state

Clicking an equity should enter the Stock Workspace.

Suggested canonical routes:

`/stocks/TCS.NS`

`/forecast/TCS.NS`

Use URL-addressable states.

---

# DASHBOARD

The Dashboard should answer:

# WHAT IS HAPPENING RIGHT NOW?

Use a strong visual hierarchy.

Suggested content:

## Market header

Nifty 50
Bank Nifty
Sensex
S&P 500
Nasdaq
India VIX
USD/INR

Only display values supported by available APIs.

## Market Pulse

- market trend
- breadth where available
- volatility state
- sector leadership
- global cues

## Sector Heatmap

Create a professional compact sector view.

## Watchlist

Show:

- ticker
- price
- daily move
- compact sparkline
- research forecast state where appropriate

## Portfolio snapshot

If data exists:

- current value
- P&L
- exposure
- risk summary

## Recent StockPro forecasts

Show:

- ticker
- horizon
- forecast
- confidence
- pending/matured
- abstention state

Never translate an abstention into BUY/SELL.

## Prospective evidence

Show something like:

Prospective universe
100 stocks

Forecasts recorded
XXX

Matured
XXX

Pending
XXX

Current model status
RESEARCH ONLY

## System health

Compactly show:

- backend
- market-data freshness
- model state
- latest successful collection

Do not make the engineering task board the primary Dashboard experience.

---

# STOCK WORKSPACE

This should become one of the strongest parts of StockPro.

When a user opens a company, keep them inside one cohesive workspace rather than forcing them through unrelated pages.

Example structure:

TCS
Tata Consultancy Services

Price
Daily move
Market
Sector
Data timestamp

Tabs:

Overview

Chart

Forecast

Technicals

Fundamentals

News

Factors

Risk

Prediction History

Use shared ticker context across these views.

---

# STOCK OVERVIEW

Create an executive summary.

Include where data supports it:

- price
- daily change
- market cap
- valuation
- 52-week range
- performance periods
- trend state
- volatility
- selected fundamentals
- selected technical state
- latest news
- forecast status
- risk status

Do not dump 25 indicators into the Overview.

Show only important information.

Allow deeper pages for detail.

---

# CHART WORKSPACE

Build a high-quality financial chart.

Use TradingView Lightweight Charts where appropriate.

Support:

- candlestick
- volume
- crosshair
- tooltip
- responsive resize
- timeframe selection

Suggested periods:

1M
3M
6M
1Y
3Y
5Y
MAX

Optional overlays:

- moving averages
- Bollinger Bands
- selected indicators

Do not enable all indicators by default.

Keep the initial chart clean.

---

# FORECAST — FLAGSHIP FEATURE

Forecast should be one of the most polished pages in the application.

Above the fold show:

Ticker
Company
Current price
Price timestamp
Market state

Then direct horizon forecasts.

Support current horizons such as:

1D
5D
10D
20D

Show only valid backend values.

Each horizon card can include:

- expected return
- implied target price
- lower range
- central estimate
- upper range
- calibrated probability if accepted
- confidence
- abstention

Do not manufacture precision.

Do not display ₹3,421.184983.

Use financially sensible formatting.

---

# RESEARCH STATUS

The current model is not production validated.

Therefore prominently but professionally display:

RESEARCH MODEL

or:

RESEARCH ONLY

Do NOT use alarming giant warning banners on every section.

Use a consistent research-status component.

When:

`abstain = true`

show something like:

NO VALIDATED STATISTICAL EDGE

Do NOT show:

BUY
SELL
STRONG BUY
STRONG SELL

unless a future backend state explicitly allows recommendations.

The frontend must never reinterpret backend semantics.

---

# FORECAST SUMMARY

The user should be able to understand within seconds:

Expected 5D result

Expected 10D result

Expected 20D result

Potential range

Model confidence

Model agreement

Market regime

Data quality

Research status

Abstention

This should be visually clean.

---

# FORECAST DRIVERS

Create a professional model explanation section.

Use genuine backend attribution such as SHAP.

Display:

Positive contributors

Negative contributors

Relative contribution magnitude

Possible categories:

Stock technicals
Sector
Market
Global
FX
Commodity
Volatility

Clearly state that model attribution is not proof of causation.

Do not generate fake explanations.

---

# SCENARIO LAB

Create a dedicated scenario interface.

This may live inside Forecast or as an advanced tool.

Allow supported inputs such as:

Nifty
Nasdaq
VIX
USD/INR
Crude
Sector

Show:

BASE

versus

SCENARIO

For example:

Base expected 20D return

Scenario expected 20D return

Probability change

Range change

Model-state change

Clearly label these as conditional simulations.

Not predictions guaranteed to occur.

---

# PREDICTION INVALIDATION

Where backend support exists, show:

What conditions would materially weaken this forecast?

For example:

- volatility increases
- sector reverses
- Nifty falls
- USD/INR changes
- crude moves

Do not invent invalidation rules in frontend code.

Use backend results.

---

# HISTORICAL ANALOGUES

Create an advanced analogue section.

Show:

- similar historical states
- matured sample count
- median outcome
- outcome distribution
- best
- worst
- similarity

Clearly mark small samples.

Only show matured outcomes.

---

# PREDICTION HISTORY

Make this transparent.

Show every recorded forecast.

Columns:

Origin

Horizon

Prediction

Confidence

Abstained

Model

Status

Actual

Error

Distinguish:

PENDING

from:

MATURED

Original forecast values must never change visually when the eventual outcome is added.

---

# VALIDATE — MAJOR DIFFERENTIATOR

This should be treated as a core StockPro product feature, not a developer page.

The user should be able to answer:

# DOES STOCKPRO ACTUALLY WORK?

Sections:

## Current model state

Candidate
Version
Research status
Training cutoff
Protocol

## Genuine prospective evidence

Recorded forecasts

Matured forecasts

Pending forecasts

Universe size

Industries/sectors represented

## Forecast quality

As sufficient data matures:

MAE
RMSE
Directional performance
Brier
Calibration
Interval coverage
Prediction bias

## Performance by horizon

1D

5D

10D

20D

## Performance breakdown

Stock

Sector

Industry

Regime

Confidence

Model version

## Historical research

Preserve V4/V5 retrospective results.

Clearly label them:

RETROSPECTIVE / RECONSTRUCTED

## Forward evidence

Clearly label:

PROSPECTIVE / RECORDED BEFORE OUTCOME

Never mix these datasets visually.

---

# FAILED MODEL RESULTS

Do not hide the failed V5 gate.

Show historical research candidly.

This is a trust feature.

Users should see that StockPro distinguishes:

research

from:

validated evidence.

---

# MARKETS

Build a professional market workspace.

Use hierarchy.

Sections may include:

India

US

Global

Sectors

FX

Commodities

Volatility

Show supported data such as:

Nifty

Bank Nifty

Sensex

S&P 500

Nasdaq

VIX

India VIX

USD/INR

Crude

Gold

Sector performance

Avoid displaying dozens of unrelated metric cards.

Use:

tables
heatmaps
charts
compact tickers

appropriately.

---

# SCREENER

Build a powerful table.

Possible filters where supported:

Market

Sector

Industry

Market cap

P/E

Beta

RSI

Momentum

Volatility

Factor scores

Research forecast state

Use:

sorting

filtering

column customization if reasonable

pagination or virtualization

CSV export if existing functionality supports it

Never imply research signals are investment recommendations.

---

# TECHNICAL ANALYSIS

Do not display 25 indicators as 25 disconnected cards.

Organize them.

## Trend

Moving averages

ADX

trend structure

## Momentum

RSI

MACD

Stochastic

Williams %R

CCI

## Volatility

ATR

Bollinger Bands

realized volatility

## Volume

volume trend

MFI

OBV

Provide:

Summary

then:

Advanced details

---

# FUNDAMENTALS

Organize around investor questions.

Valuation

Growth

Profitability

Financial health

Efficiency

Dividends

Where historical series exist, chart them.

Clearly identify unavailable data.

---

# FACTORS

Preserve Fama-French analysis.

Display:

Market beta

Size

Value

Profitability

Investment

Alpha

R²

T-statistics

Provide accessible methodology explanation without overwhelming the page.

---

# RISK

Create a professional risk workspace.

Show where supported:

Volatility

Beta

Maximum drawdown

VaR

CVaR

Sharpe

Sortino

Calmar

Return distribution

Drawdown chart

Monthly heatmap

Avoid presenting risk numbers without explanation.

---

# PORTFOLIO

Build this as a real workspace.

Show:

Total value

Total P&L

Daily move

Positions

Allocation

Sector exposure

Correlation

Risk

Portfolio return

Use clear tables and charts.

Do not implement broker execution.

---

# WATCHLIST

Create an efficient research watchlist.

Rows can contain:

Ticker

Price

Daily %

Sparkline

Target

Stop

Forecast state

Confidence

Data health

Alerts

Allow sorting/filtering where useful.

---

# COMPARE

Build multi-stock comparison.

Show:

Normalized performance

Fundamentals

Technicals

Risk

Drawdowns

Factors

Forecast/research state where supported

Use tables with clear winner/difference highlighting, but avoid gamified green/red overuse.

---

# BACKTESTING

Migrate the existing strategy backtester into a professional interface.

Support existing strategies.

Display:

CAGR

Sharpe

Sortino

Calmar

Maximum drawdown

Win rate

Profit factor

Trade count

Fees/slippage

Equity curve

Drawdown

Trade log

Clearly differentiate:

strategy backtest

from:

ML forecast validation.

---

# NEWS & EVENTS

Build the interface now using existing capabilities.

Sections:

Company news

Market news

Sentiment

Filings where supported

Corporate events

If advanced NLP is not implemented:

do NOT fake advanced AI analysis.

Build the UI so advanced NLP can be plugged in later.

---

# FUTURE SPECIALTY MODULES

Do NOT implement fake data for features we don't yet have.

Architect navigation/components so these can later be added:

Options

Options flow

Open interest

IV analytics

GEX

Futures

FII/DII

Institutional activity

Bulk/block deals

Advanced alerts

Geopolitical intelligence

Do not show empty fake product pages to ordinary users.

Document them as future modules.

---

# SIMPLE VS ADVANCED VIEW

Implement progressive disclosure throughout the application.

A user should not need to understand:

SHAP
OOD
Brier
factor beta

just to understand a forecast.

Where appropriate support:

Simple

Advanced

Simple shows:

Price

Trend

Forecast

Risk

Top reasons

Reliability

Advanced exposes:

quantiles

SHAP

calibration

OOD

regime

alpha

factors

provenance

model metadata

---

# DESIGN LANGUAGE

Aim for:

serious financial terminal

meets

high-quality modern SaaS

Avoid copying any specific competitor.

Visual characteristics:

- dark-first but excellent light theme
- restrained surfaces
- subtle borders
- excellent typography
- dense but breathable layout
- minimal gradients
- minimal shadows
- meaningful color
- compact tables
- high-quality charts
- consistent spacing

Do not make every section a giant rounded card.

Use layout hierarchy instead.

---

# COLOR SEMANTICS

Green:

positive measured value

Red:

negative measured value

Amber:

warning/degraded/research caution

Neutral:

ordinary state

Do not equate:

green = BUY

red = SELL

unless that semantic is explicitly supported.

Status must also have text/icon meaning so color is not the only signal.

---

# DARK MODE

Make dark mode feel intentionally designed.

Do not simply invert colors.

Pay attention to:

chart backgrounds

gridlines

table borders

hover states

tooltips

muted text

form controls

focus outlines

---

# LIGHT MODE

Light mode must be fully polished.

Do not treat it as secondary.

---

# TYPOGRAPHY

Use a professional compact type scale.

Create clear hierarchy for:

Ticker

Company

Price

Major metric

Section heading

Body

Metadata

Dense tables

Avoid giant marketing-site headings inside research pages.

---

# DATA FORMATTING

Standardize:

Currency

Percentages

Large numbers

Dates

Timestamps

Indian currency formatting where appropriate

US currency formatting where appropriate

Missing values

Example:

₹3,241.50

+2.41%

₹1.82T or ₹1.82 lakh crore only if product convention is explicitly chosen

Do not mix conventions randomly.

---

# LOADING UX

Use skeletons.

Keep layout stable.

Avoid blank screens.

Avoid constant full-page loading spinners.

For individual data sections, use local loading state.

---

# EMPTY STATES

Design empty states for:

No portfolio

Empty watchlist

No matured forecasts

No analogues

No news

No factor data

No forecast

Do not display broken blank cards.

---

# ERROR STATES

Handle:

Backend unavailable

Provider unavailable

Stale data

Invalid ticker

Model unavailable

Forecast abstained

Outcome pending

No fundamentals

No news

Show human-readable explanations.

Do not expose Python errors.

---

# DEGRADED DATA

Respect backend health information.

If the forecast input is degraded, display it.

Example:

DATA HEALTH

Price        Current

Sector       Current

FX           Delayed

News         Unavailable

Overall      DEGRADED

Do not make a degraded forecast visually identical to a healthy one.

---

# RESPONSIVE DESIGN

The entire product must work at approximately:

1440px

1280px

1024px

768px

390px

Test major workflows.

Mobile does not need to show every desktop column.

Use responsive prioritization.

Charts must remain useful.

---

# ACCESSIBILITY

Implement:

Keyboard navigation

Focus states

Semantic HTML

ARIA where needed

Contrast

Accessible dialogs

Accessible dropdowns

Accessible search

Textual chart summaries where practical

Do not rely on color alone.

---

# PERFORMANCE

Optimize after measuring.

Use:

TanStack Query caching

dynamic imports for heavy charts

route splitting

debounced search

request deduplication

pagination/virtualization for large tables

Do not request every API endpoint when Dashboard loads.

---

# API INTEGRATION STRATEGY

For every component:

API exists
→ connect it.

API partially exists
→ implement UI using supported fields and document missing data.

API does not exist
→ do NOT fabricate backend data.

Instead:

hide the feature from ordinary users

or use a controlled internal placeholder.

Maintain:

`docs/FRONTEND_BACKEND_GAPS.md`

For every missing capability record:

- page
- component
- desired data
- existing endpoint
- missing field
- proposed future endpoint
- priority

This file will become the roadmap for the next backend phase.

---

# API CLIENT

Keep one typed API layer.

Do not place arbitrary raw fetch logic throughout UI components.

Use the generated OpenAPI contract where practical.

Handle:

loading

error

retry

caching

cancellation

consistently.

---

# COMPONENT SYSTEM

Create reusable financial UI components.

Examples:

PriceDisplay

PercentageChange

MarketBadge

ResearchStatus

ForecastCard

ConfidenceBadge

AbstentionState

DataHealth

ModelHealth

MetricCard

MetricRow

FinancialTable

Sparkline

CandlestickChart

SectionHeader

TickerSearch

MarketTickerStrip

ForecastRange

QuantileDisplay

ValidationMetric

EvidenceStatus

EmptyState

ErrorState

Skeletons

Do not duplicate the same markup across pages.

---

# AUTHENTICATION

Do not compromise security to finish frontend work.

If current authentication cannot safely migrate yet:

preserve a controlled development mode

and document the auth gap.

Do not expose passwords or local user files.

A dedicated authentication migration can follow.

---

# STREAMLIT

Do not delete Streamlit in this task.

Streamlit remains:

internal/research/admin fallback.

Next.js becomes:

primary product interface.

Eventually we may retire Streamlit, but only after feature parity and verification.

---

# FRONTEND QA

Add tests for critical states:

Dashboard rendering

Search

Stock workspace

Forecast

RESEARCH ONLY

LOW confidence

Abstention

Pending prediction

Matured prediction

Degraded data

Backend error

Invalid ticker

Validate metrics

Responsive navigation

Dark/light theme

Critical invariant:

A research abstention must never become a BUY/SELL recommendation in the frontend.

---

# BROWSER QA

If browser tooling is available, inspect major pages visually.

Check:

desktop

tablet

mobile

dark mode

light mode

loading

errors

empty states

Do not claim browser QA if no browser environment exists.

---

# PRODUCTION BUILD

The frontend must finish with a successful production build.

No TypeScript errors.

No unresolved critical runtime errors.

No hard-coded local backend URL.

No dependencies on developer-specific filesystem paths.

---

# VERCEL READINESS

Prepare frontend for Vercel deployment.

Do NOT require that FastAPI also run on Vercel.

The frontend must support a configurable API URL.

Document required environment variables.

Example:

`NEXT_PUBLIC_API_BASE_URL`

Do not deploy unless explicitly authorized.

---

# DO NOT DO THESE THINGS

Do not:

- retrain models
- modify model thresholds
- weaken abstention
- change consumed V5 results
- fabricate missing data
- fabricate prediction accuracy
- add fake options analytics
- rewrite Python ML in JS
- copy Tradytics
- hide failed validation
- add BUY/SELL merely because competitors have it
- remove Streamlit before verification
- deploy without approval

---

# FRONTEND COMPLETION DEFINITION

This phase is complete when:

1. Professional Next.js/TypeScript product shell exists.
2. Dashboard is complete.
3. Stock Workspace is complete.
4. Forecast is complete.
5. Validate is complete.
6. Markets is complete.
7. Screener is complete.
8. Portfolio is complete.
9. Watchlist is complete.
10. Compare is complete.
11. Risk is complete.
12. Research/technical/fundamental/factor views are migrated.
13. Existing backtesting has a frontend.
14. Existing news/sentiment has a frontend.
15. Mobile/responsive layout works.
16. Dark/light themes work.
17. Loading/error/empty/degraded states exist.
18. Existing APIs are integrated.
19. Missing backend capabilities are documented.
20. Frontend tests pass.
21. Production build succeeds.
22. Current model/research state is unchanged.

---

# REQUIRED DOCUMENTS

Create/update:

`docs/FRONTEND_ARCHITECTURE.md`

`docs/UI_DESIGN_SYSTEM.md`

`docs/API_CONTRACT.md`

`docs/FRONTEND_BACKEND_GAPS.md`

`docs/STREAMLIT_MIGRATION.md`

`docs/FRONTEND_DEPLOYMENT.md`

`docs/WEB_PLATFORM_COMPLETION_REPORT.md`

---

# FINAL REPORT

At completion provide:

- branch
- commit
- frontend routes
- major components
- API endpoints integrated
- API gaps
- tests
- production build result
- browser QA result
- responsive QA
- accessibility status
- remaining Streamlit dependencies
- auth status
- Vercel readiness
- backend hosting assumptions
- known UI issues
- future feature opportunities

Also explicitly answer:

Did any prediction model change?

Did any calibration threshold change?

Did abstention logic change?

Was any successor model trained?

Did any historical result change?

Was production enabled?

For this task these should all remain NO.

---

# PRIORITY OVERRIDE

Frontend product quality is now the main development priority.

Do not allow optional cloud-scheduling work, additional research experimentation or future data-provider expansion to delay the core frontend.

If those areas are unfinished but non-blocking, document them and continue.

The objective is to first create the complete StockPro product experience.

Afterward we will use `FRONTEND_BACKEND_GAPS.md` to systematically connect the remaining backend/data/research pieces.

Begin by reviewing the frontend work already completed.

Preserve good existing implementation.

Then establish the final design system and application shell before implementing individual pages.

Proceed autonomously through the complete frontend phase.