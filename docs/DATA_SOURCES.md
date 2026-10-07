# External data source inventory

Audited 07 October 2026 from implementation, including active and inactive modules, notebook calls and browser resources. Unknown delays/timezones/terms are explicitly unknown; this document does not adopt README marketing claims. Machine-readable inventory: [data_sources.json](data_sources.json). No paid provider is required. SMTP and Resend examples in notifications are documentation strings, not active requests.

## yahoo_prices

- **provider:** Yahoo Finance.
- **library:** yfinance.
- **module:** core/data_fetcher.py.
- **function:** _fetch_yahoo_ohlcv_uncached.
- **endpoint:** Yahoo chart endpoint via Ticker.history; exact hostname managed by installed yfinance.
- **fields:** Open, High, Low, Close, Volume.
- **timezone:** Provider exchange timezone removed using tz_localize(None).
- **expected delay:** Unverified/provider-dependent; never label real-time.
- **refresh frequency:** On cache miss / explicit rerun.
- **caching policy:** Streamlit 300s + disk cached_call yahoo_ohlcv 300s.
- **authentication:** No key.
- **failure behavior:** Empty DataFrame; errors suppressed.
- **fallback:** Stooq after failure; premium keys tried before Yahoo; CoinGecko tried first for mapped crypto.
- **point in time safety:** Historical prices retrieved at present are not archived as-of vintages.
- **used in prediction:** true.
- **retroactive changes:** Prices/corrections and total-return adjustments can revise history.
- **corporate actions:** auto_adjust=True; Dividends/Stock Splits discarded.
- **v5 status:** WRAP; preserve raw prices and actions; reject ambiguous fallback.

## yahoo_fundamentals

- **provider:** Yahoo Finance.
- **library:** yfinance.
- **module:** core/data_fetcher.py.
- **function:** _fetch_fundamentals_uncached.
- **endpoint:** Ticker.fast_info and Ticker.info.
- **fields:** market_cap, fifty_two_week_high, fifty_two_week_low, currency, three_month_average_volume, ten_day_average_volume, longName, shortName, sector, industry, trailingPE, forwardPE, trailingEps, dividendYield, beta, longBusinessSummary, website, fullTimeEmployees, exchange, totalRevenue, grossMargins, operatingMargins, returnOnEquity, debtToEquity, marketCap, fiftyTwoWeekHigh, fiftyTwoWeekLow, averageVolume, averageVolume10days.
- **timezone:** No publication timestamps retained.
- **expected delay:** Unknown; current snapshot.
- **refresh frequency:** On cache miss.
- **caching policy:** Streamlit 3600s + disk fundamentals 3600s.
- **authentication:** No key.
- **failure behavior:** Partially populated dict/defaults; errors suppressed.
- **fallback:** 52-week extrema/volume calculated from fetch_ohlcv.
- **point in time safety:** Not historical PIT; prohibit current fundamentals in backtests.
- **used in prediction:** false.
- **retroactive changes:** Latest values can change; no vintages.
- **corporate actions:** Metadata may change after splits; trailing EPS adjustments unspecified.
- **v5 status:** WRAP; current research only.

## yahoo_news

- **provider:** Yahoo Finance news aggregation.
- **library:** yfinance.
- **module:** core/data_fetcher.py.
- **function:** fetch_news.
- **endpoint:** Ticker.news.
- **fields:** title, publisher, link, providerPublishTime, thumbnail; pages/sentiment expect flat legacy schema.
- **timezone:** Unix seconds assumed UTC by consumers; publication/availability not enforced.
- **expected delay:** Unknown and headline coverage incomplete.
- **refresh frequency:** On cache miss.
- **caching policy:** Streamlit 1800s.
- **authentication:** No key.
- **failure behavior:** Empty list; schema drift can produce empty/neutral display.
- **fallback:** None.
- **point in time safety:** No archived ingestion/availability timestamps; prohibit historical feature reconstruction.
- **used in prediction:** false.
- **retroactive changes:** Headline edits/deletions and result set change.
- **corporate actions:** Event headlines only; not corporate-action source.
- **v5 status:** WRAP and normalize nested content, publication and first-seen.

## coingecko_quote

- **provider:** CoinGecko.
- **library:** requests.
- **module:** core/external_data.py.
- **function:** fetch_crypto_price.
- **endpoint:** https://api.coingecko.com/api/v3/simple/price.
- **fields:** usd, inr, usd_market_cap, usd_24h_change, usd_24h_vol.
- **timezone:** No timestamp requested/stored.
- **expected delay:** Unknown; rolling quote.
- **refresh frequency:** On cache miss.
- **caching policy:** Streamlit 120s.
- **authentication:** No key in code; entitlement not verified.
- **failure behavior:** {} on failure, 8s timeout.
- **fallback:** None.
- **point in time safety:** Current quote only.
- **used in prediction:** false.
- **retroactive changes:** Rolling values change.
- **corporate actions:** Not applicable.
- **v5 status:** KEEP research; separate crypto family.

## coingecko_history

- **provider:** CoinGecko.
- **library:** requests.
- **module:** core/external_data.py.
- **function:** fetch_crypto_history.
- **endpoint:** https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart.
- **fields:** prices[][ts_ms,Close].
- **timezone:** pd.to_datetime(ms) UTC-naive then normalized to date.
- **expected delay:** Unknown daily sampling time.
- **refresh frequency:** On cache miss.
- **caching policy:** Streamlit 300s.
- **authentication:** No key configured.
- **failure behavior:** Empty DataFrame, 8s timeout.
- **fallback:** Yahoo via fetch_ohlcv if mapped crypto fails.
- **point in time safety:** Historical close sampling may revise; timestamp lost; no as-of archive.
- **used in prediction:** true.
- **retroactive changes:** Possible historical corrections.
- **corporate actions:** Open=previous Close; High/Low artificially +/-0.2%; Volume=0; NOT authentic OHLCV.
- **v5 status:** PROHIBIT equity training and synthetic candle trust.

## coingecko_markets

- **provider:** CoinGecko.
- **library:** requests.
- **module:** core/external_data.py.
- **function:** fetch_crypto_top_movers.
- **endpoint:** https://api.coingecko.com/api/v3/coins/markets.
- **fields:** symbol, name, current_price, price_change_percentage_24h, market_cap, total_volume.
- **timezone:** Not retained.
- **expected delay:** Unknown.
- **refresh frequency:** On cache miss.
- **caching policy:** Streamlit 120s.
- **authentication:** No key configured.
- **failure behavior:** Empty DataFrame, 8s timeout.
- **fallback:** None.
- **point in time safety:** Current market-cap universe; not historical.
- **used in prediction:** false.
- **retroactive changes:** Rolling stats/universe change.
- **corporate actions:** Not applicable.
- **v5 status:** KEEP research.

## frankfurter_latest

- **provider:** Frankfurter/ECB-sourced according to module.
- **library:** requests.
- **module:** core/external_data.py.
- **function:** fetch_fx_rate.
- **endpoint:** https://api.frankfurter.app/latest.
- **fields:** rates[quote].
- **timezone:** Response date discarded.
- **expected delay:** Reference-rate delay unknown; not live quote.
- **refresh frequency:** On cache miss.
- **caching policy:** Streamlit 600s.
- **authentication:** No key.
- **failure behavior:** None, 8s timeout.
- **fallback:** None.
- **point in time safety:** Not reconstructible as-of from latest.
- **used in prediction:** false.
- **retroactive changes:** Updates/corrections possible.
- **corporate actions:** Not applicable.
- **v5 status:** WRAP; preserve response date and availability.

## frankfurter_history

- **provider:** Frankfurter.
- **library:** requests.
- **module:** core/external_data.py.
- **function:** fetch_fx_history.
- **endpoint:** https://api.frankfurter.app/{start}..{end}.
- **fields:** rates[date][quote].
- **timezone:** Naive date; range based on UTC date.
- **expected delay:** Publication time unrecorded.
- **refresh frequency:** On cache miss.
- **caching policy:** Streamlit 600s.
- **authentication:** No key.
- **failure behavior:** Empty Series, 8s timeout.
- **fallback:** None.
- **point in time safety:** Dated observations do not prove release-time availability.
- **used in prediction:** false.
- **retroactive changes:** Corrections possible.
- **corporate actions:** Not applicable.
- **v5 status:** WRAP; delayed conservative availability or actual vintage.

## worldbank_macro

- **provider:** World Bank.
- **library:** requests.
- **module:** core/external_data.py.
- **function:** fetch_macro_indicator.
- **endpoint:** https://api.worldbank.org/v2/country/{country}/indicator/{indicator}.
- **fields:** date(year), value; NY.GDP.MKTP.KD.ZG, FP.CPI.TOTL.ZG, FR.INR.RINR, SL.UEM.TOTL.ZS, NY.GDP.MKTP.CD.
- **timezone:** Annual observation year, no release timestamp.
- **expected delay:** Annual and often delayed; exact delay unknown.
- **refresh frequency:** On cache miss.
- **caching policy:** Streamlit 86400s.
- **authentication:** No key.
- **failure behavior:** Empty Series, 8s timeout.
- **fallback:** None.
- **point in time safety:** Annual observation year cannot be used as release date; current vintage unsafe historically.
- **used in prediction:** false.
- **retroactive changes:** Macro estimates revised retroactively.
- **corporate actions:** Not applicable.
- **v5 status:** KEEP research; prohibit historical forecast features without vintages.

## sec_tickers

- **provider:** SEC EDGAR.
- **library:** requests.
- **module:** core/external_data.py.
- **function:** fetch_sec_cik.
- **endpoint:** https://www.sec.gov/files/company_tickers.json.
- **fields:** ticker, cik_str.
- **timezone:** No timestamp retained.
- **expected delay:** Unknown.
- **refresh frequency:** On cache miss.
- **caching policy:** Streamlit 3600s.
- **authentication:** No API key; code hardcodes research@example.com in User-Agent, owner contact must replace.
- **failure behavior:** None, 8s timeout.
- **fallback:** None.
- **point in time safety:** Current symbol/CIK mapping can change; archive mapping for historical tests.
- **used in prediction:** false.
- **retroactive changes:** Symbol/entity mapping changes.
- **corporate actions:** Corporate reorganization changes CIK/mapping.
- **v5 status:** WRAP; configurable real contact.

## sec_submissions

- **provider:** SEC EDGAR.
- **library:** requests.
- **module:** core/external_data.py.
- **function:** fetch_sec_filings.
- **endpoint:** https://data.sec.gov/submissions/CIK{cik}.json.
- **fields:** form, filingDate, primaryDocDescription, primaryDocument, accessionNumber.
- **timezone:** Date strings retained; acceptanceDateTime not consumed.
- **expected delay:** Posting delay unknown.
- **refresh frequency:** On cache miss.
- **caching policy:** Streamlit 3600s.
- **authentication:** Same SEC User-Agent requirement.
- **failure behavior:** Empty DataFrame, 8s timeout.
- **fallback:** None.
- **point in time safety:** Filing date alone lacks intraday cutoff; store acceptance + first-seen.
- **used in prediction:** false.
- **retroactive changes:** Amendments are distinct accessions; metadata may update.
- **corporate actions:** No price adjustment source.
- **v5 status:** KEEP; implement acceptance semantics.

## sec_document_links

- **provider:** SEC EDGAR archives.
- **library:** Browser link; no server document download.
- **module:** core/external_data.py.
- **function:** fetch_sec_filings generates URL.
- **endpoint:** https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}.
- **fields:** Only user-opened document link; metadata returned, no body fetched.
- **timezone:** Not applicable.
- **expected delay:** Not applicable.
- **refresh frequency:** On user click.
- **caching policy:** Browser cache.
- **authentication:** SEC access policies apply.
- **failure behavior:** External browser failure.
- **fallback:** None.
- **point in time safety:** Accessions identify documents but app does not archive contents.
- **used in prediction:** false.
- **retroactive changes:** Amendments separate.
- **corporate actions:** Not applicable.
- **v5 status:** Document link only; do not claim automated XBRL/text ingestion.

## stooq_prices

- **provider:** Stooq.
- **library:** requests + pandas.read_csv.
- **module:** core/external_data.py.
- **function:** fetch_stooq_ohlcv.
- **endpoint:** https://stooq.com/q/d/l/?s={symbol}&i=d.
- **fields:** Open, High, Low, Close, Volume, Date.
- **timezone:** Naive Date; session close timezone unrecorded.
- **expected delay:** Daily delay unknown.
- **refresh frequency:** On cache miss.
- **caching policy:** Streamlit 300s.
- **authentication:** No key.
- **failure behavior:** Empty DataFrame, 8s timeout.
- **fallback:** Last free fallback in fetch_ohlcv.
- **point in time safety:** No vintage/availability data; ignores requested period and interval.
- **used in prediction:** true.
- **retroactive changes:** Adjustment/correction policy not recorded.
- **corporate actions:** Unverified; no actions retained; incompatible with verified total-return Yahoo semantics.
- **v5 status:** REJECT forecast fallback until contract verified; label research source.

## polygon_prices

- **provider:** Polygon configured optional provider.
- **library:** requests.
- **module:** core/premium_providers.py.
- **function:** fetch_polygon_ohlcv.
- **endpoint:** https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}.
- **fields:** t(ms), o, h, l, c, v.
- **timezone:** UTC-naive from Unix milliseconds.
- **expected delay:** Plan/provider-dependent; unverified.
- **refresh frequency:** On fetch_ohlcv cache miss.
- **caching policy:** Outer fetch_ohlcv 300s; no own cache.
- **authentication:** POLYGON_API_KEY env/Streamlit secrets; optional account, cost/entitlements not assumed.
- **failure behavior:** None, 8s timeout.
- **fallback:** Tiingo then free routing.
- **point in time safety:** No archival PIT guarantee in adapter.
- **used in prediction:** true.
- **retroactive changes:** Adjusted historical bars may revise.
- **corporate actions:** adjusted=true requested; adjustment scope not checked; actions discarded.
- **v5 status:** OPTIONAL; explicit compatible adapter only.

## tiingo_prices

- **provider:** Tiingo optional provider.
- **library:** requests.
- **module:** core/premium_providers.py.
- **function:** fetch_tiingo_ohlcv.
- **endpoint:** https://api.tiingo.com/tiingo/daily/{ticker}/prices.
- **fields:** date, open, high, low, close, volume.
- **timezone:** Provider timestamp made naive.
- **expected delay:** Daily/provider-dependent; unverified.
- **refresh frequency:** On fetch_ohlcv cache miss.
- **caching policy:** Outer fetch_ohlcv 300s.
- **authentication:** TIINGO_API_KEY env/Streamlit secrets; optional account.
- **failure behavior:** None, 8s timeout.
- **fallback:** Free routing.
- **point in time safety:** No PIT archive.
- **used in prediction:** true.
- **retroactive changes:** Historical corrections possible.
- **corporate actions:** Uses raw fields; adjClose/adjOpen/etc not consumed.
- **v5 status:** OPTIONAL; never silently mix raw and adjusted.

## fama_french

- **provider:** Kenneth French Data Library.
- **library:** pandas_datareader.
- **module:** core/factor_models.py.
- **function:** fetch_fama_french_factors.
- **endpoint:** DataReader(dataset, famafrench); library constructs Dartmouth factor ZIP URL.
- **fields:** Mkt-RF, SMB, HML, RMW, CMA, RF monthly percentages converted /100.
- **timezone:** Month index converted to month-start naive Timestamp, NOT publication time.
- **expected delay:** Monthly release delay unrecorded.
- **refresh frequency:** Every call; page result may remain in session_state.
- **caching policy:** No provider TTL/cache.
- **authentication:** No key.
- **failure behavior:** Empty DataFrame on missing dependency/fetch error.
- **fallback:** None.
- **point in time safety:** Current revision vintage, month-start incorrectly suggests early availability if used as features.
- **used in prediction:** false.
- **retroactive changes:** Revisions/reconstitution possible; dataset retrieval date absent.
- **corporate actions:** Not applicable.
- **v5 status:** KEEP exposures; snapshots/vintage gate for forecasts.

## clearbit_logo

- **provider:** Clearbit logo CDN.
- **library:** Browser img URL.
- **module:** core/data_fetcher.py.
- **function:** get_logo_url.
- **endpoint:** https://logo.clearbit.com/{domain}.
- **fields:** Logo image from static ticker-domain map / Yahoo website.
- **timezone:** Not applicable.
- **expected delay:** Not applicable.
- **refresh frequency:** Browser image load.
- **caching policy:** Browser cache.
- **authentication:** No key in code; current service availability not verified.
- **failure behavior:** Blank/broken logo; no data fallback.
- **fallback:** None.
- **point in time safety:** Not applicable.
- **used in prediction:** false.
- **retroactive changes:** Images change.
- **corporate actions:** Not applicable.
- **v5 status:** OPTIONAL presentation; remove if unavailable.

## google_fonts

- **provider:** Google Fonts.
- **library:** CSS @import in browser.
- **module:** utils/helpers.py.
- **function:** inject_css.
- **endpoint:** https://fonts.googleapis.com/css2?family=IBM+Plex+Mono...&family=IBM+Plex+Sans....
- **fields:** Font stylesheet/files.
- **timezone:** Not applicable.
- **expected delay:** Not applicable.
- **refresh frequency:** Browser stylesheet load.
- **caching policy:** Browser cache.
- **authentication:** No key.
- **failure behavior:** System/font fallbacks.
- **fallback:** Local/system fonts.
- **point in time safety:** Not applicable.
- **used in prediction:** false.
- **retroactive changes:** Font delivery changes.
- **corporate actions:** Not applicable.
- **v5 status:** OPTIONAL; prefer system/local fonts.

## flaticon_icon

- **provider:** Flaticon CDN.
- **library:** Browser Notification icon.
- **module:** core/notifications.py.
- **function:** browser_notify.
- **endpoint:** https://cdn-icons-png.flaticon.com/512/2830/2830284.png.
- **fields:** Notification image.
- **timezone:** Not applicable.
- **expected delay:** Not applicable.
- **refresh frequency:** When notification fires.
- **caching policy:** Browser cache.
- **authentication:** No key in code.
- **failure behavior:** Notification may omit icon.
- **fallback:** None.
- **point in time safety:** Not applicable.
- **used in prediction:** false.
- **retroactive changes:** Asset changes.
- **corporate actions:** Not applicable.
- **v5 status:** OPTIONAL presentation.

## legacy_notebook_yahoo

- **provider:** Yahoo Finance.
- **library:** yfinance.download.
- **module:** lit.ipynb.
- **function:** Code cell: yf.download(TATAMOTORS.NS, period=5y, interval=1d).
- **endpoint:** Library-managed Yahoo download endpoint.
- **fields:** OHLCV/adjusted price columns depend on notebook-era library defaults.
- **timezone:** Not normalized by active app.
- **expected delay:** Unverified.
- **refresh frequency:** Manual notebook execution.
- **caching policy:** No recorded snapshot.
- **authentication:** No key.
- **failure behavior:** Uncaught notebook exceptions.
- **fallback:** None.
- **point in time safety:** No reproducible vintage or frozen splits.
- **used in prediction:** false.
- **retroactive changes:** Historical prices can revise.
- **corporate actions:** Notebook adjustment default not explicit.
- **v5 status:** ARCHIVE only; not V5 forecast.

## Governance decisions

All V5 adapters must emit source_timestamp, available_at, fetched_at, source/provider ID, interval, adjustment policy, snapshot ID and quality flags. Never assign fetched_at as historical available_at without flagging reconstruction uncertainty. Persist original raw prices, adjustment factors and actions; adjusted labels and raw display prices must remain separate. Fallbacks must preserve interval, timezone, OHLCV integrity and adjustment semantics or fail closed. Freshness uses expected completed exchange sessions, not TTL alone. No current fundamentals/news/revised World Bank/French factors may enter historical features without archived availability/vintages. Yahoo is not verified as real-time or exchange-grade.
