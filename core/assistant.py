"""
Rule-Based Assistant — zero external accounts, zero API keys.

This is explicitly NOT an LLM chat assistant — it's honest about that.
No OpenAI/Anthropic/any-LLM API key is used. Instead, it pattern-matches
a small set of common question types (via regex/keyword intent
detection) and answers using REAL data already computed elsewhere in
this app (indicators, risk metrics, forecasts) — formatted as a
natural-sounding sentence.

The tradeoff, stated plainly: it can only answer the question types it
recognizes, phrased close to how it expects. It won't handle open-ended
free-form questions the way a real LLM would. What it DOES give you for
free: instant answers, zero hallucination risk (every number comes
directly from the same computation the rest of the app already trusts),
and zero ongoing API cost.

When you're ready to add a real LLM (OpenAI/Anthropic key), the
INTENT_HANDLERS structure here maps directly onto "function calling" /
tool-use patterns most LLM APIs support — this becomes the tool
definitions, and the LLM takes over phrasing and open-ended questions.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable, List
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
# INTENT DETECTION — keyword/regex based, not ML
# ─────────────────────────────────────────────────────────────────

_TICKER_PATTERN = re.compile(r'\b([A-Z]{1,10}(?:\.[A-Z]{2})?)\b')

# Common English words that happen to be all-caps-compatible but are
# never real tickers — kept as a safety net, though the primary defense
# is only scanning the ORIGINAL (not uppercased) text for already-caps
# tokens, since people naturally type tickers in caps ("AAPL") but type
# regular words in lowercase ("what", "is", "good") even in a casual
# question. That one design choice avoids the whole class of false
# positives like "WHAT" or "TIME" being mistaken for a ticker.
_STOPWORDS = {"IS", "IT", "A", "I", "THE", "OF", "ON", "IN", "TO", "FOR",
             "AND", "OR", "BUY", "SELL", "RSI", "MACD", "PE", "CAGR",
             "OK", "USD", "INR", "CEO", "CFO", "IPO", "ETF", "USA"}


def extract_ticker(question: str, default_ticker: Optional[str] = None) -> Optional[str]:
    """
    Looks for an explicit ticker in the question. Primary strategy: scan
    the ORIGINAL text (not uppercased) for tokens that are ALREADY in
    all-caps — e.g. "What's AAPL doing?" naturally has AAPL capitalized
    while the rest of the sentence is normal case. This avoids false
    positives like "WHAT" or "TIME" matching after uppercasing the whole
    question, which a naive case-insensitive scan would produce.

    Edge case handled explicitly: if the ENTIRE question is typed in
    caps ("WHAT IS THE RSI"), the case-based heuristic can't distinguish
    a real ticker from a regular word — every word looks the same. In
    that situation we deliberately give up and fall back to
    default_ticker rather than guessing wrong.

    Falls back to default_ticker (typically "whatever ticker is
    currently loaded on this page") if no clear all-caps token is found.
    """
    letters = [c for c in question if c.isalpha()]
    if letters:
        upper_fraction = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_fraction > 0.8:
            # Whole question is "shouting" — case gives no signal, don't guess
            return default_ticker

    raw_candidates = _TICKER_PATTERN.findall(question)
    for c in raw_candidates:
        if c.upper() not in _STOPWORDS and len(c) >= 2 and c.isupper():
            return c

    return default_ticker


def detect_intent(question: str) -> str:
    """Returns one of the keys in INTENT_HANDLERS, or 'unknown'."""
    q = question.lower()
    if any(w in q for w in ["rsi", "overbought", "oversold"]):
        return "rsi"
    if any(w in q for w in ["macd"]):
        return "macd"
    if any(w in q for w in ["signal", "should i buy", "good time to buy",
                            "time to sell", "worth buying", "recommendation",
                            "buy or sell", "good buy", "good investment"]):
        return "signal"
    if any(w in q for w in ["risk", "volatil", "var ", "value at risk", "sharpe", "drawdown"]):
        return "risk"
    if any(w in q for w in ["forecast", "predict", "future price", "where will", "target price"]):
        return "forecast"
    if any(w in q for w in ["sentiment", "news"]):
        return "sentiment"
    if any(w in q for w in ["price", "trading at", "current price", "quote", "worth"]):
        return "price"
    return "unknown"


# ─────────────────────────────────────────────────────────────────
# INTENT HANDLERS — each returns a plain-English answer string
# ─────────────────────────────────────────────────────────────────

def _handle_price(ticker: str, df: pd.DataFrame, currency_sym: str = "$", **kwargs) -> str:
    if df.empty:
        return f"I don't have price data loaded for {ticker} right now."
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
    chg_pct = (last - prev) / prev * 100 if prev else 0
    direction = "up" if chg_pct >= 0 else "down"
    return (f"{ticker} is currently trading at {currency_sym}{last:,.2f}, "
           f"{direction} {abs(chg_pct):.2f}% from the previous close.")


def _handle_rsi(ticker: str, df: pd.DataFrame, **kwargs) -> str:
    from core.indicators import rsi
    if df.empty or len(df) < 15:
        return f"Not enough price history for {ticker} to compute RSI."
    rsi_val = float(rsi(df["Close"]).iloc[-1])
    if rsi_val > 70:
        zone = "overbought territory — historically associated with pullback risk"
    elif rsi_val < 30:
        zone = "oversold territory — historically associated with bounce potential"
    else:
        zone = "a neutral zone"
    return f"{ticker}'s 14-day RSI is {rsi_val:.1f}, which is in {zone}."


def _handle_macd(ticker: str, df: pd.DataFrame, **kwargs) -> str:
    from core.indicators import macd
    if df.empty or len(df) < 35:
        return f"Not enough price history for {ticker} to compute MACD."
    m = macd(df["Close"])
    hist = float(m["Hist"].iloc[-1])
    hist_prev = float(m["Hist"].iloc[-2])
    if hist > 0 and hist_prev <= 0:
        note = "just crossed bullish — the MACD line moved above the signal line"
    elif hist < 0 and hist_prev >= 0:
        note = "just crossed bearish — the MACD line moved below the signal line"
    elif hist > 0:
        note = "positive, indicating bullish momentum"
    else:
        note = "negative, indicating bearish momentum"
    return f"{ticker}'s MACD histogram is {hist:+.3f}, which is {note}."


def _handle_signal(ticker: str, df: pd.DataFrame, **kwargs) -> str:
    from core.indicators import generate_signals
    if df.empty or len(df) < 60:
        return f"Not enough price history for {ticker} to generate a composite signal."
    sig = generate_signals(df)
    comp = sig["composite"]
    return (f"The composite technical signal for {ticker} is **{comp}** "
           f"({sig['buy_count']} bullish vs {sig['sell_count']} bearish indicators out of 8). "
           f"This is based on RSI, MACD, Bollinger Bands, moving averages, Stochastic, "
           f"ADX, volume, and CCI — not investment advice, just a technical summary.")


def _handle_risk(ticker: str, df: pd.DataFrame, currency_sym: str = "$", **kwargs) -> str:
    from core.risk_metrics import full_risk_report
    if df.empty or len(df) < 30:
        return f"Not enough price history for {ticker} to compute risk metrics."
    report = full_risk_report(df["Close"])
    vol = report["annualised_volatility"] * 100
    sharpe = report["sharpe_ratio"]
    max_dd = report["max_drawdown"] * 100
    var95 = report["var_95_historical"] * 100
    risk_level = "high" if vol > 40 else ("moderate" if vol > 20 else "low")
    return (f"{ticker} has {risk_level} risk: {vol:.1f}% annualised volatility, "
           f"a Sharpe ratio of {sharpe:.2f}, and has drawn down as much as {max_dd:.1f}% "
           f"from its peak. Daily 95% Value-at-Risk is {var95:.2f}%.")


def _handle_forecast(ticker: str, df: pd.DataFrame, currency_sym: str = "$", **kwargs) -> str:
    from core.models import forecast_future
    if df.empty or len(df) < 80:
        return f"Not enough price history for {ticker} to generate a forecast."
    try:
        result = forecast_future(df, horizon=10, n_paths=20)
    except Exception as e:
        return f"I couldn't generate a forecast for {ticker} right now ({e})."
    fc = result["forecast"]
    last_p = float(df["Close"].iloc[-1])
    f_end = float(fc["Forecast"].iloc[-1])
    chg = (f_end / last_p - 1) * 100
    lo = float(fc["Lower_80"].iloc[-1])
    hi = float(fc["Upper_80"].iloc[-1])
    direction = "higher" if chg >= 0 else "lower"
    mode_note = "using the universal ML model" if result["mode"] == "universal" else "using a quick per-ticker model"
    return (f"The 10-day forecast for {ticker} ({mode_note}) projects "
           f"{currency_sym}{f_end:,.2f}, {abs(chg):.1f}% {direction} than today's "
           f"{currency_sym}{last_p:,.2f}. The 80% confidence range is "
           f"{currency_sym}{lo:,.2f} to {currency_sym}{hi:,.2f}. "
           f"This is a statistical projection, not a guarantee.")


def _handle_sentiment(ticker: str, news_items: Optional[List[Dict]] = None, **kwargs) -> str:
    from core.sentiment import analyze_news_sentiment
    if not news_items:
        return f"I don't have recent news loaded for {ticker} to analyze sentiment."
    report = analyze_news_sentiment(news_items)
    if report["n_items"] == 0:
        return f"No news headlines available for {ticker} right now."
    return (f"Recent news sentiment for {ticker} is **{report['overall_label']}** "
           f"(score: {report['overall_score']:+.2f}), based on {report['n_items']} headlines "
           f"({report['pos_count']} positive, {report['neg_count']} negative, "
           f"{report['neu_count']} neutral).")


def _handle_unknown(ticker: str, **kwargs) -> str:
    return (
        "I can answer questions about: current price, RSI, MACD, the overall "
        "technical signal, risk/volatility, price forecasts, and news sentiment. "
        "Try asking something like \"What's the RSI?\" or \"Is this a good time to buy?\""
    )


INTENT_HANDLERS: Dict[str, Callable] = {
    "price": _handle_price, "rsi": _handle_rsi, "macd": _handle_macd,
    "signal": _handle_signal, "risk": _handle_risk, "forecast": _handle_forecast,
    "sentiment": _handle_sentiment, "unknown": _handle_unknown,
}


def answer_question(question: str, df: pd.DataFrame, default_ticker: str,
                    currency_sym: str = "$", news_items: Optional[List[Dict]] = None) -> Dict:
    """
    Main entry point. Returns {"answer": str, "intent": str, "ticker": str}
    so the UI can show what was understood, not just the raw answer —
    being transparent about what was matched builds trust and helps
    users learn how to phrase follow-ups when a question isn't
    recognized.
    """
    ticker = extract_ticker(question, default_ticker) or default_ticker
    intent = detect_intent(question)
    handler = INTENT_HANDLERS.get(intent, _handle_unknown)

    try:
        answer = handler(ticker=ticker, df=df, currency_sym=currency_sym, news_items=news_items)
    except Exception as e:
        answer = f"I ran into an issue answering that: {e}"

    return {"answer": answer, "intent": intent, "ticker": ticker}


SUGGESTED_QUESTIONS = [
    "What's the current price?",
    "What's the RSI telling us?",
    "Is this a good time to buy?",
    "How risky is this stock?",
    "What's the 10-day forecast?",
    "What's the news sentiment?",
]
