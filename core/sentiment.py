"""
Sentiment & Alternative Data — zero external accounts, zero API keys.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for
sentiment scoring — a lexicon-based, fully OFFLINE Python package. No
API calls, no rate limits, no account. It's tuned for short, informal
text (headlines, social posts), which is exactly the kind of text
available from free sources: yfinance news headlines and SEC filing
excerpts.

This is a genuine tradeoff worth being explicit about: VADER is not as
accurate as a modern transformer-based sentiment model (e.g. FinBERT),
but those require either downloading a large model file (slow on
Streamlit Cloud's free tier) or an API call (Hugging Face Inference API
— needs a free account, which contradicts the no-signup requirement
here). VADER is the right choice for "good enough, instant, zero
dependencies beyond a pip install."
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
    _VADER_AVAILABLE = True
except ImportError:
    _analyzer = None
    _VADER_AVAILABLE = False


# Finance-specific lexicon additions — VADER's base dictionary is
# general-purpose (movie reviews, tweets) and misses common finance
# jargon. This nudges scores for domain terms without needing a full
# custom model.
_FINANCE_LEXICON_ADDITIONS = {
    "beat": 2.0, "beats": 2.0, "beating": 2.0,
    "outperform": 2.5, "outperforms": 2.5, "outperformed": 2.5, "outperforming": 2.5,
    "upgrade": 2.0, "upgrades": 2.0, "upgraded": 2.0, "upgrading": 2.0,
    "bullish": 2.5, "rally": 2.0, "rallies": 2.0, "rallied": 2.0, "rallying": 2.0,
    "surge": 2.5, "surges": 2.5, "surged": 2.5, "surging": 2.5,
    "soar": 2.5, "soars": 2.5, "soared": 2.5, "soaring": 2.5,
    "downgrade": -2.0, "downgrades": -2.0, "downgraded": -2.0, "downgrading": -2.0,
    "bearish": -2.5, "plunge": -2.5, "plunges": -2.5, "plunged": -2.5, "plunging": -2.5,
    "slump": -2.0, "slumps": -2.0, "slumped": -2.0, "slumping": -2.0,
    "miss": -1.8, "misses": -1.8, "missed": -1.8, "missing": -1.0,
    "cut": -1.5, "cuts": -1.5, "cutting": -1.5,
    "layoffs": -2.0, "layoff": -2.0,
    "bankruptcy": -3.0, "bankrupt": -3.0,
    "fraud": -3.0, "fraudulent": -3.0,
    "lawsuit": -1.8, "lawsuits": -1.8, "sued": -1.8, "sues": -1.8,
    "probe": -1.5, "probes": -1.5, "probed": -1.5,
    "investigation": -1.5, "investigations": -1.5, "investigated": -1.5,
    "recall": -1.8, "recalls": -1.8, "recalled": -1.8,
    "default": -2.5, "defaults": -2.5, "defaulted": -2.5,
    "restructuring": -1.0,
}

if _VADER_AVAILABLE:
    _analyzer.lexicon.update(_FINANCE_LEXICON_ADDITIONS)


def analyze_text_sentiment(text: str) -> Dict:
    """
    Returns VADER's 4 scores for a single piece of text:
      neg, neu, pos (proportions, sum to 1.0)
      compound (normalized -1 to +1 overall score — the one to use for ranking)
    """
    if not _VADER_AVAILABLE or not text or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return _analyzer.polarity_scores(text)


def classify_sentiment(compound_score: float) -> str:
    """VADER's own recommended thresholds for compound score classification."""
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    return "Neutral"


def analyze_news_sentiment(news_items: List[Dict]) -> Dict:
    """
    news_items: list of dicts from fetch_news() — each has at least
    'title'. Analyzes each headline and returns an aggregate report.
    """
    if not news_items:
        return {"overall_score": 0.0, "overall_label": "Neutral",
                "n_items": 0, "items": [], "pos_count": 0, "neg_count": 0, "neu_count": 0}

    results = []
    for item in news_items:
        title = item.get("title", "")
        if not title:
            continue
        scores = analyze_text_sentiment(title)
        label = classify_sentiment(scores["compound"])
        results.append({
            "title": title,
            "publisher": item.get("publisher", ""),
            "compound": scores["compound"],
            "label": label,
            "timestamp": item.get("providerPublishTime", 0),
        })

    if not results:
        return {"overall_score": 0.0, "overall_label": "Neutral",
                "n_items": 0, "items": [], "pos_count": 0, "neg_count": 0, "neu_count": 0}

    overall = float(np.mean([r["compound"] for r in results]))
    pos_count = sum(1 for r in results if r["label"] == "Positive")
    neg_count = sum(1 for r in results if r["label"] == "Negative")
    neu_count = sum(1 for r in results if r["label"] == "Neutral")

    return {
        "overall_score": round(overall, 4),
        "overall_label": classify_sentiment(overall),
        "n_items": len(results),
        "items": sorted(results, key=lambda r: -abs(r["compound"])),  # most extreme first
        "pos_count": pos_count, "neg_count": neg_count, "neu_count": neu_count,
    }


def analyze_filing_sentiment(filing_text: str, max_chars: int = 5000) -> Dict:
    """
    Sentiment on an SEC filing excerpt (e.g. the risk-factors section or
    MD&A). Filings are long — this analyzes sentence-by-sentence over the
    first max_chars and aggregates, which is far more informative than
    scoring the whole blob as one unit (VADER is tuned for
    sentence-length text, not multi-page documents).
    """
    if not filing_text or not filing_text.strip():
        return {"overall_score": 0.0, "overall_label": "Neutral", "n_sentences": 0}

    text = filing_text[:max_chars]
    # Crude sentence split — good enough for scoring purposes, avoids
    # pulling in a full NLP tokenizer dependency for this
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return {"overall_score": 0.0, "overall_label": "Neutral", "n_sentences": 0}

    scores = [analyze_text_sentiment(s)["compound"] for s in sentences]
    overall = float(np.mean(scores))

    return {
        "overall_score": round(overall, 4),
        "overall_label": classify_sentiment(overall),
        "n_sentences": len(sentences),
        "score_std": round(float(np.std(scores)), 4),
        "most_negative_sentence": sentences[int(np.argmin(scores))] if sentences else "",
        "most_positive_sentence": sentences[int(np.argmax(scores))] if sentences else "",
    }


def sentiment_time_series(news_items: List[Dict]) -> pd.DataFrame:
    """
    Groups news sentiment by day for a trend chart — "is sentiment
    improving or deteriorating over the available headline history."
    """
    if not news_items:
        return pd.DataFrame()

    rows = []
    for item in news_items:
        title = item.get("title", "")
        ts = item.get("providerPublishTime", 0)
        if not title or not ts:
            continue
        try:
            date = datetime.utcfromtimestamp(ts).date()
        except Exception:
            continue
        score = analyze_text_sentiment(title)["compound"]
        rows.append({"date": date, "score": score})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    daily = df.groupby("date")["score"].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.set_index("date").sort_index()
