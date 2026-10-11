"""Read-only Insights facade; archived research modules remain unchanged."""
from core.assistant import detect_intent,extract_ticker,SUGGESTED_QUESTIONS
from research.post_v5.assistant import answer_question as descriptive_answer
from application.services.prospective import ProspectiveService,ServiceError
def answer_question(question,df,default_ticker,currency_sym="$",news_items=None):
    intent=detect_intent(question)
    if intent!="forecast":return descriptive_answer(question,df,default_ticker,currency_sym,news_items)
    ticker=extract_ticker(question,default_ticker) or default_ticker
    try:
        result=ProspectiveService().forecast(ticker,"canonical")
        horizon=next(h for h in result.horizons if h.horizon==10)
        answer=(f"RESEARCH MODEL: saved canonical 10-session estimate for {ticker} at {result.information_cutoff.isoformat()} "
            f"is {horizon.predicted_return*100:+.1f}%, with implied origin-based price {currency_sym}{horizon.predicted_price:,.2f}. "
            "LOW confidence; ABSTAINED. No actionable recommendation is allowed. No new forecast was generated.")
    except ServiceError:
        answer=f"No saved canonical forecast is available for {ticker}. No new forecast was generated. Collection runs independently; research promotion remains blocked."
    return {"answer":answer,"intent":intent,"ticker":ticker}
