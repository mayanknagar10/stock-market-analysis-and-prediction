"""Preserve existing insights answers; isolate forecast side effects from frozen V5."""
from core.assistant import detect_intent,extract_ticker,SUGGESTED_QUESTIONS

def answer_question(question,df,default_ticker,currency_sym='$',news_items=None):
    from core.assistant import answer_question as legacy_answer
    intent=detect_intent(question)
    if intent=='signal':
        ticker=extract_ticker(question,default_ticker) or default_ticker
        if df.empty or len(df)<60:return {'answer':'Not enough history for a measured technical summary. Research forecasts abstain.','intent':intent,'ticker':ticker}
        from core.indicators import generate_signals
        signals=generate_signals(df)
        answer=(f"Technical indicator context for {ticker}: {signals['buy_count']} bullish and {signals['sell_count']} bearish indicators. "
            "These are indicator counts, not a validated forecast edge. The research model continues to abstain; no actionable recommendation is supported.")
        return {'answer':answer,'intent':intent,'ticker':ticker}
    if intent!='forecast':return legacy_answer(question,df,default_ticker,currency_sym,news_items)
    ticker=extract_ticker(question,default_ticker) or default_ticker
    try:
        from research.post_v5.shadow import forecast_request
        result=forecast_request(ticker)
        forecast=next(item['forecast'] for item in result['forecasts'] if item['forecast']['horizon']==10)
        answer=(f"RESEARCH MODEL: the frozen V5 10-session estimate for {ticker} is {forecast['central_return']*100:+.1f}%, "
            f"implying {currency_sym}{forecast['central_price']:,.2f} under the total-return-equivalent assumption. "
            f"The 80% interval is {currency_sym}{forecast['quantile_prices'][0]:,.2f}–{currency_sym}{forecast['quantile_prices'][4]:,.2f}. "
            f"{forecast['trust']['status']}. Actual inputs/output are archived in the prospective shadow study; production remains blocked.")
    except Exception as error:answer=f'Research forecast unavailable for {ticker}: {error}. Failure is retained in the shadow archive; no fallback forecast is substituted.'
    return {'answer':answer,'intent':intent,'ticker':ticker}
