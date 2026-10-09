import pandas as pd

def test_insights_forecast_uses_shadow_and_never_frozen_writer(monkeypatch):
    import core.assistant
    import research.post_v5.shadow
    from research.post_v5.assistant import answer_question
    def prohibited(*args,**kwargs):raise AssertionError('Frozen writer must not handle new forecast')
    monkeypatch.setattr(core.assistant,'answer_question',prohibited)
    monkeypatch.setattr(research.post_v5.shadow,'forecast_request',lambda ticker:{'forecasts':[{'forecast':{'horizon':10,'central_return':.02,'central_price':102.,'quantile_prices':[90.,95.,102.,105.,110.],'trust':{'status':'NO STRONG STATISTICAL EDGE'}}}]})
    result=answer_question('What is the 10-day forecast?',pd.DataFrame(),'ABC.NS')
    assert result['intent']=='forecast'
    assert 'RESEARCH MODEL' in result['answer']
    assert '102.00' in result['answer']

def test_other_assistant_answers_keep_original_computation():
    from research.post_v5.assistant import answer_question
    frame=pd.DataFrame({'Close':[99.,100.]})
    result=answer_question('What is the current price?',frame,'ABC.NS')
    assert result['intent']=='price'
    assert '100.00' in result['answer']


def test_buy_question_returns_measured_indicator_context_without_actionable_claim(candles):
    from research.post_v5.assistant import answer_question
    result=answer_question('Is this a good time to buy?',candles,'ABC.NS')
    assert result['intent']=='signal'
    assert 'bullish' in result['answer'] and 'bearish' in result['answer']
    assert 'abstain' in result['answer']
    assert 'BUY' not in result['answer'] and 'SELL' not in result['answer']
