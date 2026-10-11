import pandas as pd
def test_forecast_question_uses_saved_service_never_writer(monkeypatch):
    from application.services.assistant import answer_question
    from application.services.prospective import ProspectiveService,ServiceError
    import research.post_v5.shadow
    def forbidden(*a,**k):raise AssertionError("No prediction generation from Insights")
    monkeypatch.setattr(research.post_v5.shadow,"forecast_request",forbidden)
    def unavailable(*a,**k):raise ServiceError("MODEL_UNAVAILABLE",404)
    monkeypatch.setattr(ProspectiveService,"forecast",unavailable)
    result=answer_question("What is the 10-day forecast?",pd.DataFrame(),"ABB.NS")
    assert result["intent"]=="forecast" and "saved" in result["answer"].lower()
    assert "no new forecast" in result["answer"].lower()
def test_required_stale_context_rejected_optional_is_not():
    from application.collection.engine import required_context_issues
    assert required_context_issues(["market_ret_5d"],{"market_india":"STALE"})==["market_india"]
    assert required_context_issues(["rsi_14"],{"market_india":"STALE"})==[]
