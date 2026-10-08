from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

def test_new_navigation_dashboard_and_progress_board():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=30).run()
    assert not app.exception,[e.message for e in app.exception]
    text=' '.join(e.value for e in app.markdown)
    assert 'master task board' in text.lower() or 'Current task:' in text

def test_validate_page_reads_real_persisted_evidence():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'pages/validate.py'),default_timeout=30).run()
    assert not app.exception,[e.message for e in app.exception]
    assert len(app.dataframe)>0

def test_ui_formatting_does_not_turn_missing_values_into_zero():
    from utils.ui_v5 import percent,price
    assert percent(None)=='Unavailable'
    assert percent(float('nan'))=='Unavailable'
    assert percent(.025)=='+2.5%'
    assert price(None)=='Unavailable'
