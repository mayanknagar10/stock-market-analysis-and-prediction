from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_prospective_view_retains_failed_capture_and_separates_evidence():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=40).run()
    app.switch_page('pages/research_results.py').run()
    assert not app.exception,[e.message for e in app.exception]
    text=' '.join(str(e.value) for e in app.markdown)+ ' '.join(str(e.value) for e in app.caption)
    assert 'Prospective' in text
    assert 'reconstructed' in text.lower()
    assert any('Production' in str(e.value) or 'production' in str(e.value) for e in app.warning)
