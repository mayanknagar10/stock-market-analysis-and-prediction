import pytest

def test_membership_retains_unavailable_securities_and_sector_identity():
    from research.post_v5.universe import parse_membership
    payload=b'Company Name,Industry,Symbol,Series,ISIN Code\nAlpha,IT,AAA,EQ,IN0000000001\nBeta,Bank,BBB,EQ,IN0000000002\n'
    rows=parse_membership(payload)
    assert [r['ticker'] for r in rows]==['AAA.NS','BBB.NS']
    assert rows[1]['industry']=='Bank'
    assert rows[1]['isin']=='IN0000000002'

def test_duplicate_or_missing_membership_identity_is_rejected():
    from research.post_v5.universe import parse_membership
    with pytest.raises(ValueError): parse_membership(b'Company Name,Industry,Symbol,Series,ISIN Code\nA,IT,AAA,EQ,ID\nB,IT,AAA,EQ,ID\n')
