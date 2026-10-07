"""Real legacy diskcache behavior in isolated temporary storage."""
import pytest
from core import cache_layer

@pytest.fixture
def cache(monkeypatch,tmp_path):
    monkeypatch.setattr(cache_layer,'CACHE_DIR',str(tmp_path/'cache'))
    monkeypatch.setattr(cache_layer,'_cache_instance',None)
    instance=cache_layer.get_cache()
    assert instance is not None
    yield instance
    instance.close()

def test_cache_reuses_value_until_ttl_then_refetches(cache,monkeypatch):
    instant=[1000.]
    monkeypatch.setattr(cache_layer.time,'time',lambda:instant[0])
    calls=[]
    def fetch(symbol):
        calls.append(symbol)
        return {'price':len(calls)}
    assert cache_layer.cached_call('quote',10,fetch,'TEST')['price']==1
    instant[0]=1009.
    assert cache_layer.cached_call('quote',10,fetch,'TEST')['price']==1
    instant[0]=1010.
    assert cache_layer.cached_call('quote',10,fetch,'TEST')['price']==2
    assert calls==['TEST','TEST']

def test_cache_keys_keep_different_symbols_separate(cache):
    def fetch(symbol): return symbol
    assert cache_layer.cached_call('quote',10,fetch,'A')=='A'
    assert cache_layer.cached_call('quote',10,fetch,'B')=='B'

def test_cache_missing_library_does_not_change_return_contract(monkeypatch):
    monkeypatch.setattr(cache_layer,'get_cache',lambda:None)
    assert cache_layer.cached_call('quote',10,lambda: {'status':'unavailable'})=={'status':'unavailable'}
