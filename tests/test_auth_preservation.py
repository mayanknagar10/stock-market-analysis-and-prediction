"""Exercise original auth/personalization using a private temporary data directory."""
import importlib.util
import json
from pathlib import Path
import shutil

def load_original_module(root,name):
    directory=root/'core'
    directory.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[1]/'core'/(name+'.py')
    target=directory/(name+'.py')
    shutil.copyfile(source,target)
    spec=importlib.util.spec_from_file_location('isolated_'+name,target)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_auth_roundtrip_keeps_password_private_and_rejects_wrong_password(tmp_path):
    auth=load_original_module(tmp_path,'auth')
    ok,_=auth.create_user('sample_user','privatepass123','Sample')
    assert ok
    assert auth.verify_login('sample_user','incorrect') is None
    user=auth.verify_login('sample_user','privatepass123')
    assert user['name']=='Sample'
    assert 'password_hash' not in user
    stored=json.loads((tmp_path/'data/users.json').read_text(encoding='utf-8'))
    assert 'privatepass123' not in stored['sample_user']['password_hash']
    ok,_=auth.change_password('sample_user','privatepass123','replacement123')
    assert ok
    assert auth.verify_login('sample_user','privatepass123') is None
    assert auth.verify_login('sample_user','replacement123') is not None

def test_personalization_preserves_auth_and_guest_is_noop(tmp_path):
    auth=load_original_module(tmp_path,'auth')
    personalization=load_original_module(tmp_path,'personalization')
    auth.create_user('sample_user','privatepass123','Sample')
    personalization.track_view('sample_user','AAPL','Technology')
    personalization.track_view(None,'MSFT','Technology')
    assert personalization.get_recently_viewed_tickers('sample_user')==['AAPL']
    recommendations=personalization.recommend_similar_stocks('sample_user',
        [{'ticker':'AAPL','sector':'Technology'},{'ticker':'MSFT','sector':'Technology'}])
    assert [item['ticker'] for item in recommendations]==['MSFT']
    assert auth.verify_login('sample_user','privatepass123') is not None
