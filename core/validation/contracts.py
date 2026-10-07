"""Model identity and feature compatibility contracts; no unsupported production claims."""
from dataclasses import dataclass, asdict
import hashlib
import json
from core.validation.alignment import aware_index

def schema_hash(names):
    if not names or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):
        raise ValueError('Feature schema must have unique nonempty names')
    payload=json.dumps(list(names),ensure_ascii=True,separators=(',',':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

@dataclass(frozen=True)
class ModelMetadata:
    model_version: str
    family: str
    horizon: int
    feature_version: str
    feature_names: tuple[str,...]
    feature_schema_hash: str
    training_start: str
    training_cutoff: str
    created_at: str
    training_universe: tuple[str,...]
    parameters: dict
    validation_metrics: dict
    code_version: str
    data_snapshot_ids: tuple[str,...]
    point_in_time_verified: bool = False

    def __post_init__(self):
        if type(self.point_in_time_verified) is not bool:
            raise ValueError('point_in_time_verified must be an actual boolean')
        if 'final_test_passed' in self.validation_metrics and type(self.validation_metrics['final_test_passed']) is not bool:
            raise ValueError('final_test_passed must be an actual boolean')
        if self.family not in {'india_equity','us_equity'} or type(self.horizon) is not int or self.horizon not in {1,5,10,20}:
            raise ValueError('Unsupported equity model family/horizon')
        if not all([self.model_version,self.feature_version,self.training_universe,self.code_version,self.data_snapshot_ids]):
            raise ValueError('Incomplete model provenance')
        if schema_hash(self.feature_names)!=self.feature_schema_hash: raise ValueError('Feature schema hash mismatch')
        start,end,created=aware_index([self.training_start,self.training_cutoff,self.created_at],'training metadata')
        if start>=end or end>created: raise ValueError('Invalid training cutoff')

    def assert_compatible(self,feature_names,family,as_of):
        if tuple(feature_names)!=self.feature_names or schema_hash(feature_names)!=self.feature_schema_hash:
            raise ValueError('Model feature schema/order mismatch')
        if family!=self.family: raise ValueError('Model market family mismatch')
        cutoff=aware_index([as_of],'forecast as_of')[0]
        if aware_index([self.training_cutoff])[0]>=cutoff:
            raise ValueError('Model training reaches historical prediction cutoff')
        if aware_index([self.created_at])[0]>cutoff:
            raise ValueError('Model artifact did not exist at prediction cutoff')
        return True

    def assert_production_eligible(self):
        if self.point_in_time_verified is not True: raise ValueError('Model is not verified point-in-time')
        if self.validation_metrics.get('final_test_passed',False) is not True:
            raise ValueError('Model has no successful frozen final-test evidence')
        return True

    def to_dict(self): return asdict(self)
