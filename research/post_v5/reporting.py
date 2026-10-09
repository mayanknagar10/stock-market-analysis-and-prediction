"""Hide untouched-window predictions and boundary-overlapping warm-up outcomes."""
import copy
import pandas as pd

def visible_shadow_records(records,boundary):
    cutoff=pd.Timestamp(boundary)
    visible=[];sealed=0
    for record in records:
        if pd.Timestamp(record['forecast_as_of'])>=cutoff:
            sealed+=1;continue
        value=copy.deepcopy(record)
        outcome=value.get('outcome')
        if outcome and outcome.get('maturity_timestamp') and pd.Timestamp(outcome['maturity_timestamp'])>=cutoff:
            value['outcome']=None;value['outcome_visibility']='sealed_boundary_overlap'
        visible.append(value)
    return visible,sealed
