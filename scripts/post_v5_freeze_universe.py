from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.post_v5.universe import freeze
universe=freeze(ROOT/'config/research/prospective-india-20261008-v1')
print('Frozen',universe['n_securities'],'securities in',len(universe['industries']),'industries at',universe['frozen_at'])
