"""Seal/verify V5 without rewriting original files."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.post_v5.preservation import seal,verify
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seal',action='store_true')
args=parser.parse_args()
file=ROOT/'reports/post_v5/V5_PRESERVATION.json'
print(seal(ROOT,file,'26c09e0') if args.seal else verify(ROOT,file))
