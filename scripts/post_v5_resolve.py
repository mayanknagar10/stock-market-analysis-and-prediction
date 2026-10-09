"""Resolve only truly matured prospective shadow predictions from a new capture."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.post_v5.outcomes import resolve_run
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
args=parser.parse_args();print(resolve_run(args.run_id))
