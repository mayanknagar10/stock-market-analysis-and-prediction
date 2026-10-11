"""Deployment-neutral canonical capture/finalize command. No Streamlit runtime required."""
import argparse
import json
import logging
from application.collection.collector import Collector
from application.collection.store import CollectionStore
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["capture","finalize"],required=True)
    parser.add_argument("--session",help="NSE origin YYYY-MM-DD; defaults yesterday in Asia/Kolkata")
    args=parser.parse_args()
    try:
        result=Collector(CollectionStore()).run(args.session,args.stage)
        print(json.dumps(result,sort_keys=True,allow_nan=False))
        return 2 if result.get("health")=="FAILED" else 0
    except Exception:
        logging.exception("Collection job failed")
        return 2
if __name__=="__main__":raise SystemExit(main())
