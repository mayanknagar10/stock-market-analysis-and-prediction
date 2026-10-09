"""Run independent append-only maturity resolution."""
import json
import logging
from application.collection.outcomes import Resolver
from application.collection.store import CollectionStore
def main():
    try:
        print(json.dumps(Resolver(CollectionStore(),include_legacy=True).run(),sort_keys=True,allow_nan=False));return 0
    except Exception:
        logging.exception("Outcome resolution job failed");return 2
if __name__=="__main__":raise SystemExit(main())
