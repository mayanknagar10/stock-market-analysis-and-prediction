"""Generate TypeScript source-of-truth without manually duplicating interfaces."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from api.main import app
def main():
    destination=Path(__file__).resolve().parents[1]/"docs/api/openapi.json"
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(app.openapi(),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("Generated docs/api/openapi.json")
if __name__=="__main__":main()
