# Consolidates the real pipeline outputs (data/*.json) into a single JS data file
# the static web app can load without needing a server or hitting CORS on file://.
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
APP_DIR = Path(__file__).resolve().parent.parent / "app"

kg = json.loads((DATA_DIR / "kg.json").read_text())
lit = json.loads((DATA_DIR / "relations_literature.json").read_text())
linkpred = json.loads((DATA_DIR / "linkpred_results.json").read_text())
corpus = json.loads((DATA_DIR / "corpus.json").read_text())

bundle = {"kg": kg, "literature": lit, "linkpred": linkpred, "corpus": corpus}

APP_DIR.mkdir(exist_ok=True)
(APP_DIR / "data.js").write_text("const BIOLINK_DATA = " + json.dumps(bundle) + ";\n")
print(f"Wrote app/data.js ({(APP_DIR / 'data.js').stat().st_size} bytes)")
