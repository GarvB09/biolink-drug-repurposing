# Shared paths and BC5CDR / Hetionet loading utilities used by every pipeline stage.
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
APP_DIR = ROOT / "app"

BC5CDR_DIR = RAW_DIR / "bc5cdr" / "CDR_Data" / "CDR.Corpus.v010516"
BC5CDR_FILES = {
    "train": BC5CDR_DIR / "CDR_TrainingSet.PubTator.txt",
    "dev": BC5CDR_DIR / "CDR_DevelopmentSet.PubTator.txt",
    "test": BC5CDR_DIR / "CDR_TestSet.PubTator.txt",
}
HETIONET_NODES = RAW_DIR / "hetionet-v1.0-nodes.tsv"
HETIONET_EDGES = RAW_DIR / "hetionet-v1.0-edges.sif.gz"

PUBMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
BIOBERT = "dmis-lab/biobert-base-cased-v1.2"
BERT_BASE = "bert-base-cased"

NER_LABELS = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]

for d in (PROC_DIR, MODEL_DIR, RESULTS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj, indent=2):
    def default(o):  # numpy scalars / arrays from seqeval, sklearn, torch
        if hasattr(o, "tolist"):
            return o.tolist()
        raise TypeError(f"{type(o).__name__} is not JSON serializable")
    Path(path).write_text(json.dumps(obj, indent=indent, ensure_ascii=False, default=default), encoding="utf-8")


# --------------------------------------------------------------------------- BC5CDR

def parse_pubtator(path):
    """Parse a PubTator file into documents with entity mentions and CID relations.

    Each doc: {pmid, title, abstract, text, entities:[{start,end,text,type,ids}], relations:[(chem_id, dis_id)]}
    `text` is title + " " + abstract, which is the coordinate system PubTator offsets use.
    """
    docs = []
    for block in Path(path).read_text(encoding="utf-8").strip().split("\n\n"):
        lines = block.strip().split("\n")
        pmid, _, title = lines[0].partition("|t|")
        _, _, abstract = lines[1].partition("|a|")
        doc = {"pmid": pmid, "title": title, "abstract": abstract,
               "text": title + " " + abstract, "entities": [], "relations": []}
        for line in lines[2:]:
            cols = line.split("\t")
            if len(cols) >= 6 and cols[1].isdigit():
                ids = [i for i in cols[5].split("|") if i != "-1"]
                doc["entities"].append({"start": int(cols[1]), "end": int(cols[2]), "text": cols[3],
                                        "type": cols[4], "ids": ids})
            elif len(cols) == 4 and cols[1] == "CID":
                doc["relations"].append((cols[2], cols[3]))
        docs.append(doc)
    return docs


_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
# sentence boundary: . ! ? followed by whitespace and an uppercase letter, digit or bracket
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def tokenize(text, offset=0):
    return [(m.group(0), m.start() + offset, m.end() + offset) for m in _TOKEN_RE.finditer(text)]


def split_sentences(doc):
    """Return [(start, end)] sentence spans over doc["text"]; never cuts through an entity."""
    text = doc["text"]
    title_end = len(doc["title"])
    spans = [(0, title_end)]
    pos = title_end + 1
    abstract = text[pos:]
    starts = [0] + [m.end() for m in _SENT_RE.finditer(abstract)]
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(abstract)
        seg = abstract[s:e].rstrip()
        if seg:
            spans.append((pos + s, pos + s + len(seg)))
    # merge sentences that would split an entity mention
    merged = []
    for s, e in spans:
        if merged and any(ent["start"] < s < ent["end"] for ent in doc["entities"]):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def bio_tags(tokens, entities):
    tags = ["O"] * len(tokens)
    for ent in entities:
        first = True
        for i, (_, ts, te) in enumerate(tokens):
            if ts < ent["end"] and te > ent["start"]:
                tags[i] = ("B-" if first else "I-") + ent["type"]
                first = False
    return tags


def tags_to_spans(tags):
    """IOB2 tags -> [(type, start_tok, end_tok_exclusive)]."""
    spans, cur = [], None
    for i, t in enumerate(tags + ["O"]):
        if t.startswith("B-") or t == "O" or (t.startswith("I-") and (cur is None or cur[0] != t[2:])):
            if cur:
                spans.append((cur[0], cur[1], i))
                cur = None
            if t.startswith("B-") or t.startswith("I-"):
                cur = (t[2:], i)
    return spans
