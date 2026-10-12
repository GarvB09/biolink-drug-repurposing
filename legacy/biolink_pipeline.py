# %% [markdown]
# # BioLink — NLP for Drug Discovery Relationship Extraction
# ### End-to-end technical pipeline: NER → Relation Extraction → Knowledge Graph → Link Prediction (TransE / RotatE) → Evaluation
#
# This notebook runs the full BioLink pipeline **live** — every cell below executes real code
# against real data; nothing is a canned/mocked result.
#
# **One documented limitation:** this sandboxed environment's network egress allowlists PyPI/npm
# registries only. `huggingface.co` (the model hub used to download pretrained BioBERT /
# PubMedBERT checkpoints) and the S3 host used by scispaCy's biomedical models are unreachable
# here, so Stage 1 (NER) uses a transparent lexicon/pattern-matching extractor instead of a
# downloaded deep model, with the identical `{text, label, start, end}` output contract — see
# `scripts/ner.py` for the one-line swap back to `transformers.pipeline(...)` once network access
# to the model hub is available. Every other stage (relation extraction, knowledge graph
# construction, TransE/RotatE training, and evaluation) is fully real and network-independent.

# %%
import sys, json
from pathlib import Path
import pandas as pd

ROOT = Path.cwd() if (Path.cwd() / "scripts").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "data"))
DATA_DIR = ROOT / "data"

import ner, relations, linkpred  # noqa: E402

# %% [markdown]
# ## Stage 0 — Corpus
# Four real, verbatim PubMed/PMC abstracts (fetched live for this project — see citations in
# `data/corpus.json`), covering four literature-documented drug/disease topics.

# %%
corpus = json.loads((DATA_DIR / "corpus.json").read_text())
pd.DataFrame([{"pmid": d["pmid"], "title": d["title"]} for d in corpus])

# %% [markdown]
# ## Stage 1 — Named Entity Recognition

# %%
ner_results = ner.run_ner_on_corpus(corpus)
n_sents = sum(1 for d in corpus for _ in ner.split_sentences(d["abstract"]))
print(f"{len(ner_results)}/{n_sents} sentences contained a recognized entity.")
pd.DataFrame([
    {"pmid": r["pmid"], "sentence": r["sentence"][:90] + "…", "entities": [(e["text"], e["label"]) for e in r["entities"]]}
    for r in ner_results[:8]
])

# %% [markdown]
# ## Stage 2 — Relation Extraction (literature-derived) & Stage 3 — Knowledge Graph Construction
# Sentence-level DISEASE × CHEMICAL co-occurrence within the real abstracts, merged with a small
# curated supplementary edge set (well-known public drug–disease indications — see
# `data/curated_edges.py`) so the graph has enough density for embedding-based link prediction.

# %%
sys.path.insert(0, str(DATA_DIR))
from curated_edges import CURATED_TRIPLES  # noqa: E402

lit_triples = relations.extract_relations(ner_results)
nodes, edges = relations.build_graph(lit_triples, CURATED_TRIPLES)
(DATA_DIR / "relations_literature.json").write_text(json.dumps(lit_triples, indent=2))
(DATA_DIR / "kg.json").write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2))

n_lit = sum(1 for e in edges if e["source"] == "literature")
n_cur = sum(1 for e in edges if e["source"] == "curated")
print(f"Graph: {len(nodes)} nodes, {len(edges)} edges ({n_lit} literature-extracted, {n_cur} curated).")
pd.DataFrame(lit_triples)[["head", "relation", "tail", "pmid"]]

# %% [markdown]
# ## Stage 4 — Knowledge Graph Embedding & Link Prediction
# TransE and RotatE, implemented from scratch in PyTorch and trained live on the graph above.

# %%
results = linkpred.main()

# %% [markdown]
# ## Evaluation summary
# Standard knowledge-graph-completion metrics on a held-out triple split (filtered ranking).

# %%
pd.DataFrame(results["metrics"]).T

# %% [markdown]
# ## Ranked novel candidates (RotatE) for example diseases
# These are the outputs of the trained link-prediction model scoring every drug in the graph
# against each disease and ranking by predicted plausibility, excluding already-known edges.
#
# **Read these as a proof-of-concept, not a clinical recommendation.** The demo graph has ~100
# edges; a production BioLink deployment would train the same TransE/RotatE code on a
# literature-scale graph (hundreds of thousands of extracted triples), where embeddings capture
# far richer structure and predictions are correspondingly more coherent.

# %%
for disease, preds in results["predictions"]["RotatE"].items():
    print(f"\n{disease}:")
    for p in preds:
        print(f"   {p['drug']:<20s} score={p['score']}")

# %% [markdown]
# ## Conclusion
# All five stages described in the BioLink proposal — NER, relation extraction, knowledge graph
# construction, TransE/RotatE link prediction, and quantitative evaluation — run end-to-end here
# against real biomedical text and produce genuine, reproducible ranked drug–disease hypotheses
# with literature evidence attached where available. The accompanying web app
# (`app/index.html`) presents these same results through the query interface described in the
# project's expected output.
