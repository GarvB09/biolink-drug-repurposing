# BioLink — NER → Relation Extraction → Knowledge-Graph Link Prediction (drug repurposing)

A three-stage biomedical NLP pipeline trained and evaluated on public benchmark data (CPU only).

| Stage | Data | Models | Metric |
| --- | --- | --- | --- |
| 1. NER (Chemical, Disease) | BC5CDR (Li et al., 2016), official train/dev/test | Dictionary, CRF, BERT-base, PubMedBERT | strict entity-level P/R/F1 |
| 2. Relation extraction (chemical-induced disease) | BC5CDR CID pairs | co-occurrence, TF-IDF + LogReg, PubMedBERT pair classifier | pair-level P/R/F1 |
| 3. Link prediction (compound treats disease) | Hetionet v1.0 subset (Himmelstein et al., 2017) | Random, Popularity, TransE, DistMult, RotatE | filtered MRR / MR / Hits@K |

## Run

```bash
pip install torch transformers scikit-learn sklearn-crfsuite seqeval
# download data into data/raw/ (BC5CDR: github.com/JHnlp/BioCreative-V-CDR-Corpus, Hetionet: github.com/hetio/hetionet)
cd scripts
python prepare_data.py      # data/processed/*, results/dataset_stats.json
./run_all.sh                # PubMedBERT NER, relation extraction, link prediction, BERT-base NER -> results/*.json
python build_app.py         # app/index.html (open in any browser)
```

`run_all.sh` skips any step whose result file already exists, so it can be re-run safely after an interruption.

## Folders
- `scripts/` — `common.py`, `prepare_data.py`, `ner_classic.py`, `ner_bert.py`, `relations.py`, `linkpred.py`, `build_app.py`, `run_all.sh`
- `data/raw/` — downloaded BC5CDR and Hetionet files; `data/processed/` — generated splits
- `results/` — metrics JSON, prediction files and logs
- `models/` — saved fine-tuned checkpoints and KGE weights (created by the scripts, not stored in git)
- `app/` — `index.html` query interface (`_base.html` is its CSS/library shell)
- `legacy/` — the earlier 4-abstract demo (hand-built lexicon and curated edges), kept for reference only

## Limitations
- Relation extraction is scored with gold entities (the official CID setting); NER errors are not propagated.
- BC5CDR (MeSH ids) and Hetionet (DrugBank / Disease Ontology ids) are not merged; the app shows literature associations next to graph predictions.
- Only 76 link-prediction test edges; transformer runs are single-seed with short CPU schedules.

## More documentation
- `VIVA_GUIDE.md` — full explanation of the datasets, every script, the maths, results, limitations and likely viva questions.
- Data licences: Hetionet is CC0. BC5CDR is distributed by the BioCreative organisers for research use; download it yourself (it is not redistributed here).
