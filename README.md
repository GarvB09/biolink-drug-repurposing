# BioLink — Biomedical NER and Relation Extraction (BC5CDR)

An NLP pipeline that finds **chemicals and diseases** in PubMed abstracts and decides whether a chemical **induces** a disease, evaluated on a public benchmark (CPU only), plus a small web app for searching the extracted relations.

| Stage | Data | Models | Metric |
| --- | --- | --- | --- |
| 1. NER (Chemical, Disease) | BC5CDR (Li et al., 2016), official train/dev/test split | Dictionary, CRF, BERT-base, PubMedBERT | strict entity-level P/R/F1 |
| 2. Relation extraction (chemical-induced disease) | BC5CDR CID pairs | co-occurrence, TF-IDF + LogReg, PubMedBERT pair classifier | pair-level P/R/F1 |

**Preprocessing study:** stop-word removal, Porter stemming, WordNet lemmatisation and POS tags, plus Bag of Words vs TF-IDF, measured on the relation task (`scripts/preprocessing_study.py`; lemmatisation best at 51.1 F1, stop-word removal no gain). ROC-AUC and confusion matrices for every model are in `results/extra_metrics.json`.

**Test results:** NER F1 — dictionary 77.1, CRF 83.0, BERT-base 86.6, **PubMedBERT 90.4**. Relations F1 — same-sentence co-occurrence 43.0, TF-IDF+LR 50.0, **PubMedBERT 59.8**.

## Run

```bash
pip install torch transformers scikit-learn sklearn-crfsuite seqeval nltk
# download BC5CDR (github.com/JHnlp/BioCreative-V-CDR-Corpus) and unzip into data/raw/bc5cdr/
cd scripts
python prepare_data.py      # data/processed/*, results/dataset_stats.json
./run_all.sh                # PubMedBERT NER, relation extraction, BERT-base NER -> results/*.json
python ner_classic.py       # dictionary + CRF baselines
python preprocessing_study.py   # preprocessing/feature study + ROC-AUC + confusion matrices (~1 min)
python build_app.py         # app/index.html (open in any browser)
```

`run_all.sh` skips any step whose result file already exists, so it can be re-run safely after an interruption.

## Folders
- `scripts/` — `common.py`, `prepare_data.py`, `ner_classic.py`, `ner_bert.py`, `relations.py`, `preprocessing_study.py`, `build_app.py`, `run_all.sh`
- `data/raw/` — downloaded BC5CDR files; `data/processed/` — generated splits (not stored in git)
- `results/` — metrics JSON, prediction files, logs, app screenshot
- `models/` — saved fine-tuned checkpoints (created by the scripts, not stored in git)
- `app/` — `index.html` search interface (`_base.html` is its CSS/library shell)
- `legacy/` — the earlier 4-abstract demo, kept for reference only

## Limitations
- Relation extraction is scored with gold entities (the official CID setting); NER errors are not propagated end to end.
- Cross-sentence relations are the weak point (F1 45.6 vs 65.9 within a sentence).
- Single-seed transformer runs with short CPU schedules; no confidence intervals.
- Only chemicals and diseases from 1,500 abstracts; no genes, full text or other domains.

## More documentation
- `VIVA_GUIDE.md` — full explanation of the dataset, every script, results, limitations and likely viva questions.
- BC5CDR is distributed by the BioCreative organisers for research use; download it yourself (it is not redistributed here).
