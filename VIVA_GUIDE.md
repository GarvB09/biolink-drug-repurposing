# BioLink — Complete Project Explanation (Viva Guide)

Everything you need to explain, defend and demo this project. Read sections 1–3 first (the 5-minute version), then use the rest as reference. All numbers are real outputs in `results/`.

---

## 1. The 60-second pitch

**BioLink** reads biomedical abstracts and extracts structured facts from them. It has two NLP stages, each trained and scored on a public benchmark:

1. **Named Entity Recognition (NER)** – find *chemical* and *disease* mentions in PubMed abstracts (BC5CDR corpus).
2. **Relation extraction (RE)** – decide whether a chemical **induces** a disease (BC5CDR "CID" relation).

A web app lets you search a disease or chemical and see the relations the pipeline extracted, with the evidence sentence and a PubMed link.

**Headline results (test data):** PubMedBERT NER **90.4 F1** (CRF 83.0, BERT-base 86.6, dictionary 77.1) · relation extraction **59.8 F1** (TF-IDF+LR 50.0, same-sentence co-occurrence 43.0).

**Novelty (be modest and accurate):** not a new model — a reproducible, CPU-only pipeline where both stages are evaluated on a public benchmark against matched baselines (including a general-vs-biomedical encoder ablation), served through a search interface with evidence.

---

## 2. Why this problem? (motivation)

- Biomedical knowledge (which drug causes which disease/side effect) is buried in millions of abstracts; nobody can read it all. NLP can extract it automatically.
- Extraction needs two steps: find the entities (NER), then decide how they relate (RE).
- **Research gap:** many papers evaluate a single stage, or demo on a few hand-picked examples. We evaluate both stages on a benchmark with simple baselines and a domain-pretraining ablation.

## 3. Big picture

```
BC5CDR abstracts ──► preprocess ──► NER (Dictionary / CRF / BERT / PubMedBERT) ──► Relation extraction (co-occurrence / TF-IDF+LR / PubMedBERT) ──► Search app
```

### Folder map
```
scripts/common.py        shared paths + BC5CDR parsing, tokenising, sentence splitting, BIO tags
scripts/prepare_data.py  builds NER data and RE candidate pairs; writes dataset_stats.json
scripts/ner_classic.py   dictionary + CRF NER, strict F1 evaluation
scripts/ner_bert.py      fine-tunes BERT-base / PubMedBERT token classifier
scripts/relations.py     RE baselines + PubMedBERT pair classifier
scripts/build_app.py     generates app/index.html from the results files
scripts/run_all.sh       runs the experiment queue (skips finished steps)
app/index.html           single-file web app (+ _base.html = CSS and vis-network shell)
results/                 all metrics JSON, predictions, logs, screenshot
legacy/                  the earlier 4-abstract demo (reference only)
```
`data/raw`, `data/processed` and `models/` (checkpoints) are **not** in git; the scripts regenerate them.

---

## 4. Dataset (know this cold)

### BC5CDR (BioCreative V Chemical–Disease Relation corpus; Li et al., 2016)
- **1,500 PubMed abstracts**, English, split by the organisers **500 train / 500 dev / 500 test**.
- Gold **Chemical** and **Disease** mentions with character offsets, each normalised to a **MeSH ID**.
- Gold **CID** (Chemical-Induced-Disease) relations at **abstract level** between (chemical ID, disease ID).
- Why chosen: standard benchmark for BioNER and RE, official split, widely cited.

| Split | Abstracts | Sentences | Tokens | Chem mentions | Disease mentions | CID relations | RE candidate pairs (pos) |
|---|---|---|---|---|---|---|---|
| Train | 500 | 4,562 | 118,170 | 5,203 | 4,182 | 1,038 | 5,432 (1,038) |
| Dev | 500 | 4,565 | 117,451 | 5,347 | 4,244 | 1,012 | 5,261 (1,012) |
| Test | 500 | 4,798 | 124,748 | 5,385 | 4,424 | 1,066 | 5,405 (1,066) |

About 70–73% of positive pairs have both mentions in one sentence; ~30% are cross-sentence (hard).

### What was wrong with the original demo (why we changed it)
It used 4 abstracts, a hand-written lexicon for NER and hand-curated edges → nothing could be evaluated honestly. That version is archived in `legacy/`.

---

## 5. Code walkthrough (stage by stage)

### 5.1 `common.py` — shared utilities
- `parse_pubtator(path)`: reads PubTator files. Each doc = `title|t|…`, `abstract|a|…`, then tab-separated entity lines (`pmid start end text type id`) and relation lines (`pmid CID chem_id dis_id`). `text = title + " " + abstract` is the coordinate system the offsets use.
- `tokenize(text)`: regex `\w+|[^\w\s]` → words and single punctuation marks, **keeping character offsets** so annotations can be mapped to tokens.
- `split_sentences(doc)`: title is sentence 0; the abstract is split at `[.!?]` followed by whitespace and a capital/digit/bracket; any split that would **cut through an entity mention is merged back**.
- `bio_tags(tokens, entities)`: IOB2 tags. A token overlapping an entity gets `B-Type` for the first token and `I-Type` afterwards.
- `tags_to_spans`: turns tag sequences back into (type, start, end) spans.
- `write_json` has a `default=` handler because seqeval/numpy return numpy floats (this caused a crash early on).

### 5.2 `prepare_data.py` — Stage 0
For each split writes `bc5cdr_*.json` (docs), `ner_*.json` (one row per sentence: tokens, offsets, tags) and `re_*.json` (candidate pairs).
- **RE candidates:** group mentions by (type, MeSH id); every chemical-ID × disease-ID in an abstract is a candidate; `label=1` if (chem,dis) is in the gold CID set; `same_sentence` flag; most common surface form kept as the name. Check printed: `relations_without_candidate = 0` (every gold relation has a candidate).

### 5.3 `ner_classic.py` — Stage 1a (baselines)
**Dictionary:** a gazetteer of every mention string in train+dev (lower-cased token tuples → type). Longest-match-first (up to 12 tokens). Single tokens of ≤2 characters are ignored. High precision, low recall (cannot generalise to unseen names).

**CRF** (`sklearn_crfsuite`, Lafferty et al. 2001): linear-chain CRF, L-BFGS, `c1=c2=0.05`, 150 iterations, `all_possible_transitions=True`. Features per token: lower-case word, 3/4-char suffix, 3-char prefix, **word shape** (e.g. `Xxxxx`, `xx-dd`), isupper/istitle/isdigit, length, plus lower/shape/suffix of tokens at offsets ±1, ±2 (and a pad marker at sentence edges).
- Why a CRF models sequences: it learns tag-transition scores (`I-Disease` cannot follow `O` easily) and decodes the best whole tag sequence with Viterbi, not token by token.

Evaluation: `seqeval` **strict entity-level** P/R/F1 (a predicted entity counts only if **type and both boundaries** match exactly).

### 5.4 `ner_bert.py` — Stage 1b (transformers)
- Model: `AutoModelForTokenClassification` = encoder + **one linear layer** over each token's final hidden state → 5 labels (`O`, `B/I-Chemical`, `B/I-Disease`).
- **Sub-word alignment (`encode`)**: BERT splits words into word-pieces. Label goes on the **first word-piece** of each word; the others get `-100` (ignored by the loss). At prediction time the first word-piece's argmax is the word's tag.
- **Illegal tag repair:** an `I-X` that follows `O` or another type is turned into `B-X`.
- **Length bucketing (`length_batches`)**: sentences sorted by length, cut into batches, batches shuffled. Reduces padding → big speed-up on CPU.
- Training: AdamW, lr **5e-5**, weight decay 0.01, batch 16, **2 epochs**, 10% linear warm-up then linear decay, gradient clipping 1.0, seed 13, max length 192 word-pieces. Trained on **train+dev** with a fixed schedule (BioBERT/PubMedBERT practice), scored **once** on test. (No model selection → no test leakage.)
- Checkpoints: `bert-base-cased` (general domain) vs `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` = **PubMedBERT** (pre-trained from scratch on PubMed text with an in-domain vocabulary; Gu et al., 2021).
- Known simplification: sentences longer than 192 word-pieces are truncated (words past the cut stay `O`).

### 5.5 `relations.py` — Stage 2
Task: for each (chemical ID, disease ID) pair in an abstract → is it a CID relation? (binary).
- **`marked_context(doc, row)`**: pick the **closest** chemical/disease mention pair (by start offset). If both are in one sentence, use that whole sentence; otherwise use the span from the first mention to the end of the second. The two mentions are wrapped in markers `[C] … [/C]` and `[D] … [/D]` (new tokens added to the tokenizer and embeddings resized) so the model knows *which* two entities it is classifying — the "entity-marker" approach.
- Baselines: **same-sentence co-occurrence** (predict 1 if mentions share a sentence), **abstract co-occurrence** (predict 1 for all pairs → recall 100%, precision 19.7%), **TF-IDF (1–2-grams, min_df 2, sublinear tf) + logistic regression** (`C=4`, `class_weight=balanced`).
- **PubMedBERT pair classifier:** `AutoModelForSequenceClassification` (2 classes), AdamW lr **3e-5**, batch 16, **2 epochs**, max 256 word-pieces, trained on **train only**. The **decision threshold is tuned on dev** (maximising F1) → final 0.21, then applied to test. A low threshold favours recall (74.1%) over precision (50.1%).
- Extra analysis: F1 split into same-sentence (65.9) vs cross-sentence (45.6).
- Scoring: pair-level P/R/F1 with **gold entities** (the official CID setting).

### 5.6 `preprocessing_study.py` — classic preprocessing, features and extra metrics
- Re-uses the relation candidates and `marked_context` from `relations.py`. Entity markers are turned into placeholder words (`chemstart`, `chemend`, `disstart`, `disend`) so no step deletes or changes them.
- Eight variants, all with the **same** logistic regression (C=4, balanced) and a threshold tuned on dev:
  Bag of Words; TF-IDF; TF-IDF + stop-word removal (NLTK English list, 198 words); stop-word removal keeping negations (no, not, nor, without, against); Porter stemming; WordNet lemmatisation (POS-guided); stop words + lemmatisation; POS-tagged tokens (`word_TAG`).
- Also computes **ROC-AUC** and **confusion matrices** for every relation model (PubMedBERT from its saved test probabilities) and **token-level confusion matrices** for the NER models, and writes a worked example of each step on one sentence (`results/preprocessing_example.json`).
- Why classic steps are applied only to BoW/TF-IDF: transformers need raw text for their word-piece vocabulary; NER needs every token so gold character offsets stay valid.

### 5.7 `build_app.py` and `app/index.html`
Pure static page (works offline, `file://`). The script embeds the PubMedBERT relation predictions for the 500 test abstracts (pairs above the tuned threshold, 1,576 of them) plus the metric tables as one JS object and writes the HTML (CSS and the vis-network library come from `_base.html`). The JS: search a disease or chemical name → list the extracted relations with the evidence sentence, probability, a "matches gold" badge if the pair is also in the human annotation, and a PubMed link; a small graph of the query and its related entities; tabs with the Relation and NER evaluation tables.

### 5.8 `run_all.sh`
Runs: PubMedBERT NER → relation extraction → BERT-base NER, each skipped if its result file exists (safe to resume after a crash). `ner_classic.py` (dictionary + CRF) is run separately. Environment variable `RE_EPOCHS` sets the relation-extraction epochs.

---

## 6. Results (test data, single run)

**NER (strict entity-level, %)**
| Model | P | R | F1 | Chem F1 | Disease F1 | Train time |
|---|---|---|---|---|---|---|
| Dictionary | 86.2 | 69.7 | 77.1 | 79.3 | 74.5 | 2 s |
| CRF | 87.3 | 79.0 | 83.0 | 86.7 | 78.5 | 35 s |
| BERT-base | 85.1 | 88.3 | 86.6 | 91.1 | 81.4 | 32 min |
| **PubMedBERT** | 89.6 | 91.1 | **90.4** | 93.9 | 86.2 | 29 min |

**Relation extraction (pair level, gold entities, %)**
| Model | P | R | F1 |
|---|---|---|---|
| Same-sentence co-occurrence | 31.0 | 70.2 | 43.0 |
| Abstract co-occurrence | 19.7 | 100 | 32.9 |
| TF-IDF + LR | 40.8 | 64.5 | 50.0 |
| **PubMedBERT** | 50.1 | 74.1 | **59.8** (same-sentence 65.9, cross-sentence 45.6) |

**ROC-AUC / confusion matrix (relations, 5,405 test pairs):** PubMedBERT AUC 0.853 (TP 790, FP 786, FN 276, TN 3,553); TF-IDF+LR AUC 0.789 (TP 688, FP 999, FN 378, TN 3,340). NER token accuracy: PubMedBERT 97.9%, BERT-base 97.3%, CRF 96.5% (dominated by the O tag, so entity F1 is the headline). Only 27 tokens confuse chemical with disease; most errors are missed/extra mentions and disease B/I boundaries.

**Preprocessing study (TF-IDF/BoW + logistic regression, relations, F1 % / ROC-AUC)**
| Variant | F1 | ROC-AUC |
|---|---|---|
| Bag of Words | 49.2 | 0.779 |
| TF-IDF | 49.4 | 0.793 |
| + stop-word removal | 49.4 | 0.786 |
| + stop-word removal, negations kept | 49.9 | 0.787 |
| + Porter stemming | 50.2 | 0.794 |
| **+ WordNet lemmatisation** | **51.1** | 0.794 |
| + stop words + lemmatisation | 49.7 | 0.787 |
| + POS-tagged tokens | 49.4 | 0.786 |

Take-away: classic preprocessing moves F1 by < 2 points (within single-run noise); lemmatisation helps most, stop-word removal not at all. PubMedBERT's lead (8.7 points over the best variant) is 5× the best preprocessing gain.

**How to interpret:** PubMedBERT > BERT-base by 3.7 F1 (biomedical pretraining helps, mostly on diseases); transformers beat the CRF mainly through **recall**; relation extraction is hardest across sentences.

---

## 7. Design decisions you may be asked to justify

| Decision | Reason |
|---|---|
| BC5CDR | public, standard, official splits, reproducible |
| Strict entity F1 (seqeval) | the standard; no partial credit |
| NER trained on train+dev, fixed schedule | follows BioBERT/PubMedBERT practice; test touched once |
| RE threshold tuned on dev only | avoids tuning on test |
| Gold entities for RE | the official CID task setting; makes RE separable from NER |
| Length bucketing, 2 epochs | made CPU-only training feasible (~30 min per model) |
| BioBERT not run | time; BERT-base vs PubMedBERT already shows the domain effect |
| Word-piece first-token labelling | standard BERT token-classification practice |
| No stop-word removal / stemming | function words carry relation cues ("induced by"); word-pieces handle inflection |

---

## 8. Honest limitations (say these before they are asked)
1. **Single run, single seed** for all models, no confidence intervals.
2. **RE uses gold entities**; real end-to-end performance would be lower (NER errors propagate). Not evaluated end to end.
3. **Cross-sentence RE is weak** (45.6 F1); no document-level model.
4. **Short CPU schedules** (2 epochs), no hyper-parameter search.
5. Only chemicals and diseases; 1,500 abstracts from a 2015 shared task; no genes, full text or other domains.
6. Sentences > 192 word-pieces are truncated in NER; the sentence splitter is rule-based.
7. The app shows model predictions on the **test** abstracts only; "extracted" pairs that are not in the gold annotation may be errors or simply un-annotated.
8. The relations are adverse-effect (induced-disease) associations, not treatment evidence.

## 9. What I would do next
Larger corpora (BioRED, PubTator), end-to-end evaluation, document-level relation extraction, entity normalisation (link mentions to MeSH IDs), multiple seeds and tuning, comparison with few-shot LLM extraction, building a knowledge graph from the extracted relations.

---

## 10. Likely viva questions with answers

**Basics**
1. *What is NER / RE?* NER tags spans of text as entity types; RE decides the relation between two entities.
2. *What is IOB2 tagging?* `B-` marks the first token of an entity, `I-` continues it, `O` is outside; e.g. "kidney failure" → `B-Disease I-Disease`.
3. *What does "chemical-induced disease" mean?* The chemical causes the disease (an adverse effect), e.g. cisplatin → nephrotoxicity.

**Data**
4. *Why BC5CDR?* Standard, official split, gold entities + relations with MeSH IDs.
5. *What is the label distribution in RE?* ~19.7% positive (1,066 of 5,405 test pairs); hence precision/recall/F1, not accuracy.
6. *Why not accuracy?* Always predicting "no relation" gives ~80% accuracy and is useless.
7. *Licence?* BC5CDR is distributed for research use; raw data are not committed — the README explains how to download them.

**NER**
8. *Why does PubMedBERT beat BERT-base?* Pretrained on PubMed text with a biomedical vocabulary → chemical/disease names split into fewer, more meaningful word-pieces; +3.7 F1 here. (Single run, so the exact gap is uncertain.)
9. *Why CRF over softmax-per-token?* Models tag transitions and decodes the best sequence jointly.
10. *Why is the dictionary precise but low recall?* It only knows names seen in training.
11. *What does strict F1 mean?* Exact type and boundary match; F1 = 2PR/(P+R).
12. *How do you handle word-pieces?* Label on first sub-token, `-100` on the rest, predict from first sub-token.
13. *Why no stop-word removal or stemming?* Function words carry relation cues ("induced by"); word-piece tokenisation handles inflection.

**Preprocessing**
P1. *Did you do stop-word removal, stemming, lemmatisation, POS tagging?* Yes — as a measured study on the TF-IDF relation model (Table 8 in the report). Lemmatisation +1.7 F1, stemming +0.8, stop-word removal 0.
P2. *Why does stop-word removal not help?* Words like "by", "in", "no", "not" carry relation cues ("induced by", "no evidence of"); removing them throws away signal. Keeping negations recovers half a point.
P3. *Why lemmatisation over stemming?* Lemmas are real words ("induce", "hypotension"); Porter stemming over-cuts biomedical terms ("hypotens", "volum").
P4. *Why not apply these to BERT?* Its word-piece vocabulary was learned on raw text; changing words would create inputs it never saw.
P5. *BoW vs TF-IDF?* TF-IDF down-weights words common to every context; it scored slightly higher (ROC-AUC 0.793 vs 0.779).

**Relation extraction**
14. *How do you tell the model which pair to classify?* Entity markers `[C]…[/C]`, `[D]…[/D]`.
15. *Why is precision only 50%?* The threshold chosen on dev favours recall; cross-sentence and negation cases are hard; no tuning.
16. *Why is cross-sentence weaker?* Evidence is spread across sentences and our context is the span between the mentions; a document-level model would be needed.
17. *Is 59.8 F1 good?* It beats all baselines clearly, but we make no claim to match the best published systems.
18. *Why a threshold of 0.21?* It maximised F1 on the dev split; test was not used for tuning.

**Evaluation / rigour**
19. *Single run — is that valid?* It is a limitation; differences of a few points are not conclusive. With more time: multiple seeds and confidence intervals.
20. *Why 2 epochs?* CPU-only laptop and a deadline; the NER loss had already fallen to ~0.03.
21. *How do you know there is no train/test overlap?* BC5CDR splits are official and disjoint by PMID.
22. *Why train NER on train+dev?* Fixed schedule, no model selection, so dev is free to use; test is scored once.

**System**
23. *What does the app show?* Search a disease/chemical → extracted relations with evidence sentence, probability, gold-match badge, PubMed link, a small relation graph and evaluation tables.
24. *Can it run offline?* Yes, single HTML file with embedded data.
25. *How is it reproducible?* One prep script + `run_all.sh`, fixed seeds, logs and result JSONs in `results/`.

**Curveballs**
26. *What is novel?* Integration and a controlled comparison under one protocol, CPU-only, with an evidence-linked interface — not a new model.
27. *Why didn't you use an LLM?* Fine-tuned domain encoders still beat zero/few-shot LLMs on extraction benchmarks (Chen et al., 2025) and give traceable outputs; LLMs are listed as future work.
28. *What changed from your first version?* Replaced 4 abstracts + a hand-written lexicon with a real benchmark; added baselines, held-out evaluation and a rebuilt app. A knowledge-graph embedding (link-prediction) stage was removed because it is graph machine learning rather than NLP and used a graph that did not come from the text pipeline.

---

## 11. Run it yourself
```bash
pip install torch transformers scikit-learn sklearn-crfsuite seqeval nltk
# unzip BC5CDR (github.com/JHnlp/BioCreative-V-CDR-Corpus) into data/raw/bc5cdr/
cd scripts
python prepare_data.py
python ner_classic.py
./run_all.sh
python preprocessing_study.py   # ~1 min, needs nltk            # ~1.5 h on a 16-core CPU
python build_app.py     # open app/index.html
```

## 12. Glossary
**BERT** transformer encoder pre-trained with masked-language modelling · **PubMedBERT** BERT pre-trained from scratch on PubMed · **CRF** conditional random field for sequence labelling · **Word-piece** sub-word token · **TF-IDF** term-frequency × inverse-document-frequency weighting · **MeSH** NLM's controlled vocabulary · **CID** chemical-induced disease · **Entity markers** special tokens placed around the two entities to classify · **Strict F1** exact span and type match.
