# %% [markdown]
# ## Preprocessing and feature study + extra metrics
#
# 1. Classic text preprocessing (stop-word removal, stemming, lemmatisation, POS tagging) and
#    two feature representations (Bag of Words, TF-IDF) are evaluated on the BC5CDR relation task
#    with the same logistic-regression classifier, so each step's effect is measured, not assumed.
# 2. ROC-AUC and confusion matrices for every relation model (incl. PubMedBERT, from saved
#    test probabilities) and token-level confusion matrices for NER.
# 3. A worked example of every preprocessing step on one test sentence (for the report).
#
# Uses NLTK (Bird et al., 2009): English stop-word list, Porter stemmer, WordNet lemmatiser,
# averaged-perceptron POS tagger. The decision threshold is always tuned on the dev split.

import re
import time
from collections import Counter

import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

from common import NER_LABELS, PROC_DIR, RESULTS_DIR, read_json, write_json
from relations import best_threshold, load, marked_context, prf

MARKERS = {"[C]": "chemstart", "[/C]": "chemend", "[D]": "disstart", "[/D]": "disend"}
PLACEHOLDERS = set(MARKERS.values())
TOKEN_RE = re.compile(r"\[/?[CD]\]|[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?")
STOP = set(stopwords.words("english"))
NEGATIONS = {"no", "not", "nor", "without", "against"}
stemmer, lemmatizer = PorterStemmer(), WordNetLemmatizer()


def tokens(text):
    """Lower-cased word tokens; entity markers become placeholder words that later steps leave alone."""
    return [MARKERS.get(t, t.lower()) for t in TOKEN_RE.findall(text)]


def wn_pos(tag):
    return {"J": "a", "V": "v", "R": "r"}.get(tag[0], "n")


def preprocess(text, steps):
    toks = tokens(text)
    if "pos" in steps or "lemma" in steps:
        tagged = pos_tag([t for t in toks])
    else:
        tagged = [(t, "") for t in toks]
    out = []
    for tok, tag in tagged:
        if tok in PLACEHOLDERS:
            out.append(tok)
            continue
        if "stop" in steps and tok in STOP:
            continue
        if "stop_keep_neg" in steps and tok in STOP and tok not in NEGATIONS:
            continue
        if "stem" in steps:
            tok = stemmer.stem(tok)
        if "lemma" in steps:
            tok = lemmatizer.lemmatize(tok, wn_pos(tag))
        if "pos" in steps:
            tok = f"{tok}_{tag}"
        out.append(tok)
    return " ".join(out)


VARIANTS = [  # (label, feature type, preprocessing steps)
    ("Bag of Words, lower-case tokens", "bow", ()),
    ("TF-IDF, lower-case tokens", "tfidf", ()),
    ("TF-IDF + stop-word removal", "tfidf", ("stop",)),
    ("TF-IDF + stop-word removal (negations kept)", "tfidf", ("stop_keep_neg",)),
    ("TF-IDF + Porter stemming", "tfidf", ("stem",)),
    ("TF-IDF + WordNet lemmatisation", "tfidf", ("lemma",)),
    ("TF-IDF + stop-word removal + lemmatisation", "tfidf", ("stop", "lemma")),
    ("TF-IDF + POS-tagged tokens", "tfidf", ("pos",)),
]


def cm(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main():
    data = {}
    for split in ("train", "dev", "test"):
        docs, rows = load(split)
        data[split] = {"texts": [marked_context(docs[r["pmid"]], r)[0] for r in rows],
                       "y": np.array([r["label"] for r in rows]), "rows": rows}
    ytest = data["test"]["y"]

    # ---- 1. preprocessing x feature study
    study = {}
    for label, feat, steps in VARIANTS:
        t0 = time.time()
        proc = {s: [preprocess(t, steps) for t in data[s]["texts"]] for s in data}
        vec_cls = CountVectorizer if feat == "bow" else TfidfVectorizer
        kw = {"ngram_range": (1, 2), "min_df": 2, "token_pattern": r"\S+", "lowercase": False}
        if feat == "tfidf":
            kw["sublinear_tf"] = True
        vec = vec_cls(**kw)
        xtr = vec.fit_transform(proc["train"])
        clf = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced").fit(xtr, data["train"]["y"])
        thr = best_threshold(data["dev"]["y"], clf.predict_proba(vec.transform(proc["dev"]))[:, 1])
        pt = clf.predict_proba(vec.transform(proc["test"]))[:, 1]
        pred = (pt >= thr).astype(int)
        vocab = Counter(w for t in proc["train"] for w in t.split())
        study[label] = {**prf(ytest, pred), "roc_auc": float(roc_auc_score(ytest, pt)), "threshold": thr,
                        "confusion": cm(ytest, pred), "vocabulary_size": len(vocab),
                        "avg_tokens": float(np.mean([len(t.split()) for t in proc["train"]])),
                        "seconds": round(time.time() - t0, 1)}
        print(label, {k: round(v, 4) if isinstance(v, float) else v for k, v in study[label].items()}, flush=True)
    write_json(RESULTS_DIR / "preprocessing_results.json", study)

    # ---- 2. extra metrics for the main relation models
    extra = {"relations": {}, "ner_token_confusion": {}}
    same = np.array([int(r["same_sentence"]) for r in data["test"]["rows"]])
    extra["relations"]["Same-sentence co-occurrence"] = {"confusion": cm(ytest, same), "roc_auc": None}
    extra["relations"]["Abstract co-occurrence (all pairs)"] = {"confusion": cm(ytest, np.ones_like(ytest)), "roc_auc": None}
    probs = read_json(RESULTS_DIR / "re_test_probs.json")
    re_res = read_json(RESULTS_DIR / "re_results.json")
    thr = re_res["PubMedBERT pair classifier"]["threshold"]
    pb = np.array([p["prob"] for p in probs])
    yb = np.array([p["label"] for p in probs])
    extra["relations"]["PubMedBERT pair classifier"] = {"confusion": cm(yb, (pb >= thr).astype(int)),
                                                        "roc_auc": float(roc_auc_score(yb, pb))}
    # the exact TF-IDF + logistic-regression model of relations.py (same settings), for its AUC and confusion
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"[A-Za-z\[\]/]+")
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(
        vec.fit_transform(data["train"]["texts"]), data["train"]["y"])
    t_thr = best_threshold(data["dev"]["y"], clf.predict_proba(vec.transform(data["dev"]["texts"]))[:, 1])
    pt = clf.predict_proba(vec.transform(data["test"]["texts"]))[:, 1]
    extra["relations"]["TF-IDF + logistic regression"] = {"confusion": cm(ytest, (pt >= t_thr).astype(int)),
                                                         "roc_auc": float(roc_auc_score(ytest, pt)),
                                                         **prf(ytest, (pt >= t_thr).astype(int))}

    gold = [t for r in read_json(PROC_DIR / "ner_test.json") for t in r["tags"]]
    for name, f in (("CRF", "ner_test_predictions_crf.json"), ("BERT-base", "ner_test_predictions_bert-base.json"),
                    ("PubMedBERT", "ner_test_predictions_pubmedbert.json")):
        pred = [t for s in read_json(RESULTS_DIR / f) for t in s]
        m = confusion_matrix(gold, pred, labels=NER_LABELS)
        extra["ner_token_confusion"][name] = {"labels": NER_LABELS, "matrix": m.tolist(),
                                              "token_accuracy": float(np.trace(m) / m.sum())}
    write_json(RESULTS_DIR / "extra_metrics.json", extra)
    print({k: v for k, v in extra["relations"].items()})
    print({k: round(v["token_accuracy"], 4) for k, v in extra["ner_token_confusion"].items()})

    # ---- 3. worked example of each step on one positive same-sentence test pair
    idx = next(i for i, r in enumerate(data["test"]["rows"]) if r["label"] and r["same_sentence"]
               and 12 <= len(data["test"]["texts"][i].split()) <= 22)
    text = data["test"]["texts"][idx]
    example = {"original": text, "tokens": tokens(text)}
    for key, steps in (("stop_words_removed", ("stop",)), ("stemmed", ("stem",)),
                       ("lemmatised", ("lemma",)), ("pos_tagged", ("pos",))):
        example[key] = preprocess(text, steps).split()
    write_json(RESULTS_DIR / "preprocessing_example.json", example)
    for k, v in example.items():
        print(k, ":", v if isinstance(v, str) else " ".join(v))


if __name__ == "__main__":
    main()
