# %% [markdown]
# ## Stage 1a — NER baselines on BC5CDR (dictionary matching, CRF)
#
# - **Dictionary baseline**: gazetteer of every mention string seen in the training split,
#   longest-match-first over tokens (case-insensitive). This is what the original demo did.
# - **CRF**: linear-chain Conditional Random Field with hand-crafted lexical, orthographic
#   and context-window features (the classical pre-neural BioNER approach).
# Both are evaluated with strict entity-level P/R/F1 (seqeval, IOB2) on the official test split.

import time

import sklearn_crfsuite
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from common import MODEL_DIR, PROC_DIR, RESULTS_DIR, read_json, tags_to_spans, write_json


def load(split):
    return read_json(PROC_DIR / f"ner_{split}.json")


def entity_scores(gold, pred):
    rep = classification_report(gold, pred, output_dict=True, zero_division=0)
    out = {"precision": precision_score(gold, pred), "recall": recall_score(gold, pred),
           "f1": f1_score(gold, pred)}
    for t in ("Chemical", "Disease"):
        out[t] = {k: rep[t][k] for k in ("precision", "recall", "f1-score")}
    return out


# ---------------------------------------------------------------- dictionary baseline

class DictionaryNER:
    def __init__(self, max_len=12):
        self.lex = {}
        self.max_len = max_len

    def fit(self, rows):
        for r in rows:
            for etype, s, e in tags_to_spans(r["tags"]):
                key = tuple(t.lower() for t in r["tokens"][s:e])
                self.lex.setdefault(key, etype)
        return self

    def predict_one(self, tokens):
        low = [t.lower() for t in tokens]
        tags, i = ["O"] * len(tokens), 0
        while i < len(tokens):
            for n in range(min(self.max_len, len(tokens) - i), 0, -1):
                etype = self.lex.get(tuple(low[i:i + n]))
                # skip single stop-word-like matches (e.g. "of") that only appear inside entities
                if etype and (n > 1 or len(low[i]) > 2):
                    tags[i] = "B-" + etype
                    for j in range(i + 1, i + n):
                        tags[j] = "I-" + etype
                    i += n
                    break
            else:
                i += 1
        return tags


# ---------------------------------------------------------------- CRF

def word_shape(w):
    s = "".join("X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c for c in w)
    out = []
    for c in s:  # collapse repeats
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def token_features(tokens, i):
    w = tokens[i]
    f = {"bias": 1.0, "w.lower": w.lower(), "w.suf3": w[-3:].lower(), "w.suf4": w[-4:].lower(),
         "w.pre3": w[:3].lower(), "w.shape": word_shape(w), "w.isupper": w.isupper(),
         "w.istitle": w.istitle(), "w.isdigit": w.isdigit(), "w.len": min(len(w), 15)}
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(tokens):
            v = tokens[j]
            f[f"{off}:w.lower"] = v.lower()
            f[f"{off}:w.shape"] = word_shape(v)
            f[f"{off}:w.suf3"] = v[-3:].lower()
        else:
            f[f"{off}:pad"] = True
    return f


def sent_features(tokens):
    return [token_features(tokens, i) for i in range(len(tokens))]


def main():
    train, dev, test = load("train"), load("dev"), load("test")
    gold = [r["tags"] for r in test]
    results = {}

    t0 = time.time()
    dic = DictionaryNER().fit(train + dev)
    pred_dic = [dic.predict_one(r["tokens"]) for r in test]
    results["Dictionary (train+dev gazetteer)"] = {**entity_scores(gold, pred_dic),
                                                   "train_seconds": round(time.time() - t0, 1)}
    print("Dictionary:", results["Dictionary (train+dev gazetteer)"])

    t0 = time.time()
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.05, c2=0.05, max_iterations=150,
                               all_possible_transitions=True)
    crf.fit([sent_features(r["tokens"]) for r in train + dev], [r["tags"] for r in train + dev])
    pred_crf = crf.predict([sent_features(r["tokens"]) for r in test])
    results["CRF (lexical + context features)"] = {**entity_scores(gold, pred_crf),
                                                   "train_seconds": round(time.time() - t0, 1)}
    print("CRF:", results["CRF (lexical + context features)"])

    write_json(RESULTS_DIR / "ner_classic_results.json", results)
    write_json(RESULTS_DIR / "ner_test_predictions_crf.json", pred_crf, indent=None)
    import pickle
    with open(MODEL_DIR / "ner_crf.pkl", "wb") as f:
        pickle.dump(crf, f)


if __name__ == "__main__":
    main()
