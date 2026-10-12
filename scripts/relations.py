# %% [markdown]
# ## Stage 2 — Chemical-induced-disease relation extraction on BC5CDR
#
# Every (Chemical MeSH id, Disease MeSH id) pair that co-occurs in an abstract is a candidate;
# the label is 1 when the pair is a gold CID relation. Entities are the gold BC5CDR mentions
# (the official CID task setting). Scored with pair-level precision / recall / F1 on the test split.
#
# Models: same-sentence co-occurrence, abstract co-occurrence, TF-IDF + logistic regression,
# and PubMedBERT fine-tuned as an entity-marker pair classifier.
# The decision threshold is tuned on the development split only.

import math
import os
import random
import time

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import MODEL_DIR, PROC_DIR, PUBMEDBERT, RESULTS_DIR, read_json, write_json

MAX_LEN = 256
MARK = {"c": ("[C]", "[/C]"), "d": ("[D]", "[/D]")}


def load(split):
    docs = {d["pmid"]: d for d in read_json(PROC_DIR / f"bc5cdr_{split}.json")}
    rows = read_json(PROC_DIR / f"re_{split}.json")
    return docs, rows


def marked_context(doc, row):
    """Closest (chemical, disease) mention pair -> context span with entity markers."""
    text = doc["text"]
    best = min(((abs(c[1] - d[1]), c, d) for c in row["chem_mentions"] for d in row["dis_mentions"]),
               key=lambda x: x[0])
    _, c, d = best
    spans = sorted([(c[1], c[2], "c"), (d[1], d[2], "d")])
    if spans[0][1] > spans[1][0]:  # overlapping mentions: keep as plain text
        return text[spans[0][0]:spans[1][1]], best
    lo, hi = spans[0][0], spans[1][1]
    # extend to sentence boundaries when both mentions are in one sentence
    for s, e in doc["sentences"]:
        if s <= lo and hi <= e:
            lo, hi = s, e
            break
    out, pos = [], lo
    for s, e, kind in spans:
        out.append(text[pos:s])
        out.append(f"{MARK[kind][0]} {text[s:e]} {MARK[kind][1]}")
        pos = e
    out.append(text[pos:hi])
    return "".join(out), best


def prf(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return {"precision": float(pr), "recall": float(rc), "f1": float(f1)}


def best_threshold(y, probs):
    best = (0.5, -1)
    for t in np.arange(0.05, 0.95, 0.01):
        f = prf(y, (probs >= t).astype(int))["f1"]
        if f > best[1]:
            best = (float(t), f)
    return best[0]


class PairData(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tok):
        self.enc = tok(texts, truncation=True, max_length=MAX_LEN)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i]}


def batches(ds, bs, shuffle, rng):
    idx = sorted(range(len(ds)), key=lambda i: len(ds.enc["input_ids"][i]))
    bl = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        rng.shuffle(bl)
    return bl


def collate(items, pad_id):
    n = max(len(x["input_ids"]) for x in items)
    return {"input_ids": torch.tensor([x["input_ids"] + [pad_id] * (n - len(x["input_ids"])) for x in items]),
            "attention_mask": torch.tensor([x["attention_mask"] + [0] * (n - len(x["attention_mask"])) for x in items]),
            "labels": torch.tensor([x["labels"] for x in items])}


@torch.no_grad()
def predict_probs(model, ds, pad_id, bs=64):
    model.eval()
    out = np.zeros(len(ds))
    for b in batches(ds, bs, False, None):
        batch = collate([ds[i] for i in b], pad_id)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        out[b] = torch.softmax(logits, -1)[:, 1].numpy()
    return out


def main(epochs=int(os.environ.get("RE_EPOCHS", 3)), lr=3e-5, bs=16):
    torch.manual_seed(13); random.seed(13); np.random.seed(13)
    torch.set_num_threads(16)
    data = {}
    for split in ("train", "dev", "test"):
        docs, rows = load(split)
        texts = [marked_context(docs[r["pmid"]], r)[0] for r in rows]
        if os.environ.get("RE_LIMIT"):
            rows = rows[:int(os.environ["RE_LIMIT"])]
        data[split] = {"rows": rows, "texts": texts, "y": np.array([r["label"] for r in rows])}
    results = {}

    # --- co-occurrence baselines (no training)
    ytest = data["test"]["y"]
    same = np.array([int(r["same_sentence"]) for r in data["test"]["rows"]])
    results["Same-sentence co-occurrence"] = prf(ytest, same)
    results["Abstract co-occurrence (all pairs)"] = prf(ytest, np.ones_like(ytest))
    print({k: v for k, v in results.items()})

    # --- TF-IDF + logistic regression (on text with markers; markers kept as tokens)
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"[A-Za-z\[\]/]+")
    Xtr = vec.fit_transform(data["train"]["texts"])
    lr_model = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
    lr_model.fit(Xtr, data["train"]["y"])
    thr = best_threshold(data["dev"]["y"], lr_model.predict_proba(vec.transform(data["dev"]["texts"]))[:, 1])
    pt = lr_model.predict_proba(vec.transform(data["test"]["texts"]))[:, 1]
    results["TF-IDF + logistic regression"] = {**prf(ytest, (pt >= thr).astype(int)), "threshold": thr,
                                                "train_seconds": round(time.time() - t0, 1)}
    print(results["TF-IDF + logistic regression"])

    # --- PubMedBERT pair classifier
    tok = AutoTokenizer.from_pretrained(PUBMEDBERT)
    tok.add_tokens(["[C]", "[/C]", "[D]", "[/D]"])
    model = AutoModelForSequenceClassification.from_pretrained(PUBMEDBERT, num_labels=2)
    model.resize_token_embeddings(len(tok))
    ds = {s: PairData(data[s]["texts"], data[s]["y"].tolist(), tok) for s in data}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = math.ceil(len(ds["train"]) / bs) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    rng, step, t0, history = random.Random(13), 0, time.time(), []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for b in batches(ds["train"], bs, True, rng):
            batch = collate([ds["train"][i] for i in b], tok.pad_token_id)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item(); step += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"step {step}/{steps} loss={loss.item():.4f} eta={(steps-step)*el/step/60:.1f}m", flush=True)
        pd_ = predict_probs(model, ds["dev"], tok.pad_token_id)
        thr = best_threshold(data["dev"]["y"], pd_)
        history.append({"epoch": ep + 1, "train_loss": tot / math.ceil(len(ds["train"]) / bs),
                        "dev_f1": prf(data["dev"]["y"], (pd_ >= thr).astype(int))["f1"], "threshold": thr})
        print(history[-1], flush=True)
    train_secs = time.time() - t0
    pt = predict_probs(model, ds["test"], tok.pad_token_id)
    results["PubMedBERT pair classifier"] = {**prf(ytest, (pt >= thr).astype(int)), "threshold": thr,
                                              "train_seconds": round(train_secs), "history": history,
                                              "epochs": epochs, "lr": lr, "batch_size": bs, "max_len": MAX_LEN}
    print(results["PubMedBERT pair classifier"])

    # same-sentence vs cross-sentence breakdown for the best model
    ss = np.array([r["same_sentence"] for r in data["test"]["rows"]])
    pred = (pt >= thr).astype(int)
    results["PubMedBERT pair classifier"]["same_sentence_f1"] = prf(ytest[ss], pred[ss])["f1"]
    results["PubMedBERT pair classifier"]["cross_sentence_f1"] = prf(ytest[~ss], pred[~ss])["f1"]

    write_json(RESULTS_DIR / "re_results.json", results)
    write_json(RESULTS_DIR / "re_test_probs.json",
               [{"pmid": r["pmid"], "chem_id": r["chem_id"], "dis_id": r["dis_id"], "chem": r["chem_name"],
                 "disease": r["dis_name"], "prob": float(p), "label": r["label"],
                 "context": data["test"]["texts"][i]} for i, (r, p) in enumerate(zip(data["test"]["rows"], pt))],
               indent=None)
    out = MODEL_DIR / "re_pubmedbert"
    model.save_pretrained(out); tok.save_pretrained(out)


if __name__ == "__main__":
    main()
