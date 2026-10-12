# %% [markdown]
# ## Stage 1b — Transformer NER on BC5CDR (fine-tuned BERT-family token classifiers)
#
# Token classification head (linear layer over the final hidden states) on top of a
# pretrained encoder; IOB2 labels are assigned to the first word-piece of each word and the
# remaining word-pieces are ignored in the loss (-100). Compares a general-domain encoder
# (BERT-base-cased) with biomedical-domain encoders (BioBERT, PubMedBERT) to measure the
# effect of domain-specific pretraining.
#
# Usage: python ner_bert.py <model-key> [--epochs 3] [--limit N]

import argparse
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import BERT_BASE, BIOBERT, MODEL_DIR, NER_LABELS, PROC_DIR, PUBMEDBERT, RESULTS_DIR, read_json, write_json
from ner_classic import entity_scores

MODELS = {"bert-base": BERT_BASE, "biobert": BIOBERT, "pubmedbert": PUBMEDBERT}
LABEL2ID = {l: i for i, l in enumerate(NER_LABELS)}
MAX_LEN = 192


def encode(rows, tok):
    enc = tok([r["tokens"] for r in rows], is_split_into_words=True, truncation=True, max_length=MAX_LEN)
    feats = []
    for i, r in enumerate(rows):
        word_ids = enc.word_ids(i)
        labels, prev = [], None
        for w in word_ids:
            labels.append(-100 if w is None or w == prev else LABEL2ID[r["tags"][w]])
            prev = w
        feats.append({"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
                      "labels": labels, "word_ids": word_ids})
    return feats


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda key, v: torch.tensor([b[key] + [v] * (n - len(b[key])) for b in batch])
    return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", -100)}


def length_batches(feats, bs, shuffle, rng):
    """Bucket by length to cut padding (a large CPU speed-up), shuffle the buckets."""
    idx = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))
    batches = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        rng.shuffle(batches)
    return batches


@torch.no_grad()
def predict(model, feats, rows, pad_id, bs=64):
    model.eval()
    preds = [None] * len(rows)
    for b in length_batches(feats, bs, False, None):
        batch = collate([feats[i] for i in b], pad_id)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        best = logits.argmax(-1).tolist()
        for k, i in enumerate(b):
            tags = ["O"] * len(rows[i]["tokens"])
            prev = None
            for pos, w in enumerate(feats[i]["word_ids"]):
                if w is not None and w != prev:
                    tags[w] = NER_LABELS[best[k][pos]]
                prev = w
            # repair illegal I- after O / other type into B-
            for j, t in enumerate(tags):
                if t.startswith("I-") and (j == 0 or tags[j - 1][2:] != t[2:]):
                    tags[j] = "B-" + t[2:]
            preds[i] = tags
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: train on first N sentences")
    args = ap.parse_args()

    torch.manual_seed(13); random.seed(13); np.random.seed(13)
    torch.set_num_threads(16)
    name = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForTokenClassification.from_pretrained(
        name, num_labels=len(NER_LABELS), id2label=dict(enumerate(NER_LABELS)), label2id=LABEL2ID)

    # following BioBERT / PubMedBERT practice on BC5CDR: train on train+dev with a fixed schedule
    train_rows = read_json(PROC_DIR / "ner_train.json") + read_json(PROC_DIR / "ner_dev.json")
    if args.limit:
        train_rows = train_rows[:args.limit]
    test_rows = read_json(PROC_DIR / "ner_test.json")
    train_f, test_f = encode(train_rows, tok), encode(test_rows, tok)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(train_f) / args.bs) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    rng = random.Random(13)
    history, step, t0 = [], 0, time.time()
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for b in length_batches(train_f, args.bs, True, rng):
            loss = model(**collate([train_f[i] for i in b], tok.pad_token_id)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item(); step += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"step {step}/{steps} loss={loss.item():.4f} elapsed={el/60:.1f}m "
                      f"eta={(steps-step)*el/step/60:.1f}m", flush=True)
        history.append({"epoch": ep + 1, "train_loss": tot / math.ceil(len(train_f) / args.bs)})
        print(history[-1], flush=True)
    train_secs = time.time() - t0

    t0 = time.time()
    preds = predict(model, test_f, test_rows, tok.pad_token_id)
    scores = entity_scores([r["tags"] for r in test_rows], preds)
    scores.update({"train_seconds": round(train_secs), "infer_seconds": round(time.time() - t0),
                   "epochs": args.epochs, "lr": args.lr, "batch_size": args.bs, "max_len": MAX_LEN,
                   "checkpoint": name, "history": history})
    print(scores)
    if not args.limit:
        write_json(RESULTS_DIR / f"ner_{args.model}_results.json", scores)
        write_json(RESULTS_DIR / f"ner_test_predictions_{args.model}.json", preds, indent=None)
        out = MODEL_DIR / f"ner_{args.model}"
        model.save_pretrained(out); tok.save_pretrained(out)


if __name__ == "__main__":
    main()
