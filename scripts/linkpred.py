# %% [markdown]
# ## Stage 3 — Knowledge-graph link prediction on Hetionet (TransE / DistMult / RotatE)
#
# Task: given a compound, rank all 136 diseases by plausibility of a "treats" edge.
# Models are implemented from scratch in PyTorch and trained on the Hetionet subset built in
# prepare_data.py (234k triples, 21.6k entities); 76 treats edges are held out for validation
# and 76 for test. Metrics: filtered Mean Rank, MRR and Hits@K over the disease candidates.
# Baselines: random ranking and popularity (training-set treats-degree of each disease).

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MODEL_DIR, PROC_DIR, RESULTS_DIR, read_json, write_json

torch.manual_seed(13)
np.random.seed(13)
torch.set_num_threads(16)

DIM = 128
NEG = 32
BATCH = 4096
LR = 1e-3
EPOCHS = int(os.environ.get("KGE_EPOCHS", 150))
EVAL_EVERY = int(os.environ.get("KGE_EVAL_EVERY", 25))
MARGIN = 9.0  # RotatE / TransE fixed margin gamma
ADV_T = 1.0   # self-adversarial temperature


def load_triples(name):
    return [tuple(l.split("\t")) for l in (PROC_DIR / "kg" / f"{name}.tsv").read_text(encoding="utf-8").splitlines() if l]


class KGE(nn.Module):
    def __init__(self, kind, n_ent, n_rel, dim):
        super().__init__()
        self.kind, self.dim = kind, dim
        scale = (MARGIN + 2.0) / dim
        if kind == "rotate":
            self.ent = nn.Parameter(torch.empty(n_ent, 2 * dim).uniform_(-scale, scale))
            self.rel = nn.Parameter(torch.empty(n_rel, dim).uniform_(-np.pi, np.pi))
        else:
            self.ent = nn.Parameter(torch.empty(n_ent, dim).uniform_(-scale, scale))
            self.rel = nn.Parameter(torch.empty(n_rel, dim).uniform_(-scale, scale))

    def score(self, h, r, t):
        """h, t: entity index tensors (broadcastable); r: relation index tensor. Higher = better."""
        H, T, R = self.ent[h], self.ent[t], self.rel[r]
        if self.kind == "transe":
            return MARGIN - torch.norm(H + R - T, p=1, dim=-1)
        if self.kind == "distmult":
            return (H * R * T).sum(-1)
        hr, hi = torch.chunk(H, 2, dim=-1)
        tr, ti = torch.chunk(T, 2, dim=-1)
        c, s = torch.cos(R), torch.sin(R)
        dr, di = hr * c - hi * s - tr, hr * s + hi * c - ti
        return MARGIN - torch.sqrt(dr ** 2 + di ** 2 + 1e-9).sum(-1)


def train(kind, train_ids, n_ent, n_rel, valid_fn):
    model = KGE(kind, n_ent, n_rel, DIM)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    tr = torch.tensor(train_ids)
    n, history, best = len(tr), [], (-1, None)
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            b = tr[perm[i:i + BATCH]]
            h, r, t = b[:, 0], b[:, 1], b[:, 2]
            neg = torch.randint(0, n_ent, (len(b), NEG))
            corrupt_head = torch.rand(len(b), 1) < 0.5
            nh = torch.where(corrupt_head, neg, h[:, None])
            nt = torch.where(corrupt_head, t[:, None], neg)
            pos = F.logsigmoid(model.score(h, r, t)) if kind != "distmult" else None
            ns = model.score(nh, r[:, None], nt)
            if kind == "distmult":  # logistic loss + L3-free small L2 for stability
                loss = (F.softplus(-model.score(h, r, t)).mean() + F.softplus(ns).mean()) / 2
                loss = loss + 1e-6 * (model.ent[h].pow(2).sum() + model.ent[t].pow(2).sum()) / len(b)
            else:  # self-adversarial negative sampling (Sun et al., 2019)
                w = torch.softmax(ADV_T * ns.detach(), dim=1)
                loss = -(pos.mean() + (w * F.logsigmoid(-ns)).sum(1).mean()) / 2
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            m = valid_fn(model)
            history.append({"epoch": ep, "loss": tot / n, "valid_mrr": m["mrr"]})
            print(f"[{kind}] epoch {ep} loss={tot/n:.4f} valid_mrr={m['mrr']:.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)
            if m["mrr"] > best[0]:
                best = (m["mrr"], {k: v.clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    return model, history, best[2], time.time() - t0


@torch.no_grad()
def evaluate(model, test, disease_ids, known_by_head, rel_id, ks=(1, 3, 10)):
    """Filtered tail ranking over all diseases for (compound, treats, ?)."""
    model.eval()
    dis = torch.tensor(disease_ids)
    ranks = []
    for h, r, t in test:
        sc = model.score(torch.full((len(dis),), h), torch.full((len(dis),), r), dis)
        mask = torch.tensor([(d in known_by_head[h]) and d != t for d in disease_ids])
        sc = sc.masked_fill(mask, -1e9)
        target = sc[disease_ids.index(t)]
        ranks.append(int((sc > target).sum().item()) + 1 + int(((sc == target).sum().item() - 1) / 2))
    return summarise(ranks, ks)


def summarise(ranks, ks=(1, 3, 10)):
    r = np.array(ranks, dtype=float)
    return {"mean_rank": float(r.mean()), "mrr": float((1 / r).mean()),
            **{f"hits@{k}": float((r <= k).mean()) for k in ks}, "n_test": len(ranks)}


def main():
    ent_meta = read_json(PROC_DIR / "kg" / "entities.json")
    ents = sorted(ent_meta)
    e2i = {e: i for i, e in enumerate(ents)}
    train_t, valid_t, test_t = (load_triples(s) for s in ("train", "valid", "test"))
    rels = sorted({r for _, r, _ in train_t + valid_t + test_t})
    r2i = {r: i for i, r in enumerate(rels)}
    enc = lambda ts: [(e2i[h], r2i[r], e2i[t]) for h, r, t in ts]
    train_ids, valid_ids, test_ids = enc(train_t), enc(valid_t), enc(test_t)
    treats = r2i["treats"]
    disease_ids = sorted(i for e, i in e2i.items() if ent_meta[e]["kind"] == "Disease")

    known = {}
    for h, r, t in train_ids + valid_ids + test_ids:
        if r == treats:
            known.setdefault(h, set()).add(t)
    known_by_head = {h: known.get(h, set()) for h in {x[0] for x in train_ids + valid_ids + test_ids}}

    rng = np.random.default_rng(13)
    # baselines
    pop = {}
    for h, r, t in train_ids:
        if r == treats:
            pop[t] = pop.get(t, 0) + 1
    results = {}
    for name, scorer in (("Random", lambda h: rng.random(len(disease_ids))),
                         ("Popularity (disease treats-degree)", lambda h: np.array([pop.get(d, 0) + 1e-6 * rng.random() for d in disease_ids]))):
        ranks = []
        for h, r, t in test_ids:
            sc = scorer(h)
            for k, d in enumerate(disease_ids):
                if d in known_by_head[h] and d != t:
                    sc[k] = -1e9
            ranks.append(int((sc > sc[disease_ids.index(t)]).sum()) + 1)
        results[name] = summarise(ranks)
        print(name, results[name])

    valid_fn = lambda m: evaluate(m, valid_ids, disease_ids, known_by_head, treats)
    trained = {}
    for kind, label in (("transe", "TransE"), ("distmult", "DistMult"), ("rotate", "RotatE")):
        model, hist, best_ep, secs = train(kind, train_ids, len(ents), len(rels), valid_fn)
        m = evaluate(model, test_ids, disease_ids, known_by_head, treats)
        results[label] = {**m, "train_seconds": round(secs), "best_epoch": best_ep, "history": hist,
                          "dim": DIM, "negatives": NEG, "batch": BATCH, "lr": LR, "epochs": EPOCHS}
        trained[label] = model
        print(label, m, flush=True)
        torch.save(model.state_dict(), MODEL_DIR / f"kge_{kind}.pt")

    write_json(RESULTS_DIR / "linkpred_results.json", results)

    # ranked candidate drugs for every disease with the best model (by validation MRR)
    best_label = max(trained, key=lambda k: max(h["valid_mrr"] for h in results[k]["history"]))
    model = trained[best_label]
    compounds = [i for e, i in e2i.items() if ent_meta[e]["kind"] == "Compound"]
    comp_t = torch.tensor(compounds)
    train_known = {}
    for h, r, t in train_ids:
        if r == treats:
            train_known.setdefault(t, set()).add(h)
    preds = {}
    with torch.no_grad():
        for d in disease_ids:
            sc = model.score(comp_t, torch.full_like(comp_t, treats), torch.full_like(comp_t, d))
            order = torch.argsort(sc, descending=True).tolist()
            held = {h for h, r, t in valid_ids + test_ids if t == d}
            top = []
            for k in order:
                c = compounds[k]
                top.append({"drug": ent_meta[ents[c]]["name"], "id": ents[c], "score": round(float(sc[k]), 3),
                            "known": c in train_known.get(d, set()), "held_out_true": c in held})
                if sum(1 for x in top if not x["known"]) >= 10 and len(top) >= 15:
                    break
            known_names = sorted(ent_meta[ents[c]]["name"] for c in train_known.get(d, set()))
            preds[ent_meta[ents[d]]["name"]] = {"id": ents[d], "top": top[:15], "known_treatments": known_names}
    write_json(RESULTS_DIR / "kg_predictions.json", {"model": best_label, "predictions": preds}, indent=None)
    print("best model for app:", best_label)


if __name__ == "__main__":
    main()
