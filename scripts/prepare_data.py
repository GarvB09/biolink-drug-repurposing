# %% [markdown]
# ## Stage 0 — Dataset preparation
#
# 1. **BC5CDR** (BioCreative V Chemical–Disease Relation corpus; Li et al., 2016): 1,500 PubMed
#    abstracts with gold Chemical/Disease mentions (MeSH-normalised) and abstract-level
#    Chemical-Induced-Disease (CID) relations, pre-split 500/500/500 into train/dev/test.
#    -> sentence-level BIO data for NER, and (chemical, disease) candidate pairs for RE.

from collections import Counter, defaultdict

from common import (BC5CDR_FILES, PROC_DIR, RESULTS_DIR, bio_tags, parse_pubtator, split_sentences,
                    tokenize, write_json)



def prepare_bc5cdr():
    stats = {}
    for split, path in BC5CDR_FILES.items():
        docs = parse_pubtator(path)
        ner_rows, re_rows = [], []
        n_missed_rel = 0
        for doc in docs:
            sents = split_sentences(doc)
            doc["sentences"] = sents
            for si, (s, e) in enumerate(sents):
                toks = tokenize(doc["text"][s:e], offset=s)
                ents = [x for x in doc["entities"] if x["start"] >= s and x["end"] <= e]
                ner_rows.append({"pmid": doc["pmid"], "sent": si, "tokens": [t[0] for t in toks],
                                 "offsets": [[t[1], t[2]] for t in toks], "tags": bio_tags(toks, ents)})

            # abstract-level candidate pairs over normalised IDs
            def sent_of(ent):
                return next(i for i, (s, e) in enumerate(sents) if s <= ent["start"] < e)

            mentions = defaultdict(list)  # id -> [(sent_idx, start, end, text)]
            for ent in doc["entities"]:
                for cid in ent["ids"]:
                    mentions[(ent["type"], cid)].append((sent_of(ent), ent["start"], ent["end"], ent["text"]))
            chems = [k[1] for k in mentions if k[0] == "Chemical"]
            dises = [k[1] for k in mentions if k[0] == "Disease"]
            gold = set(doc["relations"])
            for c in chems:
                for d in dises:
                    cm, dm = mentions[("Chemical", c)], mentions[("Disease", d)]
                    re_rows.append({
                        "pmid": doc["pmid"], "chem_id": c, "dis_id": d,
                        "chem_mentions": cm, "dis_mentions": dm,
                        "chem_name": Counter(m[3].lower() for m in cm).most_common(1)[0][0],
                        "dis_name": Counter(m[3].lower() for m in dm).most_common(1)[0][0],
                        "same_sentence": bool({m[0] for m in cm} & {m[0] for m in dm}),
                        "label": int((c, d) in gold),
                    })
            n_missed_rel += len(gold - {(c, d) for c in chems for d in dises})

        write_json(PROC_DIR / f"bc5cdr_{split}.json", docs, indent=None)
        write_json(PROC_DIR / f"ner_{split}.json", ner_rows, indent=None)
        write_json(PROC_DIR / f"re_{split}.json", re_rows, indent=None)

        ent_types = Counter(e["type"] for d in docs for e in d["entities"])
        uniq = {t: len({i for d in docs for e in d["entities"] if e["type"] == t for i in e["ids"]})
                for t in ("Chemical", "Disease")}
        stats[split] = {
            "abstracts": len(docs), "sentences": len(ner_rows),
            "tokens": sum(len(r["tokens"]) for r in ner_rows),
            "chemical_mentions": ent_types["Chemical"], "disease_mentions": ent_types["Disease"],
            "unique_chemical_ids": uniq["Chemical"], "unique_disease_ids": uniq["Disease"],
            "cid_relations": sum(len(d["relations"]) for d in docs),
            "re_candidates": len(re_rows), "re_positive": sum(r["label"] for r in re_rows),
            "re_positive_same_sentence": sum(r["label"] for r in re_rows if r["same_sentence"]),
            "relations_without_candidate": n_missed_rel,
            "avg_tokens_per_abstract": round(sum(len(r["tokens"]) for r in ner_rows) / len(docs), 1),
        }
        print(split, stats[split])
    return stats


if __name__ == "__main__":
    write_json(RESULTS_DIR / "dataset_stats.json", {"bc5cdr": prepare_bc5cdr()})
