# Builds the single-file query app (app/index.html) from the real pipeline outputs:
#   results/kg_predictions.json   Hetionet link-prediction candidates per disease
#   results/linkpred_results.json, ner_*_results.json, re_results.json   evaluation metrics
#   results/re_test_probs.json    relation-extraction predictions on the BC5CDR test abstracts
# The page shell (CSS + vis-network library) is app/_base.html.
import json

from common import APP_DIR, RESULTS_DIR, read_json

PREFERRED_CHIPS = ["multiple sclerosis", "rheumatoid arthritis", "type 2 diabetes mellitus", "breast cancer",
                   "hypertension", "asthma", "epilepsy syndrome", "schizophrenia"]


def load_optional(name):
    p = RESULTS_DIR / name
    return read_json(p) if p.exists() else None


def main():
    kg = read_json(RESULTS_DIR / "kg_predictions.json")
    lp = read_json(RESULTS_DIR / "linkpred_results.json")
    for v in lp.values():
        v.pop("history", None)
    ner = {}
    for label, f in (("Dictionary", "ner_classic_results.json"), ("BERT-base", "ner_bert-base_results.json"),
                     ("PubMedBERT", "ner_pubmedbert_results.json")):
        d = load_optional(f)
        if not d:
            continue
        if f == "ner_classic_results.json":
            for k, v in d.items():
                ner[k] = {x: v[x] for x in ("precision", "recall", "f1")}
        else:
            ner[label + " (fine-tuned)"] = {x: d[x] for x in ("precision", "recall", "f1")}
    re_res = load_optional("re_results.json") or {}
    re_res = {k: {x: v[x] for x in ("precision", "recall", "f1")} for k, v in re_res.items()}

    # literature evidence: pairs the PubMedBERT relation classifier predicts as chemical-induced disease
    probs = load_optional("re_test_probs.json") or []
    thr = (load_optional("re_results.json") or {}).get("PubMedBERT pair classifier", {}).get("threshold", 0.5)
    lit = [{"chem": p["chem"], "disease": p["disease"], "prob": round(p["prob"], 3), "pmid": p["pmid"],
            "gold": bool(p["label"]), "context": p["context"][:420]}
           for p in probs if p["prob"] >= thr]
    lit.sort(key=lambda x: -x["prob"])

    names = sorted(kg["predictions"])
    chips = [n for n in PREFERRED_CHIPS if n in kg["predictions"]]
    if len(chips) < 5:
        by_known = sorted(names, key=lambda n: -len(kg["predictions"][n]["known_treatments"]))
        chips += [n for n in by_known if n not in chips][: 6 - len(chips)]

    def has_lit(n):
        k = n.lower()
        return any(l["disease"].lower() == k or (len(l["disease"]) >= 6 and l["disease"].lower() in k) or (len(k) >= 6 and k in l["disease"].lower()) for l in lit)
    chips.sort(key=lambda n: not has_lit(n))
    bundle = {"model": kg["model"], "predictions": kg["predictions"], "linkpred": lp, "ner": ner, "re": re_res,
              "literature": lit, "chips": chips}
    base = (APP_DIR / "_base.html").read_text(encoding="utf-8")
    html = base + BODY.replace("__DATA__", json.dumps(bundle, ensure_ascii=False))
    (APP_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote app/index.html ({len(html)//1024} KB): {len(names)} diseases, {len(lit)} literature pairs")


BODY = r"""<body>
<header>
  <div class="brandrow">
    <div class="logo-badge">🧬</div>
    <div>
      <h1>BioLink</h1>
      <p>Drug-repurposing query interface — ranked candidate drugs for a disease, with evaluation and literature evidence</p>
    </div>
  </div>
</header>

<div class="search-wrap">
  <div class="search-box">
    <input type="text" id="search-input" list="disease-list" placeholder="Enter a disease from Hetionet, e.g. multiple sclerosis, rheumatoid arthritis…">
    <datalist id="disease-list"></datalist>
    <button id="search-btn">Search</button>
  </div>
  <div class="chips" id="quick-chips"></div>
</div>

<main>
  <div class="results active" id="results">
    <div class="disease-header">
      <h2 id="disease-name"></h2>
      <div class="meta" id="disease-meta"></div>
    </div>
    <div class="grid">
      <div>
        <div class="card">
          <h3><span class="icn">✓</span>Known treatments in the graph (training edges)</h3>
          <div class="pill-row" id="known-treatments"></div>
        </div>
        <div class="card">
          <h3><span class="icn">✨</span><span>Predicted candidate drugs (<span id="model-name"></span>)</span></h3>
          <div id="pred-list"></div>
          <div class="caveat">Ranked by link-prediction score over Hetionet compounds, excluding known treatments. Entries marked "held-out true" are real treats edges that were hidden during training, so they show the model recovering true links. Predictions are research hypotheses, not clinical recommendations.</div>
        </div>
        <div class="card" id="evidence-card">
          <h3><span class="icn">📄</span>Literature associations (BC5CDR test abstracts)</h3>
          <div id="evidence-list"></div>
          <div class="caveat">Chemical-induced-disease pairs predicted by the fine-tuned PubMedBERT classifier on the 500 BC5CDR test abstracts. These are adverse-effect associations, shown as supporting context, not as treatment evidence.</div>
        </div>
      </div>
      <div>
        <div class="card">
          <h3><span class="icn">🕸</span>Local graph neighbourhood</h3>
          <div id="network"></div>
          <div class="legend">
            <span><i class="dot" style="background:#F2A93B"></i>Disease</span>
            <span><i class="dot" style="background:#0F4C4A"></i>Known treatment</span>
            <span><i class="dot" style="background:#2C8C86"></i>Predicted candidate</span>
          </div>
        </div>
        <div class="card">
          <h3><span class="icn">📊</span>Evaluation (held-out test data)</h3>
          <div class="tabs" id="eval-tabs">
            <div class="tab active" data-t="lp">Link prediction</div>
            <div class="tab" data-t="re">Relations</div>
            <div class="tab" data-t="ner">NER</div>
          </div>
          <div id="metrics"></div>
        </div>
      </div>
    </div>
  </div>
</main>

<footer>
  <div><strong>Methodology:</strong> NER and relation extraction are trained and tested on the BC5CDR corpus (Li et al., 2016, 1,500 PubMed abstracts). Link prediction (TransE / DistMult / RotatE, implemented in PyTorch) is trained on a Hetionet v1.0 subset (Himmelstein et al., 2017) with 10% of compound-treats-disease edges held out for validation and 10% for testing. Code and results are in the project folder.</div>
</footer>

<script>
const BIOLINK_DATA = __DATA__;
</script>
<script>
(function(){
  const D = BIOLINK_DATA;
  const names = Object.keys(D.predictions).sort();
  const dl = document.getElementById("disease-list");
  names.forEach(n => { const o = document.createElement("option"); o.value = n; dl.appendChild(o); });
  const chipsWrap = document.getElementById("quick-chips");
  D.chips.forEach(name => {
    const c = document.createElement("div"); c.className = "chip"; c.textContent = name;
    c.onclick = () => { document.getElementById("search-input").value = name; runSearch(); };
    chipsWrap.appendChild(c);
  });
  document.getElementById("model-name").textContent = D.model;

  let network = null, evalTab = "lp";
  const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
  const pct = v => (v*100).toFixed(1) + "%";

  function renderKnown(rec){
    const wrap = document.getElementById("known-treatments"); wrap.innerHTML = "";
    if (!rec.known_treatments.length){ wrap.innerHTML = '<span class="pill empty">No treats edges for this disease in the training graph.</span>'; return; }
    rec.known_treatments.forEach(n => { const p = document.createElement("span"); p.className = "pill"; p.textContent = n; wrap.appendChild(p); });
  }
  function novel(rec){ return rec.top.filter(x => !x.known).slice(0, 10); }
  function renderPreds(rec){
    const wrap = document.getElementById("pred-list"); wrap.innerHTML = "";
    const preds = novel(rec);
    const max = Math.max(...preds.map(p => Math.abs(p.score)), 1e-9);
    preds.forEach((p, i) => {
      const row = document.createElement("div"); row.className = "pred-item";
      const w = Math.max(10, 100 - (Math.abs(p.score) / max) * 60);
      row.innerHTML = `<span class="pred-rank">${i+1}</span><span class="pred-name">${esc(p.drug)}${p.held_out_true ? ' <span class="pill" style="font-size:10px;padding:2px 8px;background:#F2A93B;color:#082E2C">held-out true</span>' : ''}</span>
        <span class="pred-bar-wrap"><span class="pred-bar" style="width:${w}%"></span></span><span class="pred-score">${p.score}</span>`;
      wrap.appendChild(row);
    });
  }
  function renderEvidence(name){
    const card = document.getElementById("evidence-card"), list = document.getElementById("evidence-list");
    const key = name.toLowerCase();
    const items = D.literature.filter(t => { const d = t.disease.toLowerCase(); return d === key || (d.length >= 6 && key.includes(d)) || (key.length >= 6 && d.includes(key)); }).slice(0, 6);
    if (!items.length){ card.style.display = "none"; return; }
    card.style.display = ""; list.innerHTML = "";
    items.forEach(t => {
      const div = document.createElement("div"); div.className = "evidence-item";
      div.innerHTML = `<div class="drugname">${esc(t.chem)} → ${esc(t.disease)} <span style="font-weight:400;color:#5C7371">(p=${t.prob}${t.gold ? ", matches gold annotation" : ""})</span></div>
        <div class="sent">"${esc(t.context.replace(/\[\/?[CD]\]\s?/g, ""))}"</div><a href="https://pubmed.ncbi.nlm.nih.gov/${t.pmid}/" target="_blank" rel="noopener">PMID ${t.pmid} →</a>`;
      list.appendChild(div);
    });
  }
  function table(rows, cols){
    return `<table style="width:100%;border-collapse:collapse;font-size:12px"><thead><tr>${cols.map(c => `<th style="text-align:left;padding:5px 4px;border-bottom:1px solid #D8E4E2;color:#5C7371">${c}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.map((c,i) => `<td style="padding:5px 4px;border-bottom:1px solid #EEF3F2;${i?'':'font-weight:700'}">${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }
  function renderMetrics(){
    const wrap = document.getElementById("metrics");
    if (evalTab === "lp"){
      const rows = Object.entries(D.linkpred).map(([k,v]) => [k, v.mrr.toFixed(3), v.mean_rank.toFixed(1), pct(v["hits@1"]), pct(v["hits@3"]), pct(v["hits@10"])]);
      wrap.innerHTML = table(rows, ["Model","MRR","MR","H@1","H@3","H@10"]) + `<div class="caveat">Filtered ranking of 136 diseases for each of ${D.linkpred[Object.keys(D.linkpred)[0]].n_test} held-out treats edges.</div>`;
    } else if (evalTab === "re"){
      const rows = Object.entries(D.re).map(([k,v]) => [k, pct(v.precision), pct(v.recall), pct(v.f1)]);
      wrap.innerHTML = table(rows, ["Model","P","R","F1"]) + '<div class="caveat">BC5CDR CID relation classification, test split (gold entities).</div>';
    } else {
      const rows = Object.entries(D.ner).map(([k,v]) => [k, pct(v.precision), pct(v.recall), pct(v.f1)]);
      wrap.innerHTML = table(rows, ["Model","P","R","F1"]) + '<div class="caveat">BC5CDR chemical + disease NER, strict entity-level scores on the test split.</div>';
    }
  }
  document.querySelectorAll("#eval-tabs .tab").forEach(t => { t.onclick = () => {
    document.querySelectorAll("#eval-tabs .tab").forEach(x => x.classList.remove("active"));
    t.classList.add("active"); evalTab = t.dataset.t; renderMetrics(); }; });

  function renderGraph(name, rec){
    const nodes = [{ id: "d", label: name, color: "#F2A93B", font:{color:"#082E2C", size:13}, shape:"dot", size:22 }], edges = [];
    rec.known_treatments.slice(0, 8).forEach((n, i) => { nodes.push({ id: "k"+i, label: n, color: "#0F4C4A", font:{color:"#fff", size:11}, shape:"dot", size:15 });
      edges.push({ from: "d", to: "k"+i, color:{color:"#0F4C4A"}, width: 2 }); });
    novel(rec).slice(0, 6).forEach((p, i) => { nodes.push({ id: "p"+i, label: p.drug, color: "#2C8C86", font:{color:"#fff", size:11}, shape:"dot", size:14 });
      edges.push({ from: "d", to: "p"+i, color:{color:"#2C8C86"}, dashes: true, width: 1.5 }); });
    if (network) network.destroy();
    network = new vis.Network(document.getElementById("network"), { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) },
      { physics:{ stabilization:true, barnesHut:{ springLength:110 } }, interaction:{ hover:true, zoomView:true } });
  }
  function runSearch(){
    const raw = document.getElementById("search-input").value.trim().toLowerCase();
    const name = names.find(n => n.toLowerCase() === raw);
    if (!name){ alert("That disease isn't in the Hetionet subset. Pick one from the suggestions."); return; }
    const rec = D.predictions[name];
    document.getElementById("disease-name").textContent = name;
    document.getElementById("disease-meta").textContent = `${rec.known_treatments.length} known treatment edge(s) in training · Hetionet node ${rec.id}`;
    renderKnown(rec); renderPreds(rec); renderEvidence(name); renderMetrics(); renderGraph(name, rec);
  }
  document.getElementById("search-btn").onclick = runSearch;
  document.getElementById("search-input").addEventListener("keydown", e => { if (e.key === "Enter") runSearch(); });
  document.getElementById("search-input").value = D.chips[0] || names[0];
  runSearch();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
