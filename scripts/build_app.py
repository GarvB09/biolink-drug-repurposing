# Builds the single-file query app (app/index.html) from the real pipeline outputs:
#   results/re_test_probs.json              relation-extraction predictions on the 500 BC5CDR test abstracts
#   results/re_results.json, ner_*_results.json   evaluation metrics
# The page shell (CSS + vis-network library) is app/_base.html.
import json
from collections import Counter

from common import APP_DIR, RESULTS_DIR, read_json


def load_optional(name):
    p = RESULTS_DIR / name
    return read_json(p) if p.exists() else None


def main():
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
    re_all = load_optional("re_results.json")
    re_res = {k: {x: v[x] for x in ("precision", "recall", "f1")} for k, v in re_all.items()}
    thr = re_all["PubMedBERT pair classifier"]["threshold"]

    # relations the PubMedBERT classifier extracts from the test abstracts
    rel = [{"c": p["chem"], "d": p["disease"], "p": round(p["prob"], 3), "pmid": p["pmid"], "g": bool(p["label"]),
            "t": p["context"][:420]} for p in read_json(RESULTS_DIR / "re_test_probs.json") if p["prob"] >= thr]
    rel.sort(key=lambda x: -x["p"])
    freq = Counter()
    for r in rel:
        freq[r["c"]] += 1
        freq[r["d"]] += 1
    bundle = {"relations": rel, "names": sorted(freq), "chips": [n for n, _ in freq.most_common(8)],
              "ner": ner, "re": re_res, "threshold": thr}
    base = (APP_DIR / "_base.html").read_text(encoding="utf-8")
    (APP_DIR / "index.html").write_text(base + BODY.replace("__DATA__", json.dumps(bundle, ensure_ascii=False)),
                                        encoding="utf-8")
    print(f"Wrote app/index.html: {len(rel)} extracted relations, {len(freq)} entity names")


BODY = r"""<body>
<header>
  <div class="brandrow">
    <div class="logo-badge">🧬</div>
    <div>
      <h1>BioLink</h1>
      <p>Search a disease or chemical — see the chemical-induced-disease relations extracted from PubMed abstracts</p>
    </div>
  </div>
</header>

<div class="search-wrap">
  <div class="search-box">
    <input type="text" id="search-input" list="name-list" placeholder="Enter a disease or chemical, e.g. hypertension, cisplatin, nephrotoxicity…">
    <datalist id="name-list"></datalist>
    <button id="search-btn">Search</button>
  </div>
  <div class="chips" id="quick-chips"></div>
</div>

<main>
  <div class="results active" id="results">
    <div class="disease-header">
      <h2 id="q-name"></h2>
      <div class="meta" id="q-meta"></div>
    </div>
    <div class="grid">
      <div>
        <div class="card">
          <h3><span class="icn">📄</span>Extracted relations with evidence</h3>
          <div id="evidence-list"></div>
          <div class="caveat">Pairs predicted by the fine-tuned PubMedBERT relation classifier on the 500 BC5CDR test abstracts (decision threshold tuned on the development split). "Matches gold" means the pair is also in the human annotation; the others are model predictions that were not annotated. These are adverse-effect associations, not treatment advice.</div>
        </div>
      </div>
      <div>
        <div class="card">
          <h3><span class="icn">🕸</span>Extracted relation graph</h3>
          <div id="network"></div>
          <div class="legend">
            <span><i class="dot" style="background:#F2A93B"></i>Searched term</span>
            <span><i class="dot" style="background:#0F4C4A"></i>Matches gold annotation</span>
            <span><i class="dot" style="background:#2C8C86"></i>Predicted only</span>
          </div>
        </div>
        <div class="card">
          <h3><span class="icn">📊</span>Evaluation (held-out test data)</h3>
          <div class="tabs" id="eval-tabs">
            <div class="tab active" data-t="re">Relations</div>
            <div class="tab" data-t="ner">NER</div>
          </div>
          <div id="metrics"></div>
        </div>
      </div>
    </div>
  </div>
</main>

<footer>
  <div><strong>Methodology:</strong> abstracts from the BC5CDR corpus (Li et al., 2016) are split into sentences and tokens; chemical and disease mentions are found by NER (dictionary, CRF, BERT, PubMedBERT); each chemical–disease pair is classified as chemical-induced-disease or not. Every score shown is computed on the official held-out test split. Code and results are in the project folder.</div>
</footer>

<script>
const BIOLINK_DATA = __DATA__;
</script>
<script>
(function(){
  const D = BIOLINK_DATA;
  const dl = document.getElementById("name-list");
  D.names.forEach(n => { const o = document.createElement("option"); o.value = n; dl.appendChild(o); });
  const chipsWrap = document.getElementById("quick-chips");
  D.chips.forEach(name => {
    const c = document.createElement("div"); c.className = "chip"; c.textContent = name;
    c.onclick = () => { document.getElementById("search-input").value = name; runSearch(); };
    chipsWrap.appendChild(c);
  });
  let network = null, evalTab = "re";
  const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
  const pct = v => (v*100).toFixed(1) + "%";

  function renderEvidence(items){
    const list = document.getElementById("evidence-list"); list.innerHTML = "";
    items.slice(0, 12).forEach(t => {
      const div = document.createElement("div"); div.className = "evidence-item";
      const badge = t.g ? '<span class="pill" style="font-size:10px;padding:2px 8px;background:#F2A93B;color:#082E2C">matches gold</span>' : '';
      div.innerHTML = `<div class="drugname">${esc(t.c)} → ${esc(t.d)} <span style="font-weight:400;color:#5C7371">(p=${t.p})</span> ${badge}</div>
        <div class="sent">"${esc(t.t.replace(/\[\/?[CD]\]\s?/g, ""))}"</div><a href="https://pubmed.ncbi.nlm.nih.gov/${t.pmid}/" target="_blank" rel="noopener">PMID ${t.pmid} →</a>`;
      list.appendChild(div);
    });
    if (items.length > 12){ const m = document.createElement("div"); m.className = "caveat"; m.textContent = `Showing the 12 most confident of ${items.length} extracted relations.`; list.appendChild(m); }
  }
  function table(rows, cols){
    return `<table style="width:100%;border-collapse:collapse;font-size:12px"><thead><tr>${cols.map(c => `<th style="text-align:left;padding:5px 4px;border-bottom:1px solid #D8E4E2;color:#5C7371">${c}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.map((c,i) => `<td style="padding:5px 4px;border-bottom:1px solid #EEF3F2;${i?'':'font-weight:700'}">${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }
  function renderMetrics(){
    const wrap = document.getElementById("metrics");
    const src = evalTab === "re" ? D.re : D.ner;
    const rows = Object.entries(src).map(([k,v]) => [k, pct(v.precision), pct(v.recall), pct(v.f1)]);
    wrap.innerHTML = table(rows, ["Model","P","R","F1"]) + (evalTab === "re"
      ? '<div class="caveat">BC5CDR chemical-induced-disease relation classification, test split (gold entities, pair level).</div>'
      : '<div class="caveat">BC5CDR chemical + disease NER, strict entity-level scores on the test split.</div>');
  }
  document.querySelectorAll("#eval-tabs .tab").forEach(t => { t.onclick = () => {
    document.querySelectorAll("#eval-tabs .tab").forEach(x => x.classList.remove("active"));
    t.classList.add("active"); evalTab = t.dataset.t; renderMetrics(); }; });

  function renderGraph(name, items){
    const nodes = [{ id: "q", label: name, color: "#F2A93B", font:{color:"#082E2C", size:13}, shape:"dot", size:22 }], edges = [], seen = {};
    items.slice(0, 14).forEach((t, i) => {
      const other = t.c.toLowerCase() === name ? t.d : t.c;
      if (seen[other]) return; seen[other] = true;
      const col = t.g ? "#0F4C4A" : "#2C8C86";
      nodes.push({ id: "n"+i, label: other, color: col, font:{color:"#fff", size:11}, shape:"dot", size:14 });
      edges.push({ from: "q", to: "n"+i, color:{color:col}, dashes: !t.g, width: t.g ? 2 : 1.5 });
    });
    if (network) network.destroy();
    network = new vis.Network(document.getElementById("network"), { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) },
      { physics:{ stabilization:true, barnesHut:{ springLength:110 } }, interaction:{ hover:true, zoomView:true } });
  }
  function runSearch(){
    const name = document.getElementById("search-input").value.trim().toLowerCase();
    const items = D.relations.filter(t => t.c.toLowerCase() === name || t.d.toLowerCase() === name);
    if (!items.length){ alert("No extracted relations for that term. Pick one from the suggestions."); return; }
    const asChem = items.some(t => t.c.toLowerCase() === name);
    const gold = items.filter(t => t.g).length;
    document.getElementById("q-name").textContent = name;
    document.getElementById("q-meta").textContent = `${asChem ? "chemical" : "disease"} · ${items.length} extracted relation(s), ${gold} matching the gold annotation`;
    renderEvidence(items); renderMetrics(); renderGraph(name, items);
  }
  document.getElementById("search-btn").onclick = runSearch;
  document.getElementById("search-input").addEventListener("keydown", e => { if (e.key === "Enter") runSearch(); });
  document.getElementById("search-input").value = D.chips[0];
  runSearch();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
