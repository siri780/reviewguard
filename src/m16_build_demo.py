"""M16 - Build demo/index.html: a self-contained explorer of 400 sampled held-out test reviews.

Run from the repo root after M1 and M3 (needs results/_m3_features.csv and
results/_shared_split_row_ids.csv):
    python src/m16_build_demo.py
It fits scikit-learn logistic regressions (StandardScaler fit on train, max_iter=100) on the
36 M3 features and on the 32 handcrafted features only, scores the shared test set, and draws
100 reviews from each outcome group (true positive, false positive, false negative, true
negative at threshold 0.5; RandomState(0)). Per-feature contributions are exact for logistic
regression: coefficient x standardized value, in log-odds. The page is an explorer of saved
model outputs, not a live scoring service, and its group sizes are not the real base rates."""
import json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import common

d = pd.read_csv(common.M3_FEATURES_PATH).merge(pd.read_csv(common.SPLIT_PATH), on="row_id")
tr, te = d[d.split == "train"], d[d.split == "test"].copy()
f = [f"f{i}" for i in range(32)]
g = ["deg_rur", "deg_rsr", "deg_rtr", "pagerank"]


def fit(cols):
    sc = StandardScaler().fit(tr[cols])
    return sc, LogisticRegression(max_iter=100).fit(sc.transform(tr[cols]), tr.label)


scF, mF = fit(f + g)
scM, mM = fit(f)
ZF = scF.transform(te[f + g])
te["pF"] = mF.predict_proba(ZF)[:, 1]
te["pM"] = mM.predict_proba(scM.transform(te[f]))[:, 1]
contrib = ZF * mF.coef_[0]
te["pred"] = (te.pF >= 0.5).astype(int)
te["out"] = np.where((te.label == 1) & (te.pred == 1), "TP", np.where((te.label == 0) & (te.pred == 1), "FP",
                     np.where(te.label == 1, "FN", "TN")))
names = f + g
rs = np.random.RandomState(0)
rows = []
for o in ["TP", "FP", "FN", "TN"]:
    idx = te.index[te.out == o]
    for i in rs.choice(idx, size=min(100, len(idx)), replace=False):
        k = te.index.get_loc(i)
        c = contrib[k]
        top = np.argsort(-np.abs(c))[:6]
        rows.append(dict(id=int(te.loc[i, "row_id"]), y=int(te.loc[i, "label"]),
                         pF=round(float(te.loc[i, "pF"]), 3), pM=round(float(te.loc[i, "pM"]), 3), o=o,
                         dr=int(te.loc[i, "deg_rur"]), dt=int(te.loc[i, "deg_rtr"]),
                         top=[[names[j], round(float(c[j]), 2)] for j in top],
                         b=round(float(mF.intercept_[0]), 2), net=round(float(c[32:].sum()), 2)))
rs.shuffle(rows)
auc = [round(roc_auc_score(te.label, te.pF), 4), round(roc_auc_score(te.label, te.pM), 4)]
data = json.dumps(dict(rows=rows, auc=auc, n=len(te)), separators=(",", ":"))

TEMPLATE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ReviewGuard demo</title>
<style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mut:#6b6b66;--card:#fff;--line:#e3e2dc;--acc:#2f5d8a;--pos:#b4472f;--neg:#2f7a5a;--warn:#a8741a}
@media(prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9992;--card:#1f1f1d;--line:#34332f;--acc:#7fb0e0;--pos:#e08a73;--neg:#6fc19a;--warn:#e0b25a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:24px 18px 48px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 14px}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--warn);padding:10px 14px;border-radius:8px;font-size:13.5px;color:var(--mut);margin-bottom:16px}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.tabs button{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:6px 12px;border-radius:999px;cursor:pointer;font:inherit;font-size:13.5px}
.tabs button[aria-pressed=true]{background:var(--acc);border-color:var(--acc);color:#fff}
.grid{display:grid;grid-template-columns:260px 1fr;gap:16px;align-items:start}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.list{background:var(--card);border:1px solid var(--line);border-radius:10px;max-height:560px;overflow:auto}
.item{display:flex;justify-content:space-between;align-items:center;width:100%;border:0;border-bottom:1px solid var(--line);background:none;color:var(--fg);padding:9px 12px;cursor:pointer;font:inherit;text-align:left}
.item:hover{background:var(--bg)}.item[aria-current=true]{background:var(--bg);box-shadow:inset 3px 0 var(--acc)}
.chip{font-size:11.5px;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--mut)}
.chip.f{color:var(--pos);border-color:var(--pos)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px}
.panel h2{font-size:16px;margin:0 0 2px}.panel .s{color:var(--mut);font-size:13px;margin:0 0 14px}
.sc{margin:10px 0 16px}.sc .l{display:flex;justify-content:space-between;font-size:13.5px;margin-bottom:3px}
.bar{position:relative;height:14px;background:var(--bg);border:1px solid var(--line);border-radius:7px;overflow:hidden}
.bar i{position:absolute;left:0;top:0;bottom:0;background:var(--acc)}
.bar b{position:absolute;top:-2px;bottom:-2px;left:50%;width:2px;background:var(--fg);opacity:.55}
.kv{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}
.kv div{border:1px solid var(--line);border-radius:8px;padding:8px 10px}.kv span{display:block;color:var(--mut);font-size:12px}.kv strong{font-size:17px}
.dr{display:grid;grid-template-columns:150px 1fr 54px;gap:8px;align-items:center;font-size:13px;margin:5px 0}
.dr .t{position:relative;height:12px}.dr .t:before{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--line)}
.dr .t i{position:absolute;top:0;bottom:0;border-radius:3px}
.dr em{font-style:normal;color:var(--mut);text-align:right}
.lg{font-size:12.5px;color:var(--mut);margin-top:8px}
.out{font-size:14px;margin:6px 0 0;padding:10px 12px;border-radius:8px;background:var(--bg)}
footer{margin-top:20px;font-size:12.5px;color:var(--mut)}
</style></head><body><div class="wrap">
<h1>ReviewGuard: score one review</h1>
<p class="sub">Group 18 &middot; Big Data Systems and Algorithms &middot; Stony Brook University</p>
<div class="note"><b>What this is.</b> 400 reviews drawn from the held-out YelpChi test set, 100 from each outcome group (caught, false alarm, missed, correctly passed). Group sizes here are not the real proportions. Scores come from a scikit-learn copy of the logistic regression (test ROC-AUC __AUCF__ with network features, __AUCM__ without; the Spark model reports 0.8514 and 0.7689). "Fraud" means Yelp's filter flagged the review, not proven fraud. The 32 handcrafted features are anonymous in this dataset.</div>
<div class="tabs" id="tabs"></div>
<div class="grid"><div class="list" id="list"></div><div class="panel" id="panel"></div></div>
<footer>Contributions are exact for logistic regression: coefficient &times; standardized feature value, in log-odds. Network features: reviewer count (reviews linked through the same reviewer), same-product reviews, same-product-and-month reviews, and PageRank. In this data, reviewers with no other linked reviews score higher on average, so a count of 0 pushes the score up (the report's campaign analysis found multi-review clusters are less often filtered). Graph features are computed on the full graph including test reviews (transductive), as the report discloses. Source: github.com/siri780/reviewguard.</footer>
</div>
<script>
const D=__DATA__;
const NAMES={deg_rur:"Reviewer's linked reviews",deg_rsr:"Same product + rating",deg_rtr:"Same product + month",pagerank:"PageRank"};
const OUT={all:"All",TP:"Caught",FP:"False alarm",FN:"Missed",TN:"Correctly passed"};
const NET=new Set(Object.keys(NAMES));
let tab="all",cur=null;
const nm=k=>NAMES[k]||("Anonymous feature "+k);
function tabs(){const t=document.getElementById("tabs");t.innerHTML="";for(const k in OUT){const b=document.createElement("button");b.textContent=OUT[k];b.setAttribute("aria-pressed",k===tab);b.onclick=()=>{tab=k;cur=null;draw()};t.appendChild(b)}}
function rows(){return D.rows.filter(r=>tab==="all"||r.o===tab)}
function list(){const l=document.getElementById("list");l.innerHTML="";const rs=rows();if(!cur||!rs.includes(cur))cur=rs[0];
rs.forEach(r=>{const b=document.createElement("button");b.className="item";if(r===cur)b.setAttribute("aria-current","true");
b.innerHTML="<span>Review "+r.id+"</span><span class='chip "+(r.y?"f":"")+"'>"+(r.y?"filtered":"unfiltered")+" &middot; "+Math.round(r.pF*100)+"%</span>";
b.onclick=()=>{cur=r;list();panel()};l.appendChild(b)})}
function sc(label,p){return "<div class='sc'><div class='l'><span>"+label+"</span><strong>"+(p*100).toFixed(1)+"%</strong></div><div class='bar'><i style='width:"+(p*100)+"%'></i><b></b></div></div>"}
function panel(){const r=cur,p=document.getElementById("panel");if(!r){p.innerHTML="";return}
const flagged=r.pF>=.5;const mx=Math.max(...r.top.map(t=>Math.abs(t[1])),.5);
const verdict={TP:"Caught: flagged, and Yelp's filter agrees.",FP:"False alarm: flagged, but Yelp did not filter it.",FN:"Missed: not flagged, but Yelp's filter removed it.",TN:"Correctly passed: not flagged, and Yelp did not filter it."}[r.o];
p.innerHTML="<h2>Review "+r.id+"</h2><p class='s'>Black tick = 0.5 flag threshold</p>"+
sc("Score with network features",r.pF)+sc("Score without network features (32 handcrafted only)",r.pM)+
"<div class='kv'><div><span>Reviewer's other linked reviews</span><strong>"+r.dr+"</strong></div><div><span>Same product + month reviews</span><strong>"+r.dt+"</strong></div><div><span>Network shift (log-odds)</span><strong>"+(r.net>=0?"+":"")+r.net.toFixed(2)+"</strong></div></div>"+
"<div class='s' style='margin-bottom:6px'><b>Six largest contributions</b> (red pushes toward flagged, green toward unfiltered)</div>"+
r.top.map(t=>{const w=Math.abs(t[1])/mx*50,pos=t[1]>=0;return "<div class='dr'><span>"+nm(t[0])+(NET.has(t[0])?" *":"")+"</span><span class='t'><i style='"+(pos?"left:50%":"right:50%")+";width:"+w+"%;background:var("+(pos?"--pos":"--neg")+")'></i></span><em>"+(pos?"+":"")+t[1].toFixed(2)+"</em></div>"}).join("")+
"<div class='lg'>* network feature. Intercept "+r.b.toFixed(2)+"; contributions from all 36 features plus the intercept sum to the log-odds of the score.</div>"+
"<div class='out'><b>"+verdict+"</b> "+(flagged?"Flagged":"Not flagged")+" at 0.5.</div>"}
function draw(){tabs();list();panel()}
draw();
</script></body></html>'''
html = TEMPLATE.replace("__DATA__", data).replace("__AUCF__", f"{auc[0]:.4f}").replace("__AUCM__", f"{auc[1]:.4f}")
os.makedirs("demo", exist_ok=True)
open("demo/index.html", "w").write(html)
print(f"wrote demo/index.html ({len(html)} bytes); test AUC with network {auc[0]}, without {auc[1]}")
