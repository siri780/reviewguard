# reviewguard
Scalable &amp; explainable suspicious e-commerce review detection (Spark/MLlib/GraphFrames) — Big Data Systems course project

## Additional experiments (added for the final report)

| Script | What it does | Result |
|---|---|---|
| `m4b_label_free_campaigns.py` | Label-free score = z(R-U-R cluster size) + z(deg_rtr) + z(deg_rsr), scored against the label only afterward | ROC-AUC 0.4150 (below chance); precision@548 = 2.6%. Per signal: comp_size 0.279 (inverse), deg_rtr 0.510, deg_rsr 0.474 |
| `m9_reviewer_disjoint_split.py` | Holds out whole R-U-R clusters (no reviewer in both train and test) | M1 AUC 0.7719; M1+deg_rur 0.8519; full M3 0.8520. The network gain holds |

Findings: the gain from network features is carried by deg_rur (reviewer activity) and survives a reviewer-disjoint split. Fraud concentrates in small, low-activity accounts, not large clusters, so the campaign hypothesis is not supported on this benchmark.

Human annotation (D6): the protocol is in `docs/annotation_guideline.md`, but the dataset has no review text, so the study is blocked until raw text is available.

## Status and open items

- Core results: M1/M3, ablation, campaign analysis, leakage audit, SHAP, reviewer-disjoint split (m9), M2 proxy (m10), label-free rule (m4b).
- m14: gradient-boosted check on the shared split. The network gain shrinks from +0.08 AUC (logistic regression) to +0.019 AUC (boosting). It is a single untuned run.
- m11: scale-up is synthetic (replicated graph, one machine, about 1M rows); there is no multi-node run.
- m12 (text baseline) and m13 (annotation) use the Ott et al. deceptive-opinion corpus, a separate benchmark from YelpChi. The annotation answer key is not in this repo.
- Open: raw YelpChi review text (requested from the authors) would let text, behavior and network signals be combined on one dataset.
