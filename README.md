# reviewguard
Scalable &amp; explainable suspicious e-commerce review detection (Spark/MLlib/GraphFrames) — Big Data Systems course project

## Additional experiments (added for the final report)

| Script | What it does | Result |
|---|---|---|
| `m4b_label_free_campaigns.py` | Label-free score = z(R-U-R cluster size) + z(deg_rtr) + z(deg_rsr), scored against the label only afterward | ROC-AUC 0.4150 (below chance); precision@548 = 2.6%. Per signal: comp_size 0.279 (inverse), deg_rtr 0.510, deg_rsr 0.474 |
| `m9_reviewer_disjoint_split.py` | Holds out whole R-U-R clusters (no reviewer in both train and test) | M1 AUC 0.7719; M1+deg_rur 0.8519; full M3 0.8520. The network gain holds |

Findings: the gain from network features is carried by deg_rur (reviewer activity) and survives a reviewer-disjoint split. Fraud concentrates in small, low-activity accounts, not large clusters, so the campaign hypothesis is not supported on this benchmark.

Human annotation (D6): the protocol is in `docs/annotation_guideline.md`, but the dataset has no review text, so the study is blocked until raw text is available.
