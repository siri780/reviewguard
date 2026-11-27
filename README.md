# ReviewGuard

Measuring how much reviewer-network features add to a baseline for suspicious-review detection (Spark MLlib / GraphFrames). Big Data Systems and Algorithms (MS Data Science, Stony Brook), Team 18: Shirisha Gajendra, Vishnu Malladi, Dhanyashri Marianna, Gaurang Acharekar, Sai Varshith Sabbella.

## Data
YelpChi (Rayana and Akoglu, KDD 2015): 45,954 reviews, 14.5% filtered as spam. We use the pre-extracted 32 handcrafted features and the three relation graphs (R-U-R, R-S-R, R-T-R) from the DGL FraudYelpDataset (`YelpChi.mat`, not committed; download into `data/`). The raw text version needs a request to the authors (pending), so text work uses the separate Ott et al. deceptive-opinion corpus.

## Scripts (src/) and key results
| Script | Purpose | Result |
|---|---|---|
| m1_baseline.py | Logistic regression on 32 features, shared split | AUC 0.7689, AUPRC 0.4072, fraud F1 0.2322 |
| m3_network_features.py | + degree (3 relations) + PageRank | AUC 0.8514, AUPRC 0.5312, fraud F1 0.3954 |
| m3b_ablation.py | Five-way ablation | M1 + deg_rur (0.8515) matches full M3 |
| m4_campaign_detection.py | Same-reviewer clusters (descriptive) | Largest 15 clusters contain no fraud |
| m4b_label_free_campaigns.py | Label-free cluster/burst rule | AUC 0.415 (fails) |
| m5_leakage_audit.py | Scoped integrity checks | Row-id separation, coverage, correlation checks pass |
| m6_shap_explainability.py | SHAP on a separately fitted scikit-learn model | deg_rur is the only graph feature in the top 10 |
| m7_scalability.py, m11_scale_up.py | Runtime; synthetic scale-up to 1.01M rows | About 11 s on 4 cores; connected components scale poorly with cores |
| m9_reviewer_disjoint_split.py | Whole reviewer clusters held out | M1 0.7719; M1 + deg_rur 0.8519; full M3 0.8520 |
| m10_reviewer_aggregates.py | Reviewer-mean features (behavior proxy) | Small further gain (AUC 0.8577 random, 0.8628 disjoint) |
| m12_text_baseline.py, m12g_text_grouped.py | TF-IDF text baseline on Ott corpus; grouped-fold rerun | AUC 0.937; 4 duplicate groups, none crossing folds |
| m13_make_annotation_sheets.py | Blind annotation sheets (has a known file-name lookup bug, see below) | |
| m14_gbt_check.py | Gradient-boosted check of the network gain | M1 0.9445; + deg_rur 0.9602; full M3 0.9636 |

## Findings
1. With logistic regression, graph features raise AUC 0.769 to 0.851; most of the gain is one feature, deg_rur (reviewer activity). With a stronger gradient-boosted baseline the gain shrinks to about +0.019 AUC, so the size of the gain depends on the baseline.
2. The gain holds under a reviewer-disjoint split.
3. Fraud-labeled reviews concentrate in small, low-activity accounts. The same-reviewer cluster rule fails, which speaks to single-account behavior only, not coordination across accounts.
4. An early AUC of 0.97 was too high; two implementation problems (different splits, PageRank on the wrong graph) were fixed, and the earlier number should not be cited.

## Human annotation (D6)
Completed on 95 unique positive-sentiment reviews from the Ott corpus (YelpChi has no text). Protocol: `docs/annotation_guideline.md`. Results: `results/annotation/`. Three-label kappa 0.586; annotators about 77-78% accurate on the reviews they judged, text model 87.4% on the same reviews. Known issue: `m13_make_annotation_sheets.py` looked up review text by bare file name, so every row showed the positive-sentiment version; the corrected key (`results/annotation/answer_key_corrected.csv`) rescored the model on the text actually shown. The answer key was added after annotation finished.

## Status and open items
- Raw YelpChi review text (requested from the authors) would let text, behavior and network signals be combined on one dataset; today they are on different datasets.
- The scale-up is synthetic and single-machine; the gradient-boosted check is a single untuned run.
- Rewritten core scripts from a teammate review are untested and are kept off `main`.

## Setup
Java 17, Python 3.12, PySpark 3.5.3, GraphFrames 0.12.3 (pulled by Spark at runtime). Put `YelpChi.mat` in `data/` and run the scripts in src/ in the order listed above.
