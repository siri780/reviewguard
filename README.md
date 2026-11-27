# ReviewGuard

Spark (MLlib + GraphFrames) study of whether reviewer-network features improve suspicious-review detection on YelpChi. Team 18, Big Data Systems and Algorithms, Stony Brook University. Final report: `ReviewGuard_Final_Report.pdf`.

## 1. Repository and frozen version

- Repo: https://github.com/siri780/reviewguard (public)
- Frozen version for grading: tag `final-v1`. Anything committed after this tag is not part of the submission.
- Layout: `src/` scripts, `results/` recorded outputs, `docs/annotation_guideline.md`, `requirements.txt`, `ReviewGuard_Final_Report.pdf`, `README.md`.
- All commands below are run **from the repository root** (the scripts read `data/...` and write `results/...` relative to the current directory).

## 2. Architecture and execution path

```
data/YelpChi.mat -> common.py (load, edges, persisted split) -> m3 (GraphFrames features)
   -> m1 / m3 / m3b (Spark LR) -> m9, m10, m14 (robustness)
   -> m4, m4b, m5, m6 (campaigns, leakage, SHAP) -> m7, m11 (scale-up)
Ott corpus (separate) -> m12, m12g (text baseline) -> m13 (annotation sheets)
```

There is one train/test split. `m1_baseline.py` creates it on first run and saves it to `results/_shared_split_row_ids.csv` (stratified 80/20, seed 42; 36,764 train / 9,190 test); every other script reloads that file through `common.SPLIT_PATH`. `m3_network_features.py` writes `results/_m3_features.csv`, which m3b, m4b, m5, m6, m10 and m14 reuse. So run m1 first, then m3. (`src/make_split.py` is an unused early script and is not part of the pipeline.)

## 3. Environment and dependencies

- Java 17 (required; Java 25 breaks Spark 3.5): `export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64`
- Python 3.12, PySpark 3.5.3, GraphFrames JAR `io.graphframes:graphframes-spark3_2.12:0.12.3` (pulled by Spark at start-up), scikit-learn, pandas, numpy, scipy, shap, openpyxl
- Hardware used: GitHub Codespaces, 2 cores (main runs), 4 cores / 16 GB (scale-up)

```bash
pip install -r requirements.txt
```

### Data preparation

The scripts do not download anything. They expect these files to exist before the first run:

1. **YelpChi:** place the YelpChi `.mat` file (45,954 reviews; keys `features`, `label`, `net_rur`, `net_rsr`, `net_rtr`, `homo`) at `data/YelpChi.mat`. `data/` is not tracked in git. One way to obtain the file is through DGL's `FraudYelpDataset`, which downloads it to a cache folder under `~/.dgl`; copy it from there:
   ```bash
   mkdir -p data && cp "$(find ~/.dgl -name YelpChi.mat | head -1)" data/YelpChi.mat
   ```
2. **Ott et al. corpus** (only for m12, m12g, m13): download `op_spam_v1.4` and unpack it so that `data/optspam/op_spam_v1.4/` exists.

## 4. Reproduction procedure

From the repository root, in this order:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
python src/m1_baseline.py                 # also creates results/_shared_split_row_ids.csv
python src/m3_network_features.py         # also creates results/_m3_features.csv
python src/m3b_ablation.py
python src/m9_reviewer_disjoint_split.py
python src/m10_reviewer_aggregates.py
python src/m14_gbt_check.py
python src/m4_campaign_detection.py
python src/m4b_label_free_campaigns.py
python src/m5_leakage_audit.py
python src/m6_shap_explainability.py
python src/m7_scalability.py
python src/m11_scale_up.py                # use a 4-core machine; writes m11_scale_cores*.csv
python src/m12_text_baseline.py           # needs data/optspam; writes results/m12_oof_predictions.csv
python src/m12g_text_grouped.py           # needs data/optspam
python src/m13_make_annotation_sheets.py  # needs m12 output; see known issues
```

| Report item | Script | Output file |
|---|---|---|
| Table 3, Figure 2 (ablation) | m1, m3, m3b | `results/m1_*`, `results/m3_*`, `results/m3b_ablation_results.txt` |
| Table 4 (robustness) | m9, m10, m14 | `results/m9_*`, `results/m10_*`, `results/m14_gbt_results.txt` |
| Section 8.2 (campaigns, SHAP, leakage) | m4, m4b, m5, m6 | `results/m4*`, `results/m5*`, `results/m6*` |
| Table 6, Figure 6 (scale-up) | m11 | `results/m11_scale_cores*.csv` |
| Table 5 (text, annotation) | m12g, m13 | `results/m12g_*`, `results/annotation/` |

Timings and AUC values can differ slightly across machines and Spark runs.

## 5. Verification and ownership

Status of each result: **Re-runnable** = script and output both in the repo. **Recorded only** = output in the repo, generated in a session we cannot fully re-run.

| Area | Contents | Owner | Status |
|---|---|---|---|
| A. Data and baseline | common.py, split, m1, m3 | Shirisha Gajendra | Re-runnable |
| B. Graph analysis | m3b, m4, m4b, m9, m10 | Sai Varshith Sabbella | Re-runnable |
| C. Checks and explanation | m5, m6, m14 | Dhanyashri Marianna | Re-runnable |
| D. Scale and systems | m7, m11, environment | Gaurang Acharekar | Re-runnable |
| E. Text and human study | m12g, m13, annotation analysis | Vishnu Malladi | Annotation results recorded (see known issue below); m13 fix not re-run |

Claude (Anthropic) assisted with scaffolding, debugging, analysis review and report drafting; the team made the design decisions and is responsible for the results.

## Known issues and unfinished work

- The annotation sheets used in the study were built by an earlier version of `m13_make_annotation_sheets.py` that looked up review text by bare file name. Names repeat across the corpus's polarity folders, so every sheet row showed the positive-sentiment text. True labels were unaffected. The answer key was corrected (`results/annotation/answer_key_corrected.csv`) and the study is reported as positive-sentiment only. The script now looks up text by (polarity, file name); this fix has not been re-run, and re-running it would produce sheets that differ from the ones the annotators saw.
- `m3b_ablation.py` and `m14_gbt_check.py` were edited after the recorded outputs in `results/` were produced (a sign-wording fix and a direct split load); the saved result files will be regenerated on the next run.
- Not built from the proposal: Trust Score, tri-class labels, DistilBERT/XGBoost ensemble, MinHash/LSH, LIME, counterfactuals, LLM hard negatives, interactive demo, 10M-row benchmark.
- Text, behavior and network signals were never combined on one dataset. The scale-up uses synthetic replicas of YelpChi on a single machine.
- The command order above has been checked against the scripts' file paths but has not been executed end to end from a clean checkout.
