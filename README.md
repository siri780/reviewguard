# ReviewGuard

Group 18 · Big Data Systems and Algorithms · Stony Brook University

We measured what reviewer-network features add to suspicious-review detection on YelpChi. On the shared test set, logistic regression scored 0.7689 ROC-AUC with 32 handcrafted features, 0.8515 with the reviewer count added, and 0.8514 with all four graph features. Adding the graph features to gradient boosting moved ROC-AUC from 0.9445 to 0.9636.

## Repository and frozen version

[Public repository](https://github.com/siri780/reviewguard) · [Immutable source snapshot used for the report walkthrough](https://github.com/siri780/reviewguard/tree/ca04f3e1e6166572b6a79b6acb22305e96cf29f9) · [Final report](ReviewGuard_Final_Report.pdf)

The graded submission is the Git tag **`final-v1`**, which contains this README, the report PDF and the source. The immutable source reference is `ca04f3e1e6166572b6a79b6acb22305e96cf29f9`. The report PDF is distributed alongside this README. For the default Codespace path, open the repository on GitHub and create a Codespace. Run all commands from the checkout root. To inspect the frozen source separately, use a fresh clone:

```bash
git clone https://github.com/siri780/reviewguard.git
cd reviewguard
git switch --detach ca04f3e1e6166572b6a79b6acb22305e96cf29f9
```

## Architecture and execution path

1. `src/common.py` reads the MATLAB matrices, assigns stable row IDs, and implements the shared split, Spark logistic regression, and positive-class metrics. M1 writes or reuses `results/_shared_split_row_ids.csv`.
2. M3 converts sparse relations to edge lists and computes degrees and PageRank with GraphFrames. Its row-ID join writes `results/_m3_features.csv`. M3b reads the persisted features and compares five column subsets. M14 reads the same split for the three boosted models.
3. M9 holds out complete same-reviewer components. M10 compares reviewer means on both splits. M4 summarizes label rates by component; M4b evaluates a label-free ranking. M5 audits integrity; M6 fits a separate scikit-learn linear model for SHAP.
4. The scaling scripts time synthetic workloads. The Ott text scripts form a separate pipeline with their own folds and data. Neither uses the main test split for its primary analysis.

## Environment and installation

Original main experiments used two cores. The original scale-up used four cores and 16 GB on one machine. The November 2026 verification completed all ten YelpChi scripts with Python 3.12.3, Java 17.0.21.1 and PySpark 3.5.3. The successful environment included setuptools 84.2.1, graphframes 0.6 (Python), scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, scipy 1.18.1 and shap 0.54.1. Spark fetches the GraphFrames JVM artifact `io.graphframes:graphframes-spark3_2.12:0.12.3` at startup. Maven access is required on the first run.

In an Ubuntu-based Codespace terminal, from the checkout root:

```bash
sudo apt-get update -qq && sudo apt-get install -y -qq openjdk-17-jdk-headless
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
hash -r
java -version

python -m venv /tmp/reviewguard-tools
/tmp/reviewguard-tools/bin/python -m pip install uv
UV=/tmp/reviewguard-tools/bin/uv
"$UV" venv --python 3.12.3 /tmp/reviewguard-py312
source /tmp/reviewguard-py312/bin/activate
"$UV" pip install --python "$VIRTUAL_ENV/bin/python" -r requirements.txt \
  setuptools==84.2.1 graphframes==0.6 scikit-learn==1.9.1 pandas==3.1.1 \
  numpy==2.5.5 scipy==1.18.1 shap==0.54.1
export PYSPARK_PYTHON="$VIRTUAL_ENV/bin/python"
export PYSPARK_DRIVER_PYTHON="$VIRTUAL_ENV/bin/python"
python -c "import sys,pyspark; from distutils.version import LooseVersion; print(sys.version, pyspark.__version__)"
```

Use the same terminal for the remaining steps. If reopening a terminal, reactivate the environment and repeat the Java and PySpark exports. On the verification machine, the original Python 3.14.2 run failed during PySpark serialization. A clean Python 3.12 environment also needed setuptools for PySpark's distutils import.

### Dataset used for the verified run

The CARE-GNN archive worked after the direct DGL download and DGL import routes failed. Run from the ReviewGuard root:

```bash
mkdir -p data results
DATA_TMP=$(mktemp -d)
git clone --depth 1 https://github.com/YingtongDou/CARE-GNN.git "$DATA_TMP/CARE-GNN" &&
unzip -o "$DATA_TMP/CARE-GNN/data/YelpChi.zip" -d "$DATA_TMP" &&
cp "$DATA_TMP/YelpChi.mat" data/YelpChi.mat
python -c "import scipy.io as s; m=s.loadmat('data/YelpChi.mat'); print(sorted(k for k in m if not k.startswith('__'))); print(m['features'].shape,m['label'].flatten().mean())"
```

Expected keys: `features, homo, label, net_rsr, net_rtr, net_rur`. Expected shape and positive fraction: `(45954, 32) 0.14529747138442792`.

## Reproduction procedure

Execute each command from the repository root in order. M1 establishes the main split. M3 writes the feature table consumed by later scripts.

```bash
python src/m1_baseline.py
python src/m3_network_features.py
python src/m3b_ablation.py
python src/m14_gbt_check.py
python src/m9_reviewer_disjoint_split.py
python src/m10_reviewer_aggregates.py
python src/m4_campaign_detection.py
python src/m4b_label_free_campaigns.py
python src/m5_leakage_audit.py
python src/m6_shap_explainability.py
```

| Final-report evidence | Script and recorded output |
|---|---|
| Table 1, dimensions and prevalence | Dataset check above; M1; grouped text baseline for Ott |
| Table 2, architecture | Source walkthrough in common.py and the M1/M3 scripts |
| Tables 3 and 4, ablation and confusion counts | M1, M3, M3b; results/m3b_ablation_results.txt |
| Table 5, robustness | M9, M14; results/m9_reviewer_disjoint_results.txt and results/m14_gbt_results.txt |
| Reviewer means | M10; results/m10_reviewer_aggregates_results.txt |
| Campaign, audit and explanation | M4, M4b, M5, M6; results/m4*, results/m5*, results/m6_shap_results.txt |
| Table 6, timing | M11; results/m11_scale_cores4.csv; group trials by row count and average each stage before summing |
| Section 8, intervals and seed check | M15; results/m15_boosted_disjoint_ci_results.txt |
| Section 8, demo | M16; demo/index.html |
| Text and annotation | M12g; results/m12g_text_baseline_results.txt; historical annotation records in results/annotation/ |

```bash
grep -H 'ROC-AUC' results/m1_baseline_results.txt results/m3_network_results.txt results/m3b_ablation_results.txt
grep -A6 'Derived' results/m3b_ablation_results.txt
cat results/m14_gbt_results.txt
```

Expected ablation ROC-AUCs in order: 0.7689, 0.8515, 0.8514, 0.7844, 0.7218. Removing reviewer degree lowers AUC by 0.0670 and fraud F1 by 0.1297. Freshly generated output from the corrected script uses after minus before. Boosted AUCs: 0.9445, 0.9602, 0.9636.

### Follow-up checks and demo (M15, M16)

These two scripts need only `results/_m3_features.csv` (from M3), `results/_shared_split_row_ids.csv` (from M1) and `data/YelpChi.mat`; they do not start Spark. Run from the repository root:

```bash
python src/m15_boosted_disjoint_ci.py
python src/m16_build_demo.py
```

M15 repeats the feature comparison with boosting and logistic regression on both splits, with 1,000-replicate bootstrap intervals that resample whole reviewer clusters of the test set, plus a ten-seed check of boosting. It takes about two minutes. M16 writes `demo/index.html`, a self-contained page that lets a reader pick any of 400 sampled test reviews and see its score with and without network features and the largest feature contributions. Open the file in a browser. It is an explorer of saved model outputs, not a live scoring service, and the sample is not representative of the real class proportions.

The committed `results/m15_boosted_disjoint_ci_results.txt` was generated with graph features recomputed outside Spark (a scipy copy of M3 that reproduces the logistic-regression test AUCs to within 0.0002). Boosting results also vary slightly across environments because early stopping uses a random validation split. Rerunning M15 on the Spark-produced feature table overwrites the file; expect the same conclusions with differences of a few thousandths. The M15 and M16 scripts are the only ones not covered by the ten-script verification.

### Additional original experiments

For the four-core synthetic workload, provision a four-core machine and run:

```bash
CORES=4 KS=1,2,5,11,22 TRIALS=2 python src/m11_scale_up.py
```

For the separate text study, unpack the Ott corpus so the polarity directories are under `data/optspam/op_spam_v1.4/`. Then run:

```bash
python src/m12_text_baseline.py
python src/m12g_text_grouped.py
```

M12 creates out-of-fold predictions. M12g gives the grouped five-fold text metrics for the report. M13 uses M12 predictions to generate blank annotation sheets. To generate **new** sheets, run `python src/m13_make_annotation_sheets.py` in a separate checkout with the data and predictions present, preserving completed annotation files. The corrected generator keys text by polarity and filename. The historical human judgments apply to the original positive-only sheets.

## Verification and ownership

| Area | Owner | Modules |
|---|---|---|
| Data and baseline pipeline | Shirisha Gajendra | common, M1, M3 |
| Graph analysis and robustness | Sai Varshith Sabbella | M3b, M4, M4b, M9, M10 |
| Integrity checks and explanation | Dhanyashri Marianna | M5, M6, M14 |
| Systems and scaling | Gaurang Acharekar | M7, M11 |
| Text and human study | Vishnu Malladi | M12/M12g, M13 and annotation analysis |

Dhanyashri Marianna independently completed the ten-script YelpChi run in Codespaces and verified all eight requested ROC-AUCs. The M5 automated checks passed; the transductive design remains a limitation. SHAP ranked f13, f14 and reviewer degree first through third, with a mean top-five overlap of 4.33/5. The scale, text and human-study measurements come from the original experiments and were outside this rerun.

The negative campaign finding, single-machine synthetic benchmark, positive-only annotation sample, separate text corpus, and incomplete proposal items are described in the report. AI tools assisted with scaffolding, debugging, analysis review and report drafting.

