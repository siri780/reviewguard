# ReviewGuard: Human Annotation Guideline (D6)

> **Final study note:** the study was run on 95 unique positive-sentiment reviews from the Ott et al. deceptive-opinion corpus (not YelpChi, which has no review text), by two annotators who saw review text only. The original plan was a 20-review pilot on YelpChi with structural context; this file records the labels and protocol that were used.

## Purpose
An independent check on how well human readers can tell deceptive from truthful reviews from text alone, and a measured inter-rater agreement number (Cohen's kappa), compared with a TF-IDF text model on the same reviews.

## Labels
Each review gets exactly one of three labels. Use Insufficient Evidence rather than forcing a guess: a noisy guess is worse for computing agreement than an honest "can't tell."

- **Suspicious**: a concrete reason to doubt it is a genuine, independent customer experience, such as templated language that could apply to any hotel, implausible specificity, or extreme sentiment with no concrete detail.
- **Not Suspicious**: specific, plausible, internally consistent detail about an actual stay; nothing above applies.
- **Insufficient Evidence**: too short or too generic to support either label with confidence.

## Protocol
1. Two annotators label every review independently, with no access to each other's labels, the model's prediction or the true label until both finish.
2. Each annotator records a label and a one-line reason.
3. After both finish, compute Cohen's kappa between the two label columns, then separately join the answer key to report three different numbers: human-human agreement, human-truth agreement, and human-model agreement. These are not collapsed into one accuracy.
4. Disagreements and Insufficient Evidence calls are reported, not silently overwritten. For accuracy against the true label, only reviews where the annotator made a call are scored, and the abstention rate is reported alongside.

## Kappa interpretation (Landis and Koch, for reporting only)
| kappa | Interpretation |
|---|---|
| < 0.00 | Poor |
| 0.00-0.20 | Slight |
| 0.21-0.40 | Fair |
| 0.41-0.60 | Moderate |
| 0.61-0.80 | Substantial |
| 0.81-1.00 | Almost perfect |

With 95 reviews the confidence interval is wide, so the point estimate is reported with a bootstrap interval.

## What was run
- Sample: 100 reviews drawn from the Ott corpus, shuffled; five turned out to be repeated copies of the same review, leaving 95 unique reviews (49.5% deceptive).
- Known issue: the sheet builder looked up review text by bare file name, so every row displayed the positive-sentiment version of that review. The true labels were unaffected; the model scores were recomputed on the text actually shown. Because of this the study covers positive-sentiment reviews only.
- Results: `results/annotation/` (annotator labels and reasons, true label, corrected model scores). Three-label kappa 0.586; annotators about 77-78% accurate on the reviews they judged; text model 87.4% on the same reviews.

## Limitations
Small sample, wide interval, both annotators are teammates rather than independent crowd workers, positive-sentiment text only, and no structural signals were shown. The study does not test whether humans agree with the network signal.
