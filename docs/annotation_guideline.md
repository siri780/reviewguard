# ReviewGuard — Human Annotation Guideline (D6)

## Purpose
An independent check on whether the signals ReviewGuard treats as "suspicious" correspond to what a human reading the review would also flag, and a measured inter-rater agreement number, not an assumed one.

## Labels
Each review gets exactly one of three labels. Use Insufficient Evidence rather than forcing a call.

- **Suspicious**: a concrete reason to doubt it is a genuine, independent customer experience (templated or generic language, implausible specificity, extreme sentiment with no concrete detail).
- **Not Suspicious**: specific, plausible, internally consistent detail about an actual experience.
- **Insufficient Evidence**: too short or too generic to support either label.

## Protocol
1. Two annotators label all pilot reviews independently, without seeing each other's labels, the model's prediction, or the ground-truth label.
2. Each records a label plus a one-line reason.
3. Compute Cohen's kappa on the two label columns (sklearn.metrics.cohen_kappa_score). Report separately: human-human agreement, human-truth agreement, and human-model agreement.
4. Disagreements go to a short team discussion; the resolution is logged, not silently overwritten.

## Kappa interpretation (Landis & Koch, for reporting only)
- below 0.00 poor; 0.00-0.20 slight; 0.21-0.40 fair; 0.41-0.60 moderate; 0.61-0.80 substantial; 0.81-1.00 almost perfect.
- With n=20 the confidence interval is wide; report the point estimate with that caveat.

## Known limitation
The YelpChi copy used in this project has no review text, so this study cannot be run until raw text is available. Structural features (degrees, cluster size) are not a substitute for reading a review and should not be presented as human annotation.
