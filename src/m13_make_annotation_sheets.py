"""M13 - Build blind annotation sheets from the Ott et al. corpus: 100 reviews (25 per polarity x
truth cell), shuffled; first 20 = pilot. Annotators see only an id and the text."""
import glob, os
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

oof = pd.read_csv("results/m12_oof_predictions.csv")
paths = {os.path.basename(p): p for p in glob.glob("data/optspam/op_spam_v1.4/*/*/*/*.txt")}
rng = np.random.RandomState(42)
parts = [g.sample(25, random_state=rng) for _, g in oof.groupby(["polarity", "label"])]
s = pd.concat(parts).sample(frac=1, random_state=rng).reset_index(drop=True)
s["id"] = [f"R{i+1:03d}" for i in range(len(s))]
s["text"] = [open(paths[f], encoding="utf-8", errors="ignore").read().strip() for f in s.file]
os.makedirs("results/annotation", exist_ok=True)
s[["id", "file", "polarity", "hotel", "label", "prob", "prediction"]].rename(
    columns={"label": "truth_deceptive", "prob": "model_prob", "prediction": "model_pred"}
).to_csv("results/annotation/answer_key.csv", index=False)

for who in ["annotator_1", "annotator_2"]:
    wb = Workbook(); ws = wb.active; ws.title = "annotations"
    ws.append(["id", "review_text", "label", "one_line_reason"])
    for c in ws[1]:
        c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="DDE6F3")
    for _, r in s.iterrows():
        ws.append([r.id, r.text, None, None])
    ws.column_dimensions["A"].width = 8; ws.column_dimensions["B"].width = 95
    ws.column_dimensions["C"].width = 22; ws.column_dimensions["D"].width = 45
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    dv = DataValidation(type="list", formula1='"Suspicious,Not Suspicious,Insufficient Evidence"', allow_blank=True)
    ws.add_data_validation(dv); dv.add(f"C2:C{len(s)+1}")
    ws.freeze_panes = "A2"
    wb.save(f"results/annotation/{who}.xlsx")
print("made", len(s), "reviews; pilot = R001-R020")
print(s.groupby(["polarity", "label"]).size())
