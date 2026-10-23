from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd

THESIS = Path.home() / "OneDrive" / "Documents" / "Thesis Experiments"
ROOT = THESIS / "data" / "human_survey" / "human_resp" / "human_resp"
OUT = THESIS / "data" / "human_survey" / "derived"
WAVES = ["W26"]  # start with one wave; add more after checking the output

ORDINAL = (
    {"SAFECRIME_W26"}
    | {f"WORRY{c}_W26" for c in "ABCDEFG"}
    | {f"REASONGUN{c}_W26" for c in "ABCDE"}
    | {"GUNIDENTITY_W26", "GUNFRIEND_W26", "SHOOTFREQ_W26", "GUNACCESS_W26",
       "GUNLOCKED1_W26", "GUNLOADED1_W26", "GROWUPVIOL_W26", "GROWUPGUN4_W26",
       "GROWUPGUN6_W26", "GROWUPGUN7_W26", "CARRYGUN_W26"}
    | {f"GUNCONTRIB{c}_W26" for c in "ABCDEF"}
    | {f"GUNRESPNOKIDS{c}_W26" for c in "ABCDEFG"}
    | {f"GUNRESPKIDS{c}_W26" for c in "ABCDEFGH"}
    | {f"GUNACTIVITY{c}_W26" for c in "ABCDE"}
)

OUT.mkdir(parents=True, exist_ok=True)
questions_out = []
human_rows = []

for wave in WAVES:
    W = ROOT / f"American_Trends_Panel_{wave}"
    info = pd.read_csv(W / "info.csv")
    resp = pd.read_csv(W / "responses.csv", low_memory=False)
    wcol = f"WEIGHT_{wave}"
    weights = resp[wcol].astype(float).to_numpy()

    groups = {
        "men": (resp["SEX"] == "Male").to_numpy(),
        "women": (resp["SEX"] == "Female").to_numpy(),
        "overall": resp["SEX"].notna().to_numpy(),
    }

    for _, row in info.iterrows():
        key = row["key"]
        if key not in resp.columns:
            continue

        # option_mapping looks like {1.0: 'Very safe', ..., 99.0: 'Refused'}
        mapping = ast.literal_eval(row["option_mapping"])
        options = [str(v) for v in mapping.values() if str(v).strip().lower() != "refused"]
        label_to_idx = {o: i for i, o in enumerate(options)}
        n = len(options)
        ordinal = row["key"] in ORDINAL
        # ordinal = len(ast.literal_eval(row["option_ordinal"])) > 0

        labels = resp[key]
        idx = labels.map(label_to_idx)  # NaN for refused, skipped, or unmatched labels

        unmatched = labels.dropna()[~labels.dropna().isin(label_to_idx)].unique()
        if len(unmatched):
            print(f"[check] {key}: unmatched labels {list(unmatched)[:5]}")

        questions_out.append({
            "question_id": key,
            "topic": "",
            "question": str(row["question"]),
            "options": options,
            "ordinal": ordinal,
        })

        for group, mask in groups.items():
            valid = mask & idx.notna().to_numpy() & ~np.isnan(weights)
            if valid.sum() == 0:
                continue
            counts = np.bincount(idx[valid].astype(int), weights=weights[valid], minlength=n)
            probs = counts / counts.sum()
            for i, p in enumerate(probs):
                human_rows.append({
                    "question_id": key,
                    "group": group,
                    "option": i,
                    "prob": round(float(p), 6),
                })

with open(OUT / "questions.jsonl", "w", encoding="utf-8") as f:
    for q in questions_out:
        f.write(json.dumps(q) + "\n")

human = pd.DataFrame(human_rows)
human.to_csv(OUT / "human.csv", index=False)

print(f"Questions written: {len(questions_out)}")
print(f"Human rows written: {len(human)}")
print(f"Saved to: {OUT}")
