from pathlib import Path
import pandas as pd

# Adjust if your folder is somewhere else
THESIS = Path.home() / "OneDrive" / "Documents" / "Thesis Experiments"
W = THESIS / "data" / "human_survey" / "human_resp" / "human_resp" / "American_Trends_Panel_W26"
OUT = THESIS / "data" / "human_survey" / "inspect_W26.txt"  # inside an ignored folder

lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


# 1. Question list
info = pd.read_csv(W / "info.csv")
log(f"info.csv: {len(info)} questions")
log(f"Columns: {list(info.columns)}")
log("")
for _, row in info.head(5).iterrows():
    log(f"[{row['key']}] {row['question']}")
    log(f"   options: {row['option_mapping']}")
    log("")

# 2. Demographic metadata
meta = pd.read_csv(W / "metadata.csv")
log("=" * 60)
log(f"metadata.csv: {len(meta)} variables")
for _, row in meta.iterrows():
    log(f"{row['key']}: {row['options']}")
log("")

# 3. Respondent-level file
resp = pd.read_csv(W / "responses.csv", low_memory=False)
log("=" * 60)
log(f"responses.csv: {resp.shape[0]} respondents, {resp.shape[1]} columns")
log("")
log("ALL COLUMNS:")
for i, c in enumerate(resp.columns):
    log(f"  {i:3d}  {c}")
log("")

# 4. Columns that look like gender or weights
keywords = ["GENDER", "SEX", "WEIGHT", "WT", "CREGION", "AGE", "PARTY", "PARTISAN", "EDUC", "RACE", "IDEO"]
matches = [c for c in resp.columns if any(k in str(c).upper() for k in keywords)]
log("=" * 60)
log(f"Columns matching demographic/weight keywords: {matches}")
log("")

# 5. Value counts for likely gender columns, plus weight summaries
for c in matches:
    if any(k in c.upper() for k in ["GENDER", "SEX"]):
        log(f"--- value counts: {c}")
        log(resp[c].value_counts(dropna=False).to_string())
        log("")
    if "WEIGHT" in c.upper() or c.upper().startswith("WT"):
        log(f"--- weight summary: {c}")
        log(resp[c].describe().to_string())
        log("")

OUT.write_text("\n".join(lines), encoding="utf-8")
log("=" * 60)
log(f"Saved full report to: {OUT}")
