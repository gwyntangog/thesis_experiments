import pandas as pd

splits = {"validation": "opinionqa.dev.jsonl", "test": "opinionqa.test.jsonl"}
df = pd.read_json(
    "hf://datasets/timchen0618/OpinionQA/" + splits["validation"],
    lines=True,
)

print(df.shape)
print(df.columns.tolist())
print(df.head(3).T)  # transposed so long fields are readable

row = df.iloc[0]
print("QUESTION:", row["question"])
print("PERSPECTIVES TYPE:", type(row["perspectives"]))
print("PERSPECTIVES:", row["perspectives"])
print("N PERSPECTIVES PER ROW:", df["perspectives"].apply(len).describe())
