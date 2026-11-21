import glob
import pandas as pd
from finsent.evaluate import macro_f1

rows = []
for f in glob.glob("outputs/preds/*.parquet"):
    df = pd.read_parquet(f)
    for test_subset, d in [("allagree", df[df.all_agree]), ("50agree", df)]:
        rows.append({"model": df["model"].iloc[0], "seed": df["seed"].iloc[0],
                     "test": test_subset, "macro_f1": macro_f1(d["label"], d["pred"])})
res = pd.DataFrame(rows)

table = (res.groupby(["model", "test"])["macro_f1"]
            .agg(["mean", "std", "count"])
            .unstack("test")
            .round(3))
print(table)