"""Phase 5：把 outputs/preds/ 里所有模型的预测汇总成一张 benchmark 表。

用法：
  python -m finsent.collection                         # 打印表格 + 写 outputs/results_benchmark.csv
  python -m finsent.collection --include_degenerate     # 把 degenerate run 也算进均值（默认排除）
  python -m finsent.collection --reference opus-5-5_zeroshot-low

约定（CLAUDE.md §5–§7）：
- 文件名 = <model>_<train>_s<seed>[_r<replicate>].parquet
- 不进均值的文件：s42（本地参考 run）、_r<k>（翻转率测试的重复采样）
- degenerate run（预测类别 < 3，或 best val macro-F1 < 0.7 = majority-class plateau）默认排除，并在表里注明
- 不确定性用 bootstrap（测试集抽样），不是 seed 之间的 std —— 后者小一个数量级
- 和参考模型的比较一律用 paired_bootstrap
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from finsent.data import OUTPUTS_DIR
from finsent.evaluate import bootstrap_ci, macro_f1, paired_bootstrap

PREDS_DIR = OUTPUTS_DIR / "preds"
NAME_RE = re.compile(r"^(?P<model>[^_]+)_(?P<train>.+)_s(?P<seed>\d+)(?:_r(?P<rep>\d+))?$")
EXCLUDED_SEEDS = {42}                  # 本地 MPS 参考 run，不进 3-seed 均值（§6）
DEGENERATE_VAL_F1 = 0.7                # best val macro-F1 低于它 = 卡在 majority-class plateau（§7）

FAMILY = {"majority": "baseline", "tfidf-lr": "baseline",
          "ModernBERT-base": "encoder", "deberta-v3-base": "encoder", "finbert": "encoder",
          "haiku-5-5": "llm", "opus-5-5": "llm", "sonnet-5-5": "llm"}
NOTES = {"finbert": "contaminated (seen PhraseBank)"}
FAMILY_ORDER = ["baseline", "encoder", "llm", "other"]


def best_val_f1(run: str):
    for path in [OUTPUTS_DIR / "logs" / f"{run}.trainer_state.json",
                 OUTPUTS_DIR / "runs" / run / "trainer_state.json"]:
        if path.exists():
            vals = [h["eval_macro_f1"] for h in json.loads(path.read_text())["log_history"]
                    if "eval_macro_f1" in h]
            return max(vals) if vals else None
    return None


def load_preds() -> pd.DataFrame:
    """每个 preds 文件一行：解析出的元数据 + 预测数组 + 是否进均值。"""
    rows, ref_sentences = [], None
    for f in sorted(PREDS_DIR.glob("*.parquet")):
        m = NAME_RE.match(f.stem)
        if not m:
            print(f"[skip] unrecognized file name: {f.name}")
            continue
        d = pd.read_parquet(f)
        if ref_sentences is None:
            ref_sentences = d["sentence"].values
        # paired bootstrap 的前提：所有文件是同一批句子、同一顺序
        assert (d["sentence"].values == ref_sentences).all(), f"row order differs: {f.name}"
        model, seed = m["model"], int(m["seed"])
        family = FAMILY.get(model, "other")
        val = best_val_f1(f.stem)
        n_classes = d.loc[d["pred"] >= 0, "pred"].nunique()
        degenerate = family != "baseline" and (n_classes < 3 or (val is not None and val < DEGENERATE_VAL_F1))
        excluded = ("reference run (s42)" if seed in EXCLUDED_SEEDS
                    else "flip-test replicate" if m["rep"] else None)
        rows.append({"run": f.stem, "model": model, "train": m["train"], "seed": seed,
                     "family": family, "best_val_f1": val, "n_pred_classes": n_classes,
                     "degenerate": degenerate, "excluded": excluded,
                     "label": d["label"].values, "pred": d["pred"].values,
                     "all_agree": d["all_agree"].values})
    return pd.DataFrame(rows)


def recall(y, p, cls):
    return recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)[cls]


def config_table(runs: pd.DataFrame, reference: str, include_degenerate: bool, n_boot: int) -> pd.DataFrame:
    """每个 (model, train) 配置一行。"""
    y = runs["label"].iloc[0]
    clean = runs["all_agree"].iloc[0]
    subsets = {"50agree": np.ones(len(y), bool), "allagree": clean, "clean": clean, "ambiguous": ~clean}
    # test=allagree 和 clean 是同一批 347 句；两列都保留是为了和之前的表格对得上

    use = runs[runs["excluded"].isna()]
    if not include_degenerate:
        use = use[~use["degenerate"]]
    ref = use[(use["model"] + "_" + use["train"]) == reference]
    if ref.empty:
        raise SystemExit(f"reference config {reference!r} not found among {sorted(set(use.model + '_' + use.train))}")
    ref_preds = list(ref["pred"])

    out = []
    for (model, train), g in use.groupby(["model", "train"], sort=False):
        preds = list(g["pred"])
        row = {"family": g["family"].iloc[0], "model": model, "train": train, "n_seeds": len(g),
               "note": NOTES.get(model, "")}
        dropped = runs[(runs.model == model) & (runs.train == train) & runs.degenerate & runs.excluded.isna()]
        if len(dropped) and not include_degenerate:
            row["note"] = "; ".join(filter(None, [row["note"], f"excluded degenerate: s{','.join(map(str, dropped.seed))}"]))
        for name, m in subsets.items():
            scores = [macro_f1(y[m], p[m]) for p in preds]
            row[f"{name}"] = np.mean(scores)
            row[f"{name}_std"] = np.std(scores, ddof=1) if len(scores) > 1 else np.nan
        row["50agree_lo"], row["50agree_hi"] = bootstrap_ci(y, preds, n_boot=n_boot)
        row["recall_neg"] = np.mean([recall(y, p, 0) for p in preds])
        row["recall_pos"] = np.mean([recall(y, p, 2) for p in preds])
        if f"{model}_{train}" != reference:
            for name in ["50agree", "ambiguous"]:
                m = subsets[name]
                r = paired_bootstrap(y[m], [p[m] for p in preds], [p[m] for p in ref_preds], n_boot=n_boot)
                row[f"d_{name}"], row[f"d_{name}_lo"], row[f"d_{name}_hi"] = r["delta"], r["lo"], r["hi"]
        out.append(row)
    table = pd.DataFrame(out)
    table["family"] = pd.Categorical(table["family"], FAMILY_ORDER, ordered=True)
    return table.sort_values(["family", "50agree"], ascending=[True, False]).reset_index(drop=True)


def fmt_delta(row, name):
    if pd.isna(row.get(f"d_{name}")):
        return "ref"
    sig = "*" if row[f"d_{name}_lo"] > 0 or row[f"d_{name}_hi"] < 0 else " "
    return f"{row[f'd_{name}']:+.3f}{sig}"


def print_table(table: pd.DataFrame, reference: str):
    show = pd.DataFrame({
        "model": table["model"], "train": table["train"], "n": table["n_seeds"],
        "50agree": [f"{r['50agree']:.3f} [{r['50agree_lo']:.3f},{r['50agree_hi']:.3f}]" for _, r in table.iterrows()],
        "seed sd": table["50agree_std"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}"),
        "allagree": table["allagree"].map("{:.3f}".format),
        "ambig": table["ambiguous"].map("{:.3f}".format),
        "rec neg/pos": [f"{a:.2f}/{b:.2f}" for a, b in zip(table["recall_neg"], table["recall_pos"])],
        "Δ50agree": [fmt_delta(r, "50agree") for _, r in table.iterrows()],
        "Δambig": [fmt_delta(r, "ambiguous") for _, r in table.iterrows()],
        "note": table["note"],
    })
    for fam, g in show.groupby(table["family"], observed=True):
        print(f"\n── {fam} " + "─" * 60)
        print(g.to_string(index=False))
    print(f"\nΔ = paired bootstrap vs {reference} (* = 95% CI excludes 0). "
          f"50agree [CI] = bootstrap CI of the seed-averaged score; 'seed sd' is the much smaller seed spread.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="deberta-v3-base_50agree", help="<model>_<train> 作为 Δ 的参照")
    p.add_argument("--include_degenerate", action="store_true")
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()

    runs = load_preds()
    skipped = runs[runs["excluded"].notna()]
    if len(skipped):
        print("not in means:", ", ".join(f"{r.run} ({r.excluded})" for r in skipped.itertuples()))
    degenerate = runs[runs["degenerate"] & runs["excluded"].isna()]
    if len(degenerate):
        verb = "included" if args.include_degenerate else "excluded"
        print(f"degenerate runs ({verb}):", ", ".join(
            f"{r.run} (val {'n/a' if pd.isna(r.best_val_f1) else f'{r.best_val_f1:.3f}'}, "
            f"{r.n_pred_classes} classes)" for r in degenerate.itertuples()))

    table = config_table(runs, args.reference, args.include_degenerate, args.n_boot)
    print_table(table, args.reference)
    out = OUTPUTS_DIR / "results_benchmark.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
