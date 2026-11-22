"""Phase 5：错在哪？—— 把模型错误和「人类标注者的分歧程度」对齐来看。

PhraseBank 的四个子集是嵌套的（AllAgree ⊂ 75 ⊂ 66 ⊂ 50），一句话能进入的最高子集
= 标注者一致度的下界：100% / ≥75% / ≥66% / ≥50%。

用法：
  python -m finsent.error_analysis        # 打印分析，写 outputs/error_hard_core.csv
"""
import numpy as np
import pandas as pd

from finsent.data import ID2LABEL, OUTPUTS_DIR, PROCESSED_DIR, load_subset

PREDS_DIR = OUTPUTS_DIR / "preds"
LEVELS = ["100", "75", "66", "50"]          # 一致度从高到低
# 「干净」模型：没见过 PhraseBank 测试句的标注（FinBERT 单独作为 contamination 对照）
CLEAN = {
    "TF-IDF": ["tfidf-lr_50agree_s0"],
    "ModernBERT": [f"ModernBERT-base_50agree_s{s}" for s in (1, 2, 3)],
    "DeBERTa": [f"deberta-v3-base_50agree_s{s}" for s in (1, 2, 3)],
    "Haiku": ["haiku-5-5_zeroshot-low_s0"],
    "Opus": ["opus-5-5_zeroshot-low_s0"],
}
CONTAMINATED = {
    "FinBERT-ft": [f"finbert_50agree_s{s}" for s in (1, 2, 3)],
    "FinBERT-zs": ["finbert_zeroshot_s0"],
}


def agreement_level(test: pd.DataFrame) -> pd.Series:
    level = pd.Series("50", index=test.index)
    for name, tag in [("66agree", "66"), ("75agree", "75"), ("allagree", "100")]:
        level[test["sentence"].isin(set(load_subset(name)["sentence"]))] = tag
    return pd.Categorical(level, LEVELS, ordered=True)


def load_model_preds(runs: list[str], sentences) -> tuple[np.ndarray, np.ndarray]:
    """返回 (多数票预测, 每个 seed 的预测矩阵)。3 个 seed 各不相同时取 seed 1 的预测。"""
    mats = []
    for r in runs:
        d = pd.read_parquet(PREDS_DIR / f"{r}.parquet")
        assert (d["sentence"].values == sentences).all(), r
        mats.append(d["pred"].values)
    mats = np.stack(mats)
    vote = np.array([np.bincount(col[col >= 0], minlength=3).argmax() if (col >= 0).any() else -1
                     for col in mats.T])
    if len(runs) == 3:  # bincount 平票时 argmax 取最小 label；三票各不同时改用 seed 1
        all_diff = (mats[0] != mats[1]) & (mats[1] != mats[2]) & (mats[0] != mats[2])
        vote[all_diff] = mats[0][all_diff]
    return vote, mats


def main():
    pd.set_option("display.width", 200)
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    test["level"] = agreement_level(test)
    y, sents = test["label"].values, test["sentence"].values

    votes, per_seed = {}, {}
    for name, runs in {**CLEAN, **CONTAMINATED}.items():
        votes[name], per_seed[name] = load_model_preds(runs, sents)

    # ---- A. 错误率 vs 标注者一致度 ------------------------------------------------
    print("A. error rate by annotator agreement level (mean over seeds)")
    rows = {}
    for name, mats in per_seed.items():
        err = (mats != y).mean(axis=0)               # 每句的错误率（多 seed 取平均）
        rows[name] = test.assign(err=err).groupby("level", observed=True)["err"].mean()
    tab = pd.DataFrame(rows).T[LEVELS]
    tab.columns = [f"{c}% agree (n={(test.level == c).sum()})" for c in LEVELS]
    print(tab.round(3).to_string())

    # ---- B. 每句被几个干净模型判错 ---------------------------------------------------
    wrong = np.stack([votes[m] != y for m in CLEAN])          # (5, n)
    test["n_wrong"] = wrong.sum(axis=0)
    print(f"\nB. how many of the {len(CLEAN)} clean models get each sentence wrong")
    print(pd.crosstab(test["level"], test["n_wrong"], margins=True).to_string())

    # ---- C. hard core：所有干净模型都错 ---------------------------------------------
    hard = test[test["n_wrong"] == len(CLEAN)].copy()
    clean_votes = np.stack([votes[m] for m in CLEAN])
    hard["consensus"] = [np.bincount(clean_votes[:, i], minlength=3).argmax() for i in hard.index]
    hard["unanimous"] = [(clean_votes[:, i] == clean_votes[0, i]).all() for i in hard.index]
    for name in CONTAMINATED:
        hard[name] = votes[name][hard.index]
    opus = pd.read_parquet(PREDS_DIR / "opus-5-5_zeroshot-low_s0.parquet")
    hard["opus_conf"] = opus["confidence"].values[hard.index]

    print(f"\nC. hard core: {len(hard)} sentences ({len(hard) / len(test):.1%}) wrong for all clean models")
    print("   by agreement level:", hard["level"].value_counts().reindex(LEVELS).to_dict())
    pair = hard.apply(lambda r: f"{ID2LABEL[r.label]}→{ID2LABEL[r.consensus]}", axis=1)
    print("   gold→consensus:", pair.value_counts().to_dict())
    print(f"   all 5 clean models give the SAME wrong label: {hard['unanimous'].mean():.0%}")
    print(f"   Opus confidence on hard core: mean {hard['opus_conf'].mean():.2f} "
          f"(all test: {opus['confidence'].mean():.2f})")
    for name in CONTAMINATED:
        acc_hard = (hard[name] == hard["label"]).mean()
        acc_rest = (votes[name][test["n_wrong"] < len(CLEAN)] == y[test["n_wrong"] < len(CLEAN)]).mean()
        print(f"   {name} accuracy: hard core {acc_hard:.0%} vs rest {acc_rest:.0%}")

    out = hard.assign(gold=hard["label"].map(ID2LABEL), consensus_label=hard["consensus"].map(ID2LABEL),
                      finbert_zs=hard["FinBERT-zs"].map(ID2LABEL))
    out = out[["level", "gold", "consensus_label", "unanimous", "opus_conf", "finbert_zs", "sentence"]]
    out = out.sort_values(["level", "gold"]).reset_index(drop=True)
    path = OUTPUTS_DIR / "error_hard_core.csv"
    out.to_csv(path, index_label="id")
    print(f"\nSaved {path}")
    return test, hard


if __name__ == "__main__":
    main()
