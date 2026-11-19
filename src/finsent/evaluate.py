"""所有模型共用的评估函数。"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from finsent.data import ID2LABEL

LABELS = [0, 1, 2]
NAMES = [ID2LABEL[i] for i in LABELS]


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=LABELS)


def bootstrap_ci(y_true, y_pred, metric=macro_f1, n_boot=1000, seed=0):
    """有放回地重采样测试集，返回指标的 95% 置信区间 (low, high)。"""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)          # 抽 n 个下标，可重复
        scores.append(metric(y_true[idx], y_pred[idx]))
    return tuple(np.percentile(scores, [2.5, 97.5]))


def paired_bootstrap(y_true, preds_a, preds_b, metric=macro_f1, n_boot=1000, seed=0) -> dict:
    """比较 A 和 B 两个模型：metric(A) − metric(B) 的点估计和 95% CI。

    「paired」= 每次重采样，A 和 B 用**同一组**测试句子。两个模型在同一批句子上的分数高度相关
    （难句子对谁都难），配对后这部分共同波动被抵消，比「看两个独立 CI 是否重叠」灵敏得多。

    preds_a / preds_b 可以是一个预测数组，也可以是多个 seed 的预测数组列表：
    每次重采样时对各 seed 的 metric 取平均，再相减 → 同时考虑测试集抽样和 seed 平均。

    返回 delta、95% CI (lo, hi)，以及 p_le_0 = 重采样中 delta ≤ 0 的比例（A 并不比 B 好的近似概率）。
    """
    y_true = np.asarray(y_true)
    as_list = lambda p: [np.asarray(x) for x in p] if isinstance(p, (list, tuple)) else [np.asarray(p)]
    A, B = as_list(preds_a), as_list(preds_b)
    score = lambda P, idx: np.mean([metric(y_true[idx], p[idx]) for p in P])
    all_idx = np.arange(len(y_true))
    delta = score(A, all_idx) - score(B, all_idx)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        boots.append(score(A, idx) - score(B, idx))
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": delta, "lo": lo, "hi": hi, "p_le_0": float((boots <= 0).mean())}


def summarize(y_true, y_pred, name: str = "model") -> dict:
    """打印完整报告，并返回一行可写入 results.csv 的结果。"""
    lo, hi = bootstrap_ci(y_true, y_pred)
    print(f"\n===== {name} =====")
    print(classification_report(y_true, y_pred, labels=LABELS, target_names=NAMES, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABELS), index=NAMES, columns=NAMES))
    print(f"macro-F1 = {macro_f1(y_true, y_pred):.3f}  (95% CI {lo:.3f}–{hi:.3f})")
    return {"model": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": macro_f1(y_true, y_pred),
            "macro_f1_lo": lo, "macro_f1_hi": hi}