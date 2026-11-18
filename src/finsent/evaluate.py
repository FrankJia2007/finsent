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