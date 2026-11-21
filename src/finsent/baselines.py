import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from finsent.data import OUTPUTS_DIR, PROCESSED_DIR
from finsent.evaluate import summarize

# 用 data.py 的绝对路径常量，从任何目录运行都正确
train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

results = []
for train_subset in ["allagree", "50agree"]:
    tr = train[train.all_agree] if train_subset == "allagree" else train

    # 1) 最笨的 baseline：永远猜最多的那一类
    dummy = DummyClassifier(strategy="most_frequent").fit(tr["sentence"], tr["label"])

    # 2) TF-IDF（单词 + 双词）+ 逻辑回归
    #    - 不去停用词！保留 not / down / up 这些关键信号
    #    - class_weight="balanced" 处理类别不平衡
    tfidf_lr = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0),
    ).fit(tr["sentence"], tr["label"])

    for model_name, model in [("majority", dummy), ("tfidf_lr", tfidf_lr)]:
        # 写出 preds（和 encoder / LLM 同格式），这样 baseline 也能参与 paired bootstrap。
        # 文件名里 model 用连字符（tfidf-lr）：collection.py 按第一个「_」切分 model 和训练集
        run = f"{model_name.replace('_', '-')}_{train_subset}_s0"
        out = test[["sentence", "label", "all_agree"]].copy()
        out["pred"] = model.predict(test["sentence"])
        if hasattr(model, "predict_proba"):
            out[["p_neg", "p_neu", "p_pos"]] = model.predict_proba(test["sentence"])
        out["model"], out["seed"] = run.rsplit("_s", 1)[0], 0
        (OUTPUTS_DIR / "preds").mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUTPUTS_DIR / "preds" / f"{run}.parquet", index=False)

        # 每个模型在两个测试集上都评估：简单的 AllAgree 子集 + 完整的 50Agree
        for test_subset, te in [("allagree", test[test.all_agree]), ("50agree", test)]:
            row = summarize(te["label"], model.predict(te["sentence"]),
                            name=f"{model_name} | train={train_subset} | test={test_subset}")
            row.update(train_subset=train_subset, test_subset=test_subset, seed=0)
            results.append(row)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(OUTPUTS_DIR / "results_baseline.csv", index=False)

# 3) 可解释性：每个类别权重最大的 10 个特征
vec = tfidf_lr.named_steps["tfidfvectorizer"]
lr = tfidf_lr.named_steps["logisticregression"]
feature_names = vec.get_feature_names_out()
for class_idx, class_id in enumerate(lr.classes_):
    top = np.argsort(lr.coef_[class_idx])[-10:][::-1]
    print(class_id, list(feature_names[top]))