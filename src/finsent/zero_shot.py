"""直接用一个**已经训练好的** 3 类情感模型做推理（不在我们的 train set 上训练）。

主要用途：FinBERT zero-shot，量化 contamination。
FinBERT 在 PhraseBank 上 fine-tune 过（作者自己的切分），如果它不经过我们任何训练就能在
我们的 test set 上拿到接近 fine-tune 后的分数，说明它的高分主要来自「本来就会这个 benchmark」。

用法：
  python -m finsent.zero_shot                         # 默认 ProsusAI/finbert
  python -m finsent.zero_shot --model <hf_id> --split test

输出和 train_encoder.py 同格式：outputs/preds/<model>_zeroshot_s0.parquet，Phase 5 汇总可直接读。
"""
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finsent.data import ID2LABEL, LABEL2ID, OUTPUTS_DIR, PROCESSED_DIR
from finsent.evaluate import summarize


def label_permutation(model) -> list[int]:
    """返回 perm，使得 logits[:, perm] 的列顺序 = 本项目的 {0: negative, 1: neutral, 2: positive}。

    必须通过模型自己的 config.id2label 按名字映射，绝不假设顺序一致（CLAUDE.md §5）。
    FinBERT 是 {0: positive, 1: negative, 2: neutral} → perm = [1, 2, 0]。
    """
    theirs = {str(name).lower(): int(i) for i, name in model.config.id2label.items()}
    missing = set(LABEL2ID) - set(theirs)
    if missing:
        raise ValueError(f"模型的 id2label {model.config.id2label} 缺少 {missing}，无法按名字映射")
    return [theirs[ID2LABEL[i]] for i in range(3)]


@torch.no_grad()
def predict_probs(model, tokenizer, sentences, batch_size=64, max_length=64, device="cpu"):
    probs = []
    for i in range(0, len(sentences), batch_size):
        batch = tokenizer(sentences[i:i + batch_size], truncation=True, max_length=max_length,
                          padding=True, return_tensors="pt").to(device)
        probs.append(torch.softmax(model(**batch).logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ProsusAI/finbert")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max_length", type=int, default=64)   # 和 fine-tune 时保持一致
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, dtype=torch.float32)
    model.to(device).eval()
    perm = label_permutation(model)
    print(f"[info] {args.model} id2label={model.config.id2label} → column order {perm} "
          f"= {[ID2LABEL[i] for i in range(3)]}")

    df = pd.read_parquet(PROCESSED_DIR / f"{args.split}.parquet")
    probs = predict_probs(model, tokenizer, df["sentence"].tolist(),
                          max_length=args.max_length, device=device)[:, perm]

    run_name = f"{args.model.split('/')[-1]}_zeroshot_s0"
    out = df[["sentence", "label", "all_agree"]].copy()
    out["pred"] = probs.argmax(-1)
    out[["p_neg", "p_neu", "p_pos"]] = probs
    out["model"], out["seed"] = run_name.rsplit("_s", 1)[0], 0

    if args.split == "test":   # 只有 test 的预测写进 preds/，避免和 benchmark 混在一起
        preds_dir = OUTPUTS_DIR / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(preds_dir / f"{run_name}.parquet", index=False)
        print(f"Saved predictions to {preds_dir / f'{run_name}.parquet'}")

    for tag, sub in [("allagree", out[out["all_agree"]]), ("50agree", out)]:
        summarize(sub["label"], sub["pred"], name=f"{run_name} | split={args.split} | {tag}")


if __name__ == "__main__":
    main()
