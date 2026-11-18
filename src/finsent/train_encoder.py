
"""用 Hugging Face Trainer 微调一个 encoder 分类模型。

用法示例：
  python -m finsent.train_encoder --model ProsusAI/finbert --seed 42 --only_allagree
  python -m finsent.train_encoder --model answerdotai/ModernBERT-base --seed 1
  python -m finsent.train_encoder --model microsoft/deberta-v3-base \
         --train_file data/silver/train.parquet        # Phase 7 蒸馏时用
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

from finsent.data import ID2LABEL, LABEL2ID, MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    # zero_division=0：早期 epoch 模型可能完全不预测某一类，否则会刷一堆 UndefinedMetricWarning
    return {"accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}


def build_model(model_name: str):
    """加载预训练 encoder + 3 分类头。

    ⚠️ 关键细节：`ignore_mismatched_sizes=True` 只在**尺寸不匹配**时重新初始化分类头。
    ProsusAI/finbert 本身就是 3 分类（id2label = {0:positive, 1:negative, 2:neutral}），
    尺寸恰好匹配 → 它训练好的头会被**原样加载**，然后被我们用另一套顺序
    （{0:negative, 1:neutral, 2:positive}）重新贴标签 —— 系统性错位，且不报任何错。
    所以这里显式检测并重置，保证三个模型都从「预训练 body + 随机 head」出发，
    比较的是 body 的质量，而不是让 FinBERT 带着一个已在本任务训练过的头抢跑。
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    cfg = AutoConfig.from_pretrained(model_name)
    orig_labels = {int(k): str(v).lower() for k, v in (cfg.id2label or {}).items()}
    ours = {i: ID2LABEL[i] for i in range(3)}
    head_was_loaded = cfg.num_labels == 3          # 尺寸匹配 → 旧头被加载进来了
    if head_was_loaded and orig_labels != ours:
        head = getattr(model, "classifier", None)
        if isinstance(head, nn.Linear):
            nn.init.normal_(head.weight, std=getattr(cfg, "initializer_range", 0.02))
            if head.bias is not None:
                nn.init.zeros_(head.bias)
            print(f"[info] 分类头已重置：原模型 label 顺序 {orig_labels} ≠ 本项目 {ours}")
        else:
            print(f"[warn] 原模型是 3 分类且 label 顺序不同，但未找到 nn.Linear 分类头，"
                  f"请手动确认 {type(model).__name__} 的头结构")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="answerdotai/ModernBERT-base")
    p.add_argument("--seed", type=int, default=42)
    # 默认值用绝对路径（基于 data.py 的 __file__），所以从任何目录运行都正确
    p.add_argument("--train_file", default=str(PROCESSED_DIR / "train.parquet"))
    p.add_argument("--val_file", default=str(PROCESSED_DIR / "val.parquet"))
    p.add_argument("--test_file", default=str(PROCESSED_DIR / "test.parquet"))
    # help 里的 % 必须写成 %%：argparse 会对 help 字符串做 % 格式化
    p.add_argument("--only_allagree", action="store_true", help="只用 100%% 一致的句子训练")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=32)
    # 64 而不是 128：Phase 1 EDA 测出句长 p99 = 50 词，64 覆盖 >99%；
    # attention 是 O(n²)，用 128 等于算力翻四倍去捞不到 1% 的尾部样本
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--save_model", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)   # 固定 Python / numpy / torch 的随机种子

    # resolve()：传相对路径时按当前目录解析成绝对路径，后面比较和读取都用它
    train_path = Path(args.train_file).resolve()
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(Path(args.val_file).resolve())
    test = pd.read_parquet(Path(args.test_file).resolve())
    if args.only_allagree:
        train, val = train[train.all_agree], val[val.all_agree]

    # ---- 1. Tokenize ----
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        # 自动加 [CLS]/[SEP]、截断；不在这里 padding，交给 DataCollator 做动态 padding
        return tokenizer(batch["sentence"], truncation=True, max_length=args.max_length)

    def to_ds(df):
        return Dataset.from_pandas(df[["sentence", "label"]].reset_index(drop=True)).map(tokenize, batched=True)

    ds_train, ds_val, ds_test = to_ds(train), to_ds(val), to_ds(test)

    # ---- 2. Model：预训练 encoder + 新的 3 分类头（含 FinBERT label 顺序修正）----
    model = build_model(args.model)

    # ---- 3. 训练参数 ----
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16       # 比如 Colab T4
    # 训练数据标签：allagree / 50agree / silver（Phase 7 的蒸馏数据所在文件夹名）
    # 注意：用 resolve 后的 Path 比较，不能比字符串——否则绝对路径永远匹配不上
    if args.only_allagree:
        data_tag = "allagree"
    elif train_path.parent == PROCESSED_DIR and train_path.name == "train.parquet":
        data_tag = "50agree"
    else:
        data_tag = train_path.parent.name
    run_name = f"{args.model.split('/')[-1]}_{data_tag}_s{args.seed}"
    training_args = TrainingArguments(
        output_dir=str(OUTPUTS_DIR / "runs" / run_name),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        warmup_steps=30,                 # 前 30 步学习率从 0 慢慢升上去，训练更稳
        eval_strategy="epoch",           # 每个 epoch 结束在验证集上评估
        save_strategy="epoch",
        load_best_model_at_end=True,     # 训练结束后自动加载验证集上最好的那个 checkpoint
        metric_for_best_model="macro_f1",
        save_total_limit=1,
        logging_steps=20,
        bf16=use_bf16, fp16=use_fp16,
        seed=args.seed,
        report_to="none",                # 想用 W&B 就改成 "wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),   # 动态 padding
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # ---- 4. 在测试集上预测，保存概率（后面做 benchmark 和错误分析都要用）----
    logits = trainer.predict(ds_test).predictions
    if isinstance(logits, tuple):        # 少数模型会额外返回其他输出，只取 logits
        logits = logits[0]
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    out = test[["sentence", "label", "all_agree"]].copy()
    out["pred"] = probs.argmax(-1)
    out[["p_neg", "p_neu", "p_pos"]] = probs
    out["model"], out["seed"] = run_name.rsplit("_s", 1)[0], args.seed
    preds_dir = OUTPUTS_DIR / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    out_path = preds_dir / f"{run_name}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Saved predictions to {out_path}")

    if args.save_model:
        trainer.save_model(str(MODELS_DIR / run_name))   # tokenizer 会一起保存


if __name__ == "__main__":
    main()