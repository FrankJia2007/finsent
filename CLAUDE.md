# FinSent — 项目说明（给 Claude 的上下文）

> 读完这个文件就不用再去探索项目状态了。改动了结构/结论请顺手更新本文件。

## 1. 这是什么

**金融新闻情绪分析**，从 benchmark 做到上线看板。既是学习项目（复现并升级 2024 年一门 NLP for Finance 课程作业），也是简历级作品。核心目标**不是刷高分**，而是能讲清每一步为什么这么做、结果说明什么、局限在哪。

四层结构，逐层递进：

| 层 | 问题 | 手段 |
|---|---|---|
| **研究 Research** | 在干净基准上，各类模型差多少？ | Financial PhraseBank 上对比 majority / TF-IDF+LR / 3×encoder / LLM，统一指标 + bootstrap CI |
| **泛化 Generalization** | 换到真实世界还准吗？ | 2026 年真实标题手标 gold set 测 domain shift；再把 LLM 蒸馏成小模型 |
| **工程 Engineering** | 能每天自动跑吗？ | GitHub Actions 定时抓新闻 → 小模型打分 → Parquet → Streamlit 看板 |
| **金融 Finance** | 情绪有预测力吗？ | lead-lag 相关、Granger 因果、Rank IC、简单多空回测 |

主指标：**macro-F1**（不是 accuracy —— 见第 6 节，neutral 占 59.4%，accuracy 会骗人）。

## 2. 当前进度

- [x] **Phase 0** 环境 + GitHub 仓库
- [x] **Phase 1** 无泄漏切分 + EDA
- [x] **Phase 2** `evaluate.py` + majority/TF-IDF baseline（结论见第 6 节）
- [ ] **Phase 3** 微调 encoder ← **现在在这里**。`train_encoder.py` 已写好并改过，还没跑
- [ ] Phase 4 LLM zero/few-shot · Phase 5 benchmark 汇总 + 错误分析
- [ ] Phase 6 gold set / domain shift · Phase 7 蒸馏
- [ ] Phase 8 每日流水线 + 看板 · Phase 9 信号分析 · Phase 10 README/简历

## 3. 目录结构

```
finsent/
├── CLAUDE.md                 # 本文件
├── pyproject.toml            # uv 管理；uv_build 后端
├── .env                      # ANTHROPIC_API_KEY / FINNHUB_API_KEY（gitignored）
├── data/
│   ├── raw/                  # PhraseBank zip + 解压（gitignored，可重新下载）
│   ├── processed/            # train/val/test.parquet ← 已生成，建议提交
│   ├── gold/                 # Phase 6 手标真实新闻（空）
│   ├── silver/               # Phase 7 LLM 银标签（空）
│   └── live/                 # Phase 8 每日抓取（空）
├── notebooks/01_eda.ipynb    # Phase 1 EDA
├── src/finsent/
│   ├── data.py               # 路径常量 + 下载/解析/切分 PhraseBank
│   ├── evaluate.py           # macro_f1 / bootstrap_ci / summarize ← 所有模型共用
│   ├── baselines.py          # Phase 2：DummyClassifier + TF-IDF/LR
│   └── train_encoder.py      # Phase 3：HF Trainer 微调
├── app/                      # Phase 8 Streamlit（空）
├── outputs/                  # results_*.csv、preds/、runs/（gitignored）
└── models/                   # 保存的权重（gitignored）
```

待建：`llm_classify.py`(P4)、`news.py`(P6)、`market.py`+`pipeline.py`(P8)、`.github/workflows/daily.yml`(P8)

## 4. 环境与常用命令

Python 3.11 + uv。**版本比教程假设的新一个大版本**：`transformers 5.23`、`torch 2.15`、`numpy 2.4`、`pandas 3.1`、`datasets 5.2`。已验证 `eval_strategy` / `processing_class` 在 v5 仍可用。

本机是 Mac（**MPS 可用，无 CUDA**）→ 适合跑 baseline 和验证脚本能通；Phase 3 的 18 次训练去 Colab。

```bash
uv run python -m finsent.data              # 重建切分（确定性，seed=42）
uv run python -m finsent.baselines         # 必须在项目根目录运行（见第 5 节）
uv run python -m finsent.train_encoder --model answerdotai/ModernBERT-base --seed 1
```

Colab 上装包**不要让 pip 重装 torch**（`torch>=2.15` 会破坏 Colab 的 CUDA 环境）：

```python
!pip install -e . --no-deps -q
!pip install -q "transformers>=5.0" datasets accelerate
# 兜底：sys.path.insert(0, "/content/finsent/src")
```

## 5. 代码约定

- **路径一律用 `data.py` 里的常量**，不要写相对路径。`PROJECT_ROOT = Path(__file__).resolve().parents[2]`，派生出 `RAW_DIR / PROCESSED_DIR / GOLD_DIR / SILVER_DIR / LIVE_DIR / OUTPUTS_DIR / MODELS_DIR`。
  - ✅ 全部文件已统一（2026-11-18）。已从 `/tmp` 运行验证：`baselines.py` 的 8 个分数与项目根运行时完全一致，产物正确落在项目内。
  - ⚠️ argparse 的 `help=` 字符串里写 `%` 必须转义成 `%%`，否则 `--help` 直接抛 `ValueError`（中文 help 里写 "100%" 踩过一次）。
- **标签顺序全项目固定** `{0: negative, 1: neutral, 2: positive}`（`data.py` 的 `LABEL2ID` / `ID2LABEL`）。任何外部模型都必须按 `model.config.id2label` 映射过来，不能假设一致。
- **一把尺子量所有模型**：所有评估走 `evaluate.summarize()`，自带 bootstrap 95% CI。不要为某个模型另写指标函数，否则 benchmark 不可比。
- 切分策略：在**最全的 50agree** 上做一次 70/15/15 分层切分（3,386 / 726 / 726），再给每行标 `all_agree` 布尔列。这样无论用哪个子集训练，测试集都不重叠 —— 子集是嵌套的（AllAgree ⊂ 75 ⊂ 66 ⊂ 50），直接按子集切会泄漏。
- 每个模型都在**两个测试集**上报分：`test=allagree`（347 句，干净）和 `test=50agree`（726 句，完整含模糊样本）。
- 探索在 `notebooks/`，定型的逻辑搬进 `src/finsent/` 写成函数。

## 6. 已确立的结果与决策（不要重算）

**数据**：PhraseBank 50agree 去重后约 4,840 句。类别分布 negative 12.5% / neutral 59.4% / positive 28.1%。句长 mean 23 词、p99 50 词 → `max_length=64` 足够（attention 是 O(n²)，128 是浪费）。

**Phase 2 benchmark（macro-F1）**：

| 模型 | 训练集 | test=allagree | test=50agree |
|---|---|---|---|
| majority | 任意 | 0.260 | 0.248（accuracy 0.594） |
| tfidf_lr | allagree (1,569) | **0.881** [.836,.919] | 0.680 [.635,.720] |
| tfidf_lr | 50agree (3,386) | 0.861 [.817,.907] | **0.750** [.712,.785] |

**三个结论**：

1. **用 50agree 训练是默认选择。** 在完整测试集上显著更好（CI 几乎不重叠），在干净测试集上不显著更差（CI 大幅重叠）。⚠️ 但存在**混淆变量**：50agree 同时有 2.16× 的数据量，无法区分"模糊样本有价值"和"单纯数据多"。对照实验：从 50agree 抽 1,569 句再训练。
2. **错误集中在"有没有信号"，不在方向。** 最佳模型 155 个错误中 positive↔neutral 占 66%，negative↔positive 仅 13%。这个模式和人类标注员的分歧模式重合 → 部分错误是任务内在模糊性，不是模型能力问题。**Phase 3 该盯 positive/negative 的 recall（现 0.696/0.692），不是总体 accuracy。**
3. **没有 shortcut。** TF-IDF 高权重特征里一个公司名都没有（`min_df=2` 过滤掉了只出现一次的公司名）。neutral 最强特征是 `is/approximately/includes/no/be/other` —— 全是描述性词，**也全在 NLTK 停用词表里**，这就是"故意不去停用词"的实证。

## 7. 已知的坑

- **FinBERT (`ProsusAI/finbert`) 有两个独立问题**：
  1. label 顺序是 `{0:positive, 1:negative, 2:neutral}`，和本项目相反。且它本身就是 3 分类 → `ignore_mismatched_sizes=True` **不生效**，旧分类头会被原样加载再贴上错误标签，**静默错位不报错**。`train_encoder.py` 的 `build_model()` 已显式检测并重置分类头。
  2. 它**在 PhraseBank 上微调过**（作者自己的切分）→ 我们的 test 可能被污染，分数偏高。重置头解决不了 body 里的污染。建议额外跑一次 FinBERT **不微调**的 zero-shot 推理，把污染量化出来。ModernBERT / DeBERTa-v3 没见过 PhraseBank，是干净对照。
- 停用词陷阱：绝不要加 `stop_words="english"`，表里含 `not/no/up/down/above/below`，`profit down` → `profit` 标签直接反转。
- `transformers` 旧参数名 `evaluation_strategy` 已移除 → 用 `eval_strategy`；`tokenizer=` → `processing_class=`。
- `train_encoder.py` 里 `use_bf16/use_fp16` 都挂在 `torch.cuda.is_available()` 上 → Mac 上双 False，全程 fp32（能跑，只是慢）。
- DeBERTa-v3 的 tokenizer 需要 `sentencepiece`（已装）。
- git 现在只有一个 commit (`init project`)，**`src/` 全部未提交**。去 Colab 前必须 commit + push，否则 clone 到的是空壳。`data/processed/` 未被 gitignore，建议一起提交以保证两端用同一份切分。

## 8. Phase 3 待办

1. ~~修相对路径~~ ✅ 已完成并验证
2. ~~`transformers` 约束提到 `>=5.0`~~ ✅ 已完成
3. 本地用 ModernBERT 单跑一次验证脚本能通，再上 Colab ← **下一步**
4. Colab 跑 3 模型 × 3 seed × 2 训练集 = 18 次，`outputs/preds/*.parquet` 下载回本地
5. 建议加 `--subsample N`，补第 6 节结论 1 的等量对照实验
6. 建议加 FinBERT zero-shot 一行，量化污染
7. 验收：最好的模型在 `test=50agree` 上应明显超过 TF-IDF 的 **0.750**。**若低于它，99% 是训练代码有 bug**（label 映射 / 学习率 / padding），不是 BERT 不行

## 9. 配套笔记

详细教程和概念讲解在 Obsidian vault（不在本仓库）：

- `NLP_for_Finance_学习笔记.md` — 2341 行主笔记，Part A 概念 + Part B 的 Phase 0–10 步骤与代码
- `Finsent - 补充笔记.md` — Phase 2 的深入分析与结果解读

路径：`~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Others/`
