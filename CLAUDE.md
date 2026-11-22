# FinSent — Project Context for Claude

> Reading this file should remove any need to explore the repo to figure out its state.
> If you change the structure or establish a new result, update this file in the same pass.

## 1. What this is

**Financial news sentiment analysis**, taken from benchmark all the way to a live dashboard. It is both a learning project (redoing and upgrading a 2024 "NLP for Finance" course assignment) and a portfolio piece. The goal is **not to maximize a score** — it is to be able to explain, for every step, why it was done that way, what the result means, and where the limits are.

Four layers, each building on the previous:

| Layer | Question | Approach |
|---|---|---|
| **Research** | On a clean benchmark, how far apart are the model families? | Compare majority / TF-IDF+LR / 3×encoder / LLM on Financial PhraseBank, one metric + bootstrap CI |
| **Generalization** | Does it still hold in the real world? | Hand-labeled gold set of 2026 headlines to measure domain shift; then distill the LLM into a small model |
| **Engineering** | Can it run unattended every day? | GitHub Actions → fetch headlines → score with small model → Parquet → Streamlit dashboard |
| **Finance** | Does sentiment have predictive power? | Lead-lag correlation, Granger causality, Rank IC, simple long-short backtest |

Primary metric: **macro-F1**, not accuracy. See §6 — neutral is 59.4% of the data, so accuracy flatters useless models.

## 2. Current status

- [x] **Phase 0** Environment + GitHub repo
- [x] **Phase 1** Leakage-free split + EDA
- [x] **Phase 2** `evaluate.py` + majority/TF-IDF baselines (findings in §6)
- [x] **Phase 3** Fine-tune encoders — 27-run Colab grid + FinBERT zero-shot, analyzed in §6
- [x] **Phase 4** LLM zero/few-shot — Haiku 5.5 (zero/few-shot, effort low/medium) + Opus 5.5 zero-shot, flip-rate tests, analyzed in §6
- [x] **Phase 5** benchmark aggregation + error analysis — `collection.py` final table, `error_analysis.py` (errors vs annotator agreement, hard core), findings in §6
- [ ] **Phase 6** gold set / domain shift ← **current position** — guidelines written (`data/gold/GUIDELINES.md`), labeling not started; also decides the Phase 7 teacher (§6) · Phase 7 distillation
- [ ] Phase 8 daily pipeline + dashboard · Phase 9 signal analysis · Phase 10 README/résumé

## 3. Layout

```
finsent/
├── CLAUDE.md                 # this file
├── pyproject.toml            # managed by uv; uv_build backend
├── .env                      # ANTHROPIC_API_KEY / FINNHUB_API_KEY (gitignored)
├── data/
│   ├── raw/                  # PhraseBank zip + extracted txt (gitignored, re-downloadable)
│   ├── processed/            # train/val/test.parquet — committed on purpose, see §5
│   ├── gold/                 # Phase 6: GUIDELINES.md (annotation rules, written before labeling); gold.csv to come
│   ├── silver/               # Phase 7 LLM silver labels (empty)
│   └── live/                 # Phase 8 daily scrapes (empty)
├── notebooks/
│   ├── 01_eda.ipynb          # Phase 1 EDA
│   └── 02_colab_grid.ipynb   # Phase 3 grid on Colab: clone → install → Drive-linked outputs → run_grid → sanity check → zip
├── src/finsent/
│   ├── data.py               # path constants + download/parse/split PhraseBank
│   ├── evaluate.py           # macro_f1 / bootstrap_ci / paired_bootstrap / summarize — shared by every model
│   ├── baselines.py          # Phase 2: DummyClassifier + TF-IDF/LR (also writes preds, for paired bootstrap)
│   ├── train_encoder.py      # Phase 3: HF Trainer fine-tuning (one run)
│   ├── run_grid.py           # Phase 3: runs the model × seed × train-set grid, resumable
│   ├── zero_shot.py          # Phase 3: inference with an already-trained classifier (FinBERT contamination check)
│   ├── llm_classify.py       # Phase 4: Claude classification (structured outputs, prompt-versioned cache, runner, flip test)
│   ├── collection.py         # Phase 5: every preds file → one benchmark table (CI + paired Δ vs reference) → outputs/results_benchmark.csv
│   └── error_analysis.py     # Phase 5: error rate vs annotator agreement level, hard core → outputs/error_hard_core.csv
├── app/                      # Phase 8 Streamlit (empty)
├── outputs/                  # results_*.csv, preds/, runs/ (gitignored)
└── models/                   # saved weights (gitignored)
```

Still to be written: `news.py` (P6), `market.py` + `pipeline.py` (P8), `.github/workflows/daily.yml` (P8).

## 4. Environment and commands

Python 3.11 + uv. **The installed stack is one major version ahead of what the tutorial assumes**: `transformers 5.23`, `torch 2.15`, `numpy 2.4`, `pandas 3.1`, `datasets 5.2`. Verified that `eval_strategy` and `processing_class` still exist in transformers v5.

This machine is a Mac (M5, **16 GB unified memory**) — **MPS available, no CUDA**. A full ModernBERT-base run (50agree, 4 epochs, fp32) took ~5 min on MPS, but DeBERTa-v3-base in fp32 **thrashed memory** (swap 8.8/9.2 GB, steps went from 1.15 it/s to 13–34 s/it, ETA 2–3 h) and was stopped. → Local is fine for smoke tests and single ModernBERT/FinBERT runs; **the grid goes to Colab**, all models on one platform so precision (fp32 on MPS vs mixed precision on CUDA) is not a confound between models.

```bash
uv run python -m finsent.data              # rebuild splits (deterministic, seed=42)
uv run python -m finsent.baselines         # runnable from any directory
uv run python -m finsent.train_encoder --model answerdotai/ModernBERT-base --seed 1
uv run python -m finsent.llm_classify --limit 5                          # LLM smoke test: prints preds + cost estimate, writes nothing
uv run python -m finsent.llm_classify --model claude-opus-5-5           # full test set → outputs/preds/opus-5-5_zeroshot-low_s0.parquet
uv run python -m finsent.llm_classify --model claude-haiku-5-5 --flip_test 100   # sampling-noise check
uv run python -m finsent.collection         # final benchmark table, ~1.5 min (1000 bootstrap resamples per CI/Δ; --n_boot 200 for a quick look)
uv run python -m finsent.error_analysis     # errors by annotator agreement level + the sentences every clean model gets wrong
```

**Claude API**: needs `ANTHROPIC_API_KEY` in `.env` — a Claude Pro subscription does **not** include API credit (separate prepaid billing in the Console). Every LLM result is cached in `outputs/llm_cache.jsonl` keyed by prompt version + model + few_shot + effort + replicate + sentence, so reruns are free and resumable. Phase 4 total spend ≈ $2.5 (Haiku runs ~$0.05 each, Opus run $2.05, flip tests $0.29).

Colab workflow lives in `notebooks/02_colab_grid.ipynb` (open from GitHub; the repo is public). `outputs/preds` and `outputs/logs` are symlinked to `MyDrive/finsent_outputs/` so a disconnect loses nothing — rerun all cells and `run_grid` resumes. On Colab, **do not let pip reinstall torch** — the `torch>=2.15` requirement will clobber Colab's CUDA build:

```python
!pip install -e . --no-deps -q
!pip install -q "transformers>=5.0" datasets accelerate
# fallback if the editable install misbehaves:
# sys.path.insert(0, "/content/finsent/src")
```

## 5. Code conventions

- **All paths go through the constants in `data.py`** — never hardcode a relative path. `PROJECT_ROOT = Path(__file__).resolve().parents[2]`, from which `RAW_DIR / PROCESSED_DIR / GOLD_DIR / SILVER_DIR / LIVE_DIR / OUTPUTS_DIR / MODELS_DIR` are derived.
  - Verified 2026-11-18 by running `baselines.py` from `/tmp`: all 8 scores identical to a run from the project root, and artifacts still landed inside the project.
  - `data/processed/*.parquet` is committed deliberately, so Colab and local are guaranteed to use the same split rather than relying on `train_test_split` being reproducible across library versions.
- **One label order everywhere**: `{0: negative, 1: neutral, 2: positive}` (`LABEL2ID` / `ID2LABEL` in `data.py`). Any external model must be mapped through its own `model.config.id2label` — never assume it agrees.
- **One ruler for every model**: all evaluation goes through `evaluate.summarize()`, which carries a bootstrap 95% CI. Do not write a separate metric function for a particular model; the benchmark stops being comparable.
- Split strategy: a single stratified 70/15/15 split over the **widest subset, 50agree** (3,386 / 726 / 726), with a boolean `all_agree` column per row. This keeps train and test disjoint no matter which subset is trained on — the subsets are nested (AllAgree ⊂ 75 ⊂ 66 ⊂ 50), so splitting per subset would leak.
- Every model reports on **two test sets**: `test=allagree` (347 sentences, clean) and `test=50agree` (726 sentences, full, includes ambiguous cases). For mechanism questions, also split test=50agree into its two **disjoint** halves: clean (347, `all_agree`) and ambiguous (379, `~all_agree`) — test=50agree contains the clean half, so comparing it with test=allagree dilutes effects.
- **Comparing two models = `evaluate.paired_bootstrap()`**, not eyeballing whether two independent CIs overlap. Same resampled sentences for both sides; pass a list of per-seed prediction arrays to average over seeds. All files in `outputs/preds/` share the test-set row order (assert it before pairing).
- Explore in `notebooks/`; once logic is settled, move it into `src/finsent/` as functions.
- In argparse `help=` strings, a literal `%` must be escaped as `%%`, or `--help` raises `ValueError` (already hit once via "100%").

## 6. Established results and decisions — do not recompute

**Data**: PhraseBank 50agree, ~4,840 sentences after dedup. Class balance negative 12.5% / neutral 59.4% / positive 28.1%. Sentence length mean 23 words, p99 50 words → `max_length=64` is sufficient; attention is O(n²), so 128 quadruples the cost to catch under 1% of the tail.

**Phase 2 benchmark (macro-F1)**:

| Model | Train set | test=allagree | test=50agree |
|---|---|---|---|
| majority | either | 0.260 | 0.248 (accuracy 0.594) |
| tfidf_lr | allagree (1,569) | **0.881** [.836, .919] | 0.680 [.635, .720] |
| tfidf_lr | 50agree (3,386) | 0.861 [.817, .907] | **0.750** [.712, .785] |

**Three findings**:

1. **Train on 50agree by default.** It is clearly better on the full test set (CIs barely overlap) and not meaningfully worse on the clean one (CIs overlap heavily). The size confound (50agree carries 2.16× the data) was **resolved in Phase 3** by an equal-size control — both effects are real and separate; see "Equal-size control" below.
2. **Errors are about whether a signal exists, not about its direction.** Of the best model's 155 errors, positive↔neutral accounts for 66% and negative↔positive only 13%. This mirrors the disagreement pattern among the human annotators → a real share of the remaining error is task-intrinsic ambiguity, not model capacity. **For Phase 3, watch positive/negative recall (currently 0.696 / 0.692), not overall accuracy.**
3. **No shortcut learning.** Not a single company name appears among the top-weighted TF-IDF features; `min_df=2` filtered them out, since most occur once. The strongest neutral features are `is / approximately / includes / no / be / other` — purely descriptive words, **and all of them are in the NLTK English stopword list**. That is the empirical case for deliberately not removing stopwords.

**Phase 3 benchmark** (Colab grid, lr 2e-5, 4 epochs, bs 32, max_len 64; mean ± std over seeds {1, 2, 3}; per-run bootstrap CI half-width on test=50agree ≈ ±0.028):

| Model | Train set | test=allagree | test=50agree | neg / pos recall (50agree) |
|---|---|---|---|---|
| tfidf_lr (Phase 2) | 50agree | 0.861 | 0.750 | 0.692 / 0.696 |
| ModernBERT-base | 50agree | 0.954 ± .007 | 0.849 ± .008 | 0.875 / 0.758 |
| DeBERTa-v3-base | 50agree | 0.952 ± .011 | 0.866 ± .001 | 0.930 / 0.806 |
| FinBERT ⚠️ contaminated | 50agree | 0.966 ± .010 | 0.884 ± .007 | 0.901 / 0.861 |
| FinBERT **zero-shot** (no training on our data) | — | 0.947 | 0.881 | 0.956 / 0.907 |

(A local MPS fp32 ModernBERT s42 run gave 0.949 / 0.839 — consistent; kept as a reference, not in the means.)

- **All three encoders clear acceptance** (+0.10–0.13 over TF-IDF; CIs disjoint).
- **test=allagree is saturated** (0.95–0.97 for every encoder) — the clean test cannot rank encoders; test=50agree, specifically its ambiguous half, is where they separate.
- **Seed noise (std 0.001–0.008) ≪ test-sampling noise (±0.028)**: uncertainty comes from having 726 test sentences, not from seeds.
- **Clean-model ranking** (paired bootstrap, 3-seed means): DeBERTa − ModernBERT = +0.018 [−0.001, +0.037] on test=50agree — **borderline**. Split: clean −0.002 (tie), ambiguous **+0.033 [+0.004, +0.064]**. DeBERTa's edge is entirely on ambiguous sentences.
- **FinBERT contamination, quantified**: fine-tuned − zero-shot = **+0.003 [−0.017, +0.022]** on test=50agree — our training adds nothing (only +0.019 on the clean half). On the ambiguous half, zero-shot FinBERT (0.827) is the **best model of all** (FinBERT-ft 0.816, DeBERTa 0.794, ModernBERT 0.761): with no exposure to our labels, it best matches PhraseBank's labeling convention exactly where convention matters most. Other signs: epoch-1 val macro-F1 0.85–0.89 (ModernBERT 0.69–0.78), and halving the data costs it only 0.011. → **FinBERT is excluded from the body-quality ranking** and reported separately. Not yet separable: seen-PhraseBank vs financial-domain pretraining — the Phase 6 gold set (sentences it cannot have seen) is the clean test.
- **Finding 2 strengthens**: errors per run 155 (TF-IDF) → 104 / 88 / 77 (ModernBERT / DeBERTa / FinBERT); positive↔neutral 71–79% of errors, negative↔positive only 4–7%. Positive remains the hardest class for every model.

**Equal-size control for Finding 1** (train=allagree vs train=50agree-n1569, both 1,569 rows and 200 steps; best epoch ≈ 4 in both, so checkpoint selection on different val sets did not matter). Macro-F1 on the two disjoint halves of the test set:

| Model | Train set | clean (347) | ambiguous (379) |
|---|---|---|---|
| ModernBERT | allagree | **0.919** | 0.610 |
| | 50agree-n1569 | 0.896 | **0.680** |
| DeBERTa | allagree | 0.949 | 0.639 |
| | 50agree-n1569 (s1 only, see §8) | 0.950 | **0.760** |
| FinBERT | allagree | **0.963** | 0.747 |
| | 50agree-n1569 | 0.951 | **0.806** |

- Paired bootstrap, n1569 − allagree on the **ambiguous** half: +0.070 / +0.120 / +0.059 — **significant for all three**. On the **clean** half: −0.024 / +0.001 / −0.012 — **significant for none** (ModernBERT closest: [−0.047, +0.003]).
- → **At equal size, seeing ambiguous examples brings a large gain on ambiguous sentences at little or no cost on clean ones.** The effect is asymmetric, not a symmetric "match the test distribution." The small clean-side dip has a simpler explanation: n1569 holds only ~730 clean rows vs 1,569.
- **Data size is a separate, larger effect**: n1569 → full 50agree adds +0.070 on test=50agree for ModernBERT (+0.011 for FinBERT, which already knows the task). Caveat: more data also means more steps (200 → 424), and n1569 ModernBERT was still improving at epoch 4, so part of this "size" effect is under-training.
- Open caveat: ambiguous labels are annotator majority votes, so the gain may be learning *annotator convention* rather than financial semantics. Phase 6 (new headlines, new annotator) tests whether it transfers.

**Phase 4 benchmark — LLMs, no training data** (Claude via `llm_classify.py`, structured outputs, effort `low` unless noted, single run each; failures would count as errors — there were 0):

| Model | test=allagree | test=50agree | clean (347) | ambiguous (379) | neg / neu / pos recall | cost / 726 |
|---|---|---|---|---|---|---|
| **Opus 5.5 zero-shot** | 0.953 | **0.865** [.840, .890] | 0.953 | **0.791** | 0.934 / 0.856 / **0.848** | $2.05 |
| Haiku 5.5 zero-shot | 0.963 | 0.844 [.813, .872] | 0.963 | 0.739 | 0.923 / 0.916 / 0.676 | $0.052 |
| Haiku 5.5 few-shot | 0.963 | 0.840 | 0.963 | 0.733 | 0.934 / 0.912 / 0.667 | $0.064 |
| Haiku 5.5 zero-shot, effort medium | 0.959 | 0.835 | — | — | 0.890 / 0.910 / 0.681 | $0.055 |
| *ref: DeBERTa-ft (3 seeds)* | 0.952 | 0.866 | 0.952 | 0.794 | 0.930 / 0.902 / 0.806 | — |

Paired bootstrap (A − B):

| Comparison | test=50agree | clean | ambiguous |
|---|---|---|---|
| Opus − DeBERTa | −0.001 [−0.028, +0.026] | +0.001 | −0.003 |
| Opus − Haiku | +0.022 [−0.005, +0.049] | −0.010 | **+0.051 [+0.005, +0.097]** |
| Haiku-zs − DeBERTa | −0.023 [−0.052, +0.006] | +0.011 | **−0.055 [−0.099, −0.007]** |
| Haiku-fs − Haiku-zs | −0.004 [−0.023, +0.015] | −0.000 | −0.006 |
| Haiku medium − low | −0.009 [−0.025, +0.006] | −0.004 | −0.013 |

- **Opus zero-shot ties fine-tuned DeBERTa on every slice** (all |Δ| ≤ 0.003): a general LLM with zero training data matches the best clean fine-tuned encoder.
- **Haiku's gap is model size, not "LLMs can't learn the convention"**: Haiku trails DeBERTa only on the ambiguous half (−0.055, sig.); Opus closes that gap entirely. On the clean half Haiku is the top scorer of all (0.963, n.s.).
- **Mechanism — a threshold bias, not a comprehension gap.** On the ambiguous half, Haiku predicts positive for **21%** of sentences (true 33%; errors pos→neu 64 vs neu→pos 19); Opus predicts **35%** (errors 31 vs 39, balanced). Neither model ever confuses negative with positive. 98 of Haiku's 109 errors are on the ambiguous half. Opus and Haiku disagree on 98 sentences, 84% ambiguous.
- **Neither few-shot nor more thinking moves Haiku's threshold**: few-shot −0.004, effort medium −0.009 (both n.s.); positive share on ambiguous stays 21–22%. The 3 few-shot examples are clear-cut cases, while all of Haiku's errors are borderline ones. Output is ~19–28 tokens/sentence, i.e. almost no thinking even at medium. A prompt that states the convention (e.g. "reported growth in sales/profit counts as positive") might fix it — **if tried, tune on val, never on test**.
- **Sampling noise is negligible** (temperature can no longer be set): `--flip_test 100` flipped **2/100 (Haiku), 1/100 (Opus)**, all on ambiguous sentences with low confidence (flipped mean conf 0.75 / 0.50 vs 0.83 stable). Far below the ±0.03 CI → no self-consistency voting needed for benchmarking.
- **Verbalized confidence is informative**: ECE 0.055 (Opus) / 0.042 (Haiku). Opus accuracy rises monotonically with confidence (0.55 → 0.79 → 0.90 → 0.98 → 1.00 across bins <.7 / .7–.8 / .8–.9 / .9–.95 / ≥.95); Haiku is non-monotonic at the low end (0.72 at <.7 but 0.67 at .7–.8). Both are ~99% accurate at confidence ≥ 0.9 → usable as a silver-label filter or sample weight in Phase 7. The pre-run worry that few-shot examples fixed at 0.9 would collapse confidence did not happen (21% exactly 0.9 vs 16% zero-shot).
- **Pre-registered rule and what happened**: before running, the rule was "Haiku few-shot ≈ DeBERTa (n.s.) → skip Opus; clearly lower → run Opus zero-shot". Result: −0.026 [−0.054, **+0.000**] — exactly on the boundary (and significant on the ambiguous half). Opus was run anyway because the rule's purpose (separate "small model" from "LLMs as a family") was unresolved; this added a result, it did not change any Haiku number.
- **Caveat — contamination applies to LLMs too**: PhraseBank has been public since 2014; Opus matching the annotator convention on ambiguous sentences may partly be memorization, which larger models do more of. Not testable on PhraseBank — Phase 6 again.
- **Phase 7 teacher: deferred to Phase 6.** Opus = best quality (≈$56 per 20k headlines), Haiku = 40× cheaper (≈$1.4) but would pass its positive-undercall bias to the student. "Best" depends on which labeling convention is wanted, and the Phase 6 gold set defines that — run both LLMs on the gold set (Opus < $1 per 300 headlines) and choose then.

**Phase 5 — final table** (`collection.py`; every number above reproduces from `outputs/preds/`; Δ = paired bootstrap vs DeBERTa-50agree, * = significant):

| Family | Model | Training data | test=50agree [95% CI] | clean | ambiguous | Δ vs DeBERTa |
|---|---|---|---|---|---|---|
| baseline | majority | — | 0.248 | 0.260 | 0.237 | −0.618* |
| baseline | TF-IDF + LR | 3,386 | 0.750 [.712, .785] | 0.861 | 0.648 | −0.116* |
| encoder | ModernBERT-base | 3,386 | 0.849 [.823, .872] | 0.954 | 0.761 | −0.018 |
| encoder | **DeBERTa-v3-base** | 3,386 | **0.866** [.842, .891] | 0.952 | 0.794 | ref |
| LLM | Haiku 5.5 zero-shot | none | 0.844 [.813, .872] | 0.963 | 0.739 | −0.023 |
| LLM | **Opus 5.5 zero-shot** | none | **0.865** [.840, .890] | 0.953 | 0.791 | −0.001 |
| ⚠️ contaminated | FinBERT fine-tuned / zero-shot | saw PhraseBank | 0.884 / 0.881 | 0.966 / 0.947 | 0.816 / 0.827 | +0.018 / +0.015 |

**Headline result: among clean models, fine-tuned DeBERTa and zero-shot Opus tie for first; FinBERT's lead is contamination.**

**Phase 5 — where the errors are** (`error_analysis.py`). Annotator agreement per test sentence comes from the highest nested PhraseBank subset containing it: 100% (347), 75–99% (171), 66–74% (112), 50–65% (96); labels agree across subsets (0 conflicts).

| Error rate | 100% | 75–99% | 66–74% | 50–65% |
|---|---|---|---|---|
| TF-IDF | 10.4% | 25.1% | 35.7% | 37.5% |
| ModernBERT | 3.1% | 15.0% | 31.8% | 33.7% |
| DeBERTa | 3.1% | 12.1% | 25.6% | 29.5% |
| Haiku | 3.2% | 13.5% | 33.0% | 39.6% |
| Opus | 4.3% | 11.1% | 28.6% | 34.4% |
| FinBERT-zs ⚠️ | 3.5% | 15.2% | 17.9% | 24.0% |
| *one more human annotator vs the majority (≈ 1 − agreement, rough)* | *0%* | *~1–25%* | *~26–34%* | *~35–50%* |

1. **Errors track human disagreement, monotonically, for every model.** On ambiguous strata the clean models err about as often as one more human annotator would disagree with the majority (rough reference: bins are ranges, 5–8 annotators per sentence). → The residual error there is mostly irreducible under this label set; the only stratum with clear headroom is 100% agreement, where good models are at ~3%.
2. **Clean-sentence errors are idiosyncratic, ambiguous-sentence errors are shared.** On 100%-agreement sentences no sentence is missed by ≥3 of the 5 clean models (TF-IDF, ModernBERT, DeBERTa, Haiku, Opus; multi-seed models by majority vote). On the rest, the more ambiguous the sentence, the more models miss it together.
3. **Hard core: 16 sentences (2.2%) wrong for all 5 clean models** — 0 at 100% agreement, 2 at 75%, 7 at 66%, 7 at 50%. In 94% of them all five give the **same** wrong label (models agree with each other, not with the majority). 12/16 are gold positive → predicted neutral — the Haiku threshold mechanism, shared here by every model.
4. **Manual categories of the 16** (judgment, reviewed with the user): (a) **favorable language without numbers** — 8 (growth project, continued investment, product efficiency, "well-located" apartment); PhraseBank's majority read these as positive, models as no clear price impact; a convention question. (b) **procedural disclosures with a positive label** — 4 (AGM approval, securities lending agreement, board incentive plan, ten-year cooperation); the published instruction (P1–P3 in `data/gold/GUIDELINES.md`) supports neutral, so these are **likely label noise**; Opus called them neutral at confidence 0.88–0.92. (c) **perspective / missing context** — 4 (layoffs with no stated reason, a divestment the company calls strategic, a contract that helps Ericsson inside Elisa's network, a scam using Google's name).
5. **Strongest contamination evidence so far**: on the hard core the clean models score 0% by construction, while FinBERT zero-shot gets **62%** (10/16) right — including **3 of the 4** category-(b) labels that the text does not support. FinBERT-zs also has the lowest error of any model at 50–65% agreement (24.0%), below the rough one-more-annotator reference. Recovering labels the text cannot justify points to memorization.
6. **Consequence for Phase 6 → `data/gold/GUIDELINES.md`.** Gold labels follow Malo et al.'s published instruction verbatim (keeps the convention fixed, so a score drop measures domain shift, not a convention change), plus operational rules derived from the hard core: A1 subject company's stock; A2 procedural disclosures → neutral; **A3 option (a), chosen by the user**: a concrete favorable development → positive, generic language and the company's own evaluative claims → neutral; A4 layoffs/divestments labeled by the stated reason, else neutral; A5/A6 headline-specific (price moves, mixed signals; revisit after the pilot). Applied to the 16 hard-core sentences, these rules disagree with PhraseBank on 8 (all four category-(b) cases among them).

## 7. Known pitfalls

- **FinBERT (`ProsusAI/finbert`) has two independent problems**:
  1. Its label order is `{0: positive, 1: negative, 2: neutral}` — the reverse of this project's. Because it is already a 3-class model, `ignore_mismatched_sizes=True` **does nothing**: the trained head loads as-is and then gets relabeled under our ordering, a **silent, systematic mismatch with no warning**. `build_model()` in `train_encoder.py` detects this and resets the head explicitly.
  2. It was **fine-tuned on PhraseBank** (the authors' own split), so our test set is likely contaminated and its score is inflated. Resetting the head does not fix contamination in the body. **Confirmed** by `zero_shot.py`: zero-shot FinBERT scores 0.881 on test=50agree, the same as fine-tuned FinBERT (§6). ModernBERT and DeBERTa-v3 have not seen PhraseBank and serve as clean controls.
- **Fine-tuning instability on small data (majority-class plateau).** With 1,569 rows × 4 epochs = 200 steps, DeBERTa-v3 often spends epoch 1 predicting all-neutral (val macro-F1 exactly 0.248 = the all-neutral score). In 2/3 allagree seeds it escaped by epoch 3; in 2/3 50agree-n1569 seeds it did not (s3 never predicted negative; test=50agree 0.497 / 0.586). No NaN, loss decreasing — an optimization problem, not a bug (cf. Mosbach et al. 2021, *On the Stability of Fine-tuning BERT*). Tell-tale: val macro-F1 ≈ 0.25 at epoch 1 or `n_pred_classes < 3`. Expect the same in Phase 7 distillation (small data); remedy is more steps, applied to every model in a comparison, never by re-rolling seeds for the failed runs only.
- Stopword trap: never pass `stop_words="english"`. The list contains `not / no / up / down / above / below`; drop them and `profit down` becomes `profit`, inverting the label.
- Renamed transformers arguments: `evaluation_strategy` → `eval_strategy`, `tokenizer=` → `processing_class=`.
- In `train_encoder.py`, `use_bf16` / `use_fp16` are both gated on `torch.cuda.is_available()` → both False on Mac, so training runs in full fp32. Works, just slow.
- DeBERTa-v3's tokenizer needs `sentencepiece` (installed).
- **transformers v5 loads weights in the checkpoint's stored dtype** (`dtype="auto"` is the default). `microsoft/deberta-v3-base` is stored in **fp16**, so it loaded as a pure-fp16 model and AdamW updated fp16 weights with no master copy or loss scaling: first logged loss 1217, then NaN, val loss NaN every epoch, every prediction = class 0 (argmax of NaN), macro-F1 0.07. ModernBERT and FinBERT are stored in fp32 and were unaffected. Fixed by `dtype=torch.float32` in `build_model()`. Caught via the all-one-column confusion matrix from `summarize()` and the NaN val curve from `save_state()`. **Before `run_grid.py` re-runs anything, delete broken `outputs/preds/<run>.parquet` files** — it skips any run whose preds already exist.
- **Claude API changes vs the tutorial** (all fixed in `llm_classify.py`): (1) `temperature` returns a 400 with SDK 2.x (new API version) — sampling can't be controlled, so measure it with `--flip_test`; (2) current models think by default, so `content[0]` may be a thinking block — find the `text` block, check `stop_reason`, and keep `max_tokens` well above the answer length; (3) use structured outputs (`output_config.format` JSON schema with an `enum` label) rather than regex-parsing JSON; (4) never fall back to a default class on failure — neutral is the majority class, so failures would hide as plausible errors; failures are `pred=-1` and score as wrong; (5) the cache key must include the prompt (`PROMPT_VERSION` hash), or editing the prompt silently returns stale answers; (6) `claude-haiku-4-5` from the tutorial is the previous generation and 10× the price of `claude-haiku-5-5`.
- **Server-side `fallbacks` deliberately left off** for Opus 5.5 benchmark runs: a refused request would be answered by a different model but scored under Opus's name. Refusals show up as parse failures instead (0 so far). Reconsider for the Phase 8 production pipeline.
- Remote is `git@github.com:FrankJia2007/finsent.git` (the username's capitalization changed; the old lowercase URL still works but warns on every push).

## 8. To-do

**Phase 3** (done)

1. ~~Fix relative paths~~ done and verified
2. ~~Raise the `transformers` floor to `>=5.0`~~ done
3. ~~Run ModernBERT once locally~~ done: completes in ~5 min on MPS, passes acceptance (§6). `train_encoder.py` now also prints `summarize()` on both test sets and calls `trainer.save_state()`, so `outputs/runs/<run>/trainer_state.json` keeps the full per-epoch val curve (checkpoint rotation otherwise deletes it)
4. ~~Run the grid on Colab~~ done: `python -m finsent.run_grid --with_control`, 27 runs, results in §6. Preds in `outputs/preds/`, val curves in `outputs/logs/<run>.trainer_state.json`
5. ~~Add `--subsample N` and run the equal-size control~~ done (§6)
6. ~~FinBERT zero-shot~~ done: `python -m finsent.zero_shot` → `outputs/preds/finbert_zeroshot_s0.parquet` (§6)
7. ~~Acceptance~~ passed by all three encoders (§6): the best model should clearly beat TF-IDF's **0.750** on `test=50agree`. **If it comes in lower, it is a bug in the training code 99% of the time** — label mapping, learning rate, or padding — not a limitation of BERT
8. ~~Degenerate DeBERTa small-data runs~~ **decided: report as-is (option b)**. `deberta-v3-base_50agree-n1569_s2/s3` hit the majority-class plateau (§7); that cell rests on s1 alone and is flagged in §6. Rationale: the equal-size conclusion already holds with 3 seeds each for ModernBERT and FinBERT (significant), and DeBERTa s1 agrees in direction with the largest effect — a rerun would complete the table, not change the conclusion. Revisit before Phase 7: rerun **both** small-data configs for **all three** models with more epochs (e.g. 10), since small-data stability matters for distillation

**Phase 4** (done): `llm_classify.py` fixed (§7) and extended with a runner and `--flip_test`; Haiku zero/few-shot, Haiku effort medium, Opus zero-shot, flip tests on both — results in §6.

**Next**
1. ~~Phase 5~~ done: `collection.py` (final table) and `error_analysis.py` (errors vs agreement, hard core, manual categories) — findings in §6
2. **Phase 6** ← next: collect 2026 headlines and label them under `data/gold/GUIDELINES.md` (blind to model predictions; pilot ~50 first, then freeze the rules; record `sure`/`unsure`; relabel 50 after a week for intra-annotator agreement). Then run every model on the gold set (domain shift + contamination check for FinBERT and the LLMs) and **choose the Phase 7 teacher** (Haiku vs Opus) by agreement with the gold labels. `news.py` (headline collection) still to be written
3. Optional: a convention-explicit prompt for Haiku, tuned on **val** only, to see if its positive-undercall bias is fixable cheaply

## 9. Companion notes

Detailed tutorial and concept explanations live in an Obsidian vault, outside this repo:

- `NLP_for_Finance_学习笔记.md` — 2,341-line main note; Part A concepts, Part B the Phase 0–10 build steps and code
- `Finsent - 补充笔记.md` — deeper analysis and interpretation of the Phase 2 results

Path: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Others/`

These notes are written in Chinese. This file is kept in English to match the repo's commits and README.
