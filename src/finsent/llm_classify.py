"""用 LLM 做情感分类。带本地缓存：同一个请求只付一次钱。

用法：
  python -m finsent.llm_classify --limit 5                 # smoke test：5 句，打印成本估算，不写 preds
  python -m finsent.llm_classify                           # Haiku 5.5 zero-shot, effort=low
  python -m finsent.llm_classify --few_shot                # few-shot
  python -m finsent.llm_classify --few_shot --effort medium
"""
import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd
from dotenv import load_dotenv

from finsent.data import LABEL2ID, OUTPUTS_DIR, PROCESSED_DIR, PROJECT_ROOT
from finsent.evaluate import summarize

# [改 4] 路径全部走 data.py 的常量：从任何目录运行，.env 和缓存都指向项目内
load_dotenv(PROJECT_ROOT / ".env")          # 读取 ANTHROPIC_API_KEY
CACHE = OUTPUTS_DIR / "llm_cache.jsonl"
PREDS_DIR = OUTPUTS_DIR / "preds"
_cache_lock = threading.Lock()

_client = None


def get_client() -> anthropic.Anthropic:
    """第一次真正调用 API 时才创建 client：import 本模块（比如只读缓存、跑测试）不需要 key。"""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


SYSTEM_PROMPT = """You are a financial analyst. Classify the sentiment of a financial news sentence \
from the perspective of an investor holding the company's stock.

Labels:
- positive: the news is likely to have a positive impact on the stock price
  (e.g. revenue/profit growth, losses narrowing, new contracts, upgrades).
- negative: the news is likely to have a negative impact on the stock price
  (e.g. falling profit, widening losses, layoffs due to weak demand, downgrades).
- neutral: factual or descriptive information with no clear price impact.

Rules:
- Judge the investor impact, not the emotional tone of the words.
  "Net loss narrowed to EUR 2m from EUR 5m" is positive.
- If the sentence is purely descriptive (company profile, listing info), choose neutral.

Respond with a JSON object: {"label": "positive" | "negative" | "neutral", "confidence": <number between 0 and 1>}"""

FEW_SHOT = [
    ("Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007.", "positive"),
    ("The company's net sales fell by 5% to EUR 22.6 mn.", "negative"),
    ("The company is headquartered in Helsinki and employs about 300 people.", "neutral"),
]

# [改 6] structured outputs：API 按这个 JSON schema 约束输出，label 只能是三个值之一。
# 不再需要正则抠 JSON，「模型多说了一句话 / 拼错 label」这类解析失败从源头消失。
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["negative", "neutral", "positive"]},
        "confidence": {"type": "number"},
    },
    "required": ["label", "confidence"],
    "additionalProperties": False,
}

# [改 3] prompt 指纹：system prompt、few-shot 示例、schema 任何一处改动，指纹就变，
# 缓存 key 跟着变 → 改 prompt 之后不会再读到旧 prompt 的结果。
PROMPT_VERSION = hashlib.sha256(
    json.dumps([SYSTEM_PROMPT, FEW_SHOT, OUTPUT_SCHEMA], sort_keys=True).encode()
).hexdigest()[:12]


def build_messages(sentence: str, few_shot: bool):
    messages = []
    if few_shot:   # 把示例写成多轮对话：user 给句子，assistant 给答案
        for s, lab in FEW_SHOT:
            messages.append({"role": "user", "content": f"Sentence: {s}"})
            messages.append({"role": "assistant", "content": json.dumps({"label": lab, "confidence": 0.9})})
    messages.append({"role": "user", "content": f"Sentence: {sentence}"})
    return messages


def cache_key(sentence: str, model: str, few_shot: bool, effort: str | None, replicate: int = 0) -> str:
    # replicate：同一配置的第几次独立采样。默认 0；翻转率测试用 1、2… 绕过缓存重新请求
    parts = [PROMPT_VERSION, model, str(few_shot), str(effort), str(replicate), sentence]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def load_cache() -> dict:
    if not CACHE.exists():
        return {}
    return {d["key"]: d for d in map(json.loads, CACHE.read_text().splitlines())}


def parse_response(resp) -> dict:
    """从 API 响应里取出 label。失败时 label=None，并说明原因 —— 绝不默默退回某个类别。"""
    # [改 5] 新模型默认开 thinking，content[0] 可能是 thinking block；按 type 找 text block
    if resp.stop_reason != "end_turn":        # max_tokens（被截断）/ refusal（拒答）等
        return {"label": None, "fail_reason": f"stop_reason={resp.stop_reason}"}
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        return {"label": None, "fail_reason": "no text block"}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"label": None, "fail_reason": "invalid JSON"}
    label = str(data.get("label", "")).lower().strip()
    if label not in LABEL2ID:
        return {"label": None, "fail_reason": f"unknown label {label!r}"}
    return {"label": LABEL2ID[label], "confidence": data.get("confidence"), "fail_reason": None}


def classify(sentence: str, model: str = "claude-haiku-5-5", few_shot: bool = True,
             effort: str | None = "low", replicate: int = 0, cache=None) -> dict:
    """分类一句话。effort=None 表示不传（Haiku 4.5 等老模型不支持 effort 参数）。"""
    key = cache_key(sentence, model, few_shot, effort, replicate)
    if cache is not None and key in cache:
        return cache[key]

    output_config = {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}}
    if effort is not None:
        output_config["effort"] = effort
    # [改 1] 不再传 temperature：SDK 2.x 用的 API 版本里，Messages 请求带 temperature 会 400
    resp = get_client().messages.create(
        model=model,
        max_tokens=2048,      # 新模型的 thinking 也计入 max_tokens；100 会被 thinking 吃光
        system=SYSTEM_PROMPT,
        messages=build_messages(sentence, few_shot),
        output_config=output_config,
    )
    parsed = parse_response(resp)
    # [改 2] 解析失败 → label=None、parse_ok=False，下游单独统计，不混进预测
    result = {"key": key, "sentence": sentence, "model": model, "few_shot": few_shot,
              "effort": effort, "replicate": replicate, "prompt_version": PROMPT_VERSION,
              "label": parsed["label"], "parse_ok": parsed["label"] is not None,
              "fail_reason": parsed["fail_reason"], "confidence": parsed.get("confidence"),
              "stop_reason": resp.stop_reason,
              "input_tokens": resp.usage.input_tokens,
              "output_tokens": resp.usage.output_tokens}
    # 只缓存成功的结果：失败的下次重跑会重新请求，而不是永远读到同一个失败
    if result["parse_ok"]:
        with _cache_lock:                     # 多线程并发时，文件追加和 dict 写入要互斥
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            with CACHE.open("a") as f:
                f.write(json.dumps(result) + "\n")
            if cache is not None:
                cache[key] = result
    return result


# ---------------------------------------------------------------- runner
# 价格只用于打印成本估算（$/1M tokens）；以 Console 账单为准
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00),
          "claude-opus-5-5": (4.00, 20.00), "claude-haiku-4-5": (1.00, 5.00)}


def run_name(model: str, few_shot: bool, effort: str | None) -> str:
    """和 grid 同一命名格式 <model>_<tag>_s<seed>，Phase 5 汇总可以直接读。LLM 无训练、无 seed → s0。"""
    tag = f"{'fewshot' if few_shot else 'zeroshot'}-{effort or 'default'}"
    return f"{model.removeprefix('claude-')}_{tag}_s0"


def flip_test(test, model, few_shot, effort, n, workers):
    """同一配置独立请求两次，看 label 翻转了多少 —— 量化采样随机性（temperature 已无法设置）。

    replicate=0 是正式 run 的结果（应该全在缓存里），replicate=1 是新请求。
    翻转率远小于 bootstrap CI（±0.03 macro-F1）时可以忽略；否则考虑多次采样 + 多数投票。
    """
    sample = test.sample(n=n, random_state=0).reset_index(drop=True)
    cache = load_cache()
    missing = sum(cache_key(s, model, few_shot, effort, 0) not in cache for s in sample["sentence"])
    if missing:
        print(f"[warn] {missing} sentences have no replicate=0 result yet — they will be requested too")

    def run(rep):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda s: classify(s, model, few_shot, effort, rep, cache), sample["sentence"]))

    r0, r1 = pd.DataFrame(run(0)), pd.DataFrame(run(1))
    ok = r0["parse_ok"] & r1["parse_ok"]
    flip = (r0["label"] != r1["label"]) & ok
    print(f"{model} {'few' if few_shot else 'zero'}-shot effort={effort}: "
          f"{flip.sum()}/{ok.sum()} flipped = {flip.sum() / ok.sum():.1%}")
    for name, m in [("clean", sample["all_agree"]), ("ambiguous", ~sample["all_agree"])]:
        print(f"  {name:9s}: {flip[m].sum()}/{(ok & m).sum()} flipped")
    if flip.any():
        show = sample[flip].assign(r0=r0["label"][flip], r1=r1["label"][flip],
                                   c0=r0["confidence"][flip], c1=r1["confidence"][flip])
        print(show[["label", "r0", "r1", "c0", "c1", "all_agree", "sentence"]].to_string(max_colwidth=60))
    print(f"mean confidence of flipped vs stable: {r0['confidence'][flip].mean():.2f} vs "
          f"{r0['confidence'][~flip & ok].mean():.2f}")
    tin, tout = r1["input_tokens"].sum(), r1["output_tokens"].sum()
    pin, pout = PRICES.get(model, (float("nan"), float("nan")))
    print(f"replicate=1 cost: ${tin / 1e6 * pin + tout / 1e6 * pout:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--few_shot", action="store_true", help="不加就是 zero-shot")
    p.add_argument("--effort", default="low", help="low/medium/high/...；none = 不传（老模型）")
    p.add_argument("--limit", type=int, default=None,
                   help="只跑前 N 句做 smoke test：打印结果和成本估算，不写 preds")
    p.add_argument("--replicate", type=int, default=0, help="翻转率测试用：>0 时绕过 replicate=0 的缓存")
    p.add_argument("--workers", type=int, default=8, help="并发请求数")
    p.add_argument("--flip_test", type=int, default=None, metavar="N",
                   help="翻转率测试：随机抽 N 句（seed 0），用 replicate=1 重新请求，和 replicate=0 对比")
    args = p.parse_args()
    effort = None if args.effort.lower() == "none" else args.effort

    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    if args.flip_test:
        flip_test(test, args.model, args.few_shot, effort, args.flip_test, args.workers)
        return
    if args.limit:
        test = test.head(args.limit)
    cache = load_cache()
    n_cached = sum(cache_key(s, args.model, args.few_shot, effort, args.replicate) in cache
                   for s in test["sentence"])
    print(f"{len(test)} sentences, {n_cached} already cached, {len(test) - n_cached} to request")

    def work(s):
        try:
            return classify(s, args.model, args.few_shot, effort, args.replicate, cache)
        except anthropic.APIError as e:      # SDK 已自动重试 429/5xx；仍失败的记为失败，不中断整批
            return {"label": None, "parse_ok": False, "fail_reason": f"{type(e).__name__}: {e}",
                    "confidence": None, "input_tokens": 0, "output_tokens": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(work, test["sentence"]))   # map 保持输入顺序

    res = pd.DataFrame(results)
    fails = res[~res["parse_ok"]]
    print(f"parse failures: {len(fails)}/{len(res)}")
    if len(fails):
        print(fails["fail_reason"].value_counts().to_string())
    # 成本只算本次新请求的（缓存命中的结果里也存了 token 数，这里一并计入 = 这批数据的总成本）
    tin, tout = res["input_tokens"].sum(), res["output_tokens"].sum()
    pin, pout = PRICES.get(args.model, (float("nan"), float("nan")))
    cost = tin / 1e6 * pin + tout / 1e6 * pout
    print(f"tokens: input {tin:,} / output {tout:,} → est. ${cost:.4f} "
          f"(≈ ${cost / len(res) * 726:.3f} per 726-sentence run)")

    name = run_name(args.model, args.few_shot, effort)
    out = test[["sentence", "label", "all_agree"]].copy().reset_index(drop=True)
    # 失败记为 -1：macro_f1 只统计 labels=[0,1,2]，-1 对真实类别算漏判 → 失败一律按错误计分（保守）
    out["pred"] = res["label"].fillna(-1).astype(int)
    out[["p_neg", "p_neu", "p_pos"]] = float("nan")   # LLM 不给类别概率
    out["confidence"] = res["confidence"]
    out["model"], out["seed"] = name.rsplit("_s", 1)[0], 0

    if args.limit:
        print(out[["label", "pred", "confidence", "sentence"]].to_string(max_colwidth=70))
        print("(smoke test: preds not written)")
        return
    suffix = f"_r{args.replicate}" if args.replicate else ""
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREDS_DIR / f"{name}{suffix}.parquet", index=False)
    print(f"Saved predictions to {PREDS_DIR / f'{name}{suffix}.parquet'}")
    for tag, sub in [("allagree", out[out["all_agree"]]), ("50agree", out)]:
        summarize(sub["label"], sub["pred"], name=f"{name}{suffix} | test={tag}")


if __name__ == "__main__":
    main()
