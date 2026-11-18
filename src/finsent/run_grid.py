"""Phase 3 grid：3 models × 3 seeds × 2 train sets（可选加 equal-size 对照）。

用法：
  python -m finsent.run_grid                 # 18 runs
  python -m finsent.run_grid --with_control  # 再加 3 models × 3 seeds 的 50agree-n1569
  python -m finsent.run_grid --dry_run       # 只列出会跑什么

设计：
- 每个 run 起一个**独立子进程**：跑完就释放 GPU/MPS 显存，一个 run 崩了不影响后面的
- **可断点续跑**：outputs/preds/<run>.parquet 已存在就跳过（Colab 断线后重跑同一条命令即可）
- 每个 run 的完整输出写到 outputs/logs/<run>.log
- run 成功后把 trainer_state.json（每个 epoch 的 val curve）复制到 outputs/logs/<run>.trainer_state.json，
  然后删掉 checkpoint（每个 1–2 GB，27 个 run 会塞满磁盘）。这样 preds/ + logs/ 就是全部需要保留的结果，
  体积小，可以整个放进 Google Drive
"""
import argparse
import shutil
import subprocess
import sys
import time

from finsent.data import OUTPUTS_DIR

MODELS = ["answerdotai/ModernBERT-base", "microsoft/deberta-v3-base", "ProsusAI/finbert"]
SEEDS = [1, 2, 3]
ALLAGREE_TRAIN_SIZE = 1569   # equal-size 对照：50agree 抽到和 allagree train 一样大


def plan(with_control: bool):
    """返回 [(run_name, extra_cli_args)]，run_name 必须和 train_encoder.py 的命名一致。"""
    configs = [("50agree", []), ("allagree", ["--only_allagree"])]
    if with_control:
        configs.append((f"50agree-n{ALLAGREE_TRAIN_SIZE}", ["--subsample", str(ALLAGREE_TRAIN_SIZE)]))
    runs = []
    for model in MODELS:
        for tag, extra in configs:
            for seed in SEEDS:
                name = f"{model.split('/')[-1]}_{tag}_s{seed}"
                runs.append((name, ["--model", model, "--seed", str(seed), *extra]))
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--with_control", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    preds_dir, log_dir = OUTPUTS_DIR / "preds", OUTPUTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    runs = plan(args.with_control)
    todo = [(n, a) for n, a in runs if not (preds_dir / f"{n}.parquet").exists()]
    print(f"{len(runs)} runs planned, {len(runs) - len(todo)} already done, {len(todo)} to run")

    failed = []
    for i, (name, cli) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {name} ...", end=" ", flush=True)
        if args.dry_run:
            print("(dry run)")
            continue
        t0 = time.time()
        with open(log_dir / f"{name}.log", "w") as log:
            rc = subprocess.run([sys.executable, "-m", "finsent.train_encoder", *cli],
                                stdout=log, stderr=subprocess.STDOUT).returncode
        status = "ok" if rc == 0 else f"FAILED (exit {rc}, see logs/{name}.log)"
        print(f"{status}  {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            failed.append(name)
            continue
        run_dir = OUTPUTS_DIR / "runs" / name
        state = run_dir / "trainer_state.json"
        if state.exists():
            shutil.copy(state, log_dir / f"{name}.trainer_state.json")
        for ckpt in run_dir.glob("checkpoint-*"):
            shutil.rmtree(ckpt)

    if failed:
        print(f"\n{len(failed)} failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
