# -*- coding: utf-8 -*-
"""
run_experiments.py —— 训练完成后的一键实验编排器

===== 为什么需要这个脚本 =====

本项目的实验链条是有**先后依赖**的，且每一步都耗时：

    train (慢) ──▶ evaluate (快) ──▶ exp_domain_shift (中)
                                          │
                                          ├──▶ exp_robustness (中)
                                          └──▶ exp_ablation    (中)
                                                    │
                                                    └──▶ step6_quantize (慢)

如果手工一条条敲，容易犯两类错：
  1. **用错权重**：训练中途的 best.pt 和训练完的 best.pt 不是同一个文件，
     中途跑出来的指标会与最终结论对不上，而结果文件看起来「正常」。
  2. **并行抢卡**：多个实验同时推理会互相拖慢甚至 OOM，
     延迟类指标（量化那步）会因显卡被占而彻底失真。

本脚本做三件事：
  - **守卫**：等训练进程退出 + results.csv 行数不再增长，才认为权重已定稿
  - **串行**：一个接一个跑，避免抢 GPU 导致延迟指标失真
  - **存档**：每步的 stdout 存进 logs/experiments/<step>.log（UTF-8）

用法：
  python run_experiments.py                        # 等训练结束，然后跑全部
  python run_experiments.py --now                  # 不等训练，立刻跑
  python run_experiments.py --steps=eval,domain    # 只跑指定步骤
  python run_experiments.py --weights=<pt>         # 指定权重
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

from common import (
    CODE_DIR,
    PROJ_ROOT,
    TRAIN_DIR,
    dump_json,
    ensure_dirs,
    log,
)

LOG_DIR = PROJ_ROOT / "logs" / "experiments"
PY = sys.executable

# 单次等待训练的轮询间隔（秒）。不需要太密 —— 一轮要 ~21s。
POLL_SEC = 30
# 连续多少次轮询 results.csv 不再增长，判定训练已结束
STABLE_ROUNDS = 4


# --------------------------------------------------------------------------
# 训练守卫
# --------------------------------------------------------------------------
def _csv_state(p: Path) -> tuple[int, float]:
    """返回 (行数, mtime)，用于判断训练是否还在推进。"""
    if not p.exists():
        return 0, 0.0
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        n = sum(1 for _ in f)
    return n, p.stat().st_mtime


def wait_for_training(run: str, timeout_h: float = 6.0) -> bool:
    """
    等到指定 run 的训练真正结束。

    判定标准有两个，缺一不可：
      A) results.csv 连续 STABLE_ROUNDS 次轮询行数与 mtime 都不变
      B) 对应的 best.pt 存在且 mtime 已稳定

    为什么不能只看「python 进程是否存在」：
    train.py 会**连续跑多个 run**（本机是 v8s640 然后 v11s640）。
    v8s640 跑完时 python 进程还在（正在跑 v11），
    此时取 v8s640 的权重是安全的，但若盯着进程就会一直等到 v11 也跑完，
    白白浪费几十分钟。所以这里盯**该 run 自己的产物**，而不是进程。
    """
    csv_p = TRAIN_DIR / run / "results.csv"
    best_p = TRAIN_DIR / run / "weights" / "best.pt"
    log(f"等待 [{run}] 训练收敛…（盯 {csv_p.name} 的变化）")

    t0 = time.time()
    last_n, last_m = -1, -1.0
    stable = 0
    while time.time() - t0 < timeout_h * 3600:
        n, m = _csv_state(csv_p)
        if n == last_n and abs(m - last_m) < 1e-6 and n > 0:
            stable += 1
            if stable >= STABLE_ROUNDS and best_p.exists():
                log(f"[{run}] 训练已收敛：{n - 1} 轮，"
                    f"权重 {best_p.stat().st_size / 1024 / 1024:.1f} MB")
                return True
        else:
            if stable:
                log(f"  仍在训练：{n - 1} 轮（等待中）")
            stable = 0
        last_n, last_m = n, m
        time.sleep(POLL_SEC)

    log(f"!! 等待 [{run}] 超时（{timeout_h}h），将用当前权重继续")
    return best_p.exists()


# --------------------------------------------------------------------------
# 步骤定义
# --------------------------------------------------------------------------
def build_steps(weights: Path, name: str) -> dict[str, tuple[str, list[str]]]:
    """
    每步 = (说明, [脚本, 参数...])。
    顺序即执行顺序 —— **不要并行**，量化那步的延迟指标怕抢卡。
    """
    w = str(weights)
    return {
        "eval": (
            "独立测试集评估（报告唯一可引用指标）",
            ["evaluate.py", f"--weights={w}", f"--name={name}", "--split=test"],
        ),
        "domain": (
            "实验1 距离域偏移：GSD–mAP 衰减曲线",
            ["exp_domain_shift.py", f"--weights={w}", f"--name={name}"],
        ),
        "robust": (
            "实验2 鲁棒性：4 种劣化 × 3 档",
            ["exp_robustness.py", f"--weights={w}", f"--name={name}"],
        ),
        "ablation": (
            "实验3 消融：标定误差 / 拒答权衡 / 尺度线性",
            ["exp_ablation.py", f"--weights={w}", f"--name={name}"],
        ),
        "quant": (
            "模型轻量化 INT8（赛道二 <10MB / <100ms）",
            ["step6_quantize.py", f"--weights={w}", "--calib-n=150"],
        ),
    }


DEFAULT_ORDER = ["eval", "domain", "robust", "ablation", "quant"]


def _argv(name: str, default=None):
    prefix = f"--{name}="
    for a in sys.argv[1:]:
        if a.startswith(prefix):
            return a[len(prefix):]
    return default


def _flag(name: str) -> bool:
    return f"--{name}" in sys.argv[1:]


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    run = _argv("run", "v8s640")
    if _flag("all-ready"):
        pass
    weights_s = _argv("weights")
    weights = Path(weights_s) if weights_s else (TRAIN_DIR / run / "weights" / "best.pt")
    name = _argv("name", f"{run}_final")
    steps_s = _argv("steps", ",".join(DEFAULT_ORDER))
    steps = [s.strip() for s in steps_s.split(",") if s.strip()]

    ensure_dirs(LOG_DIR)

    if not _flag("now"):
        if not wait_for_training(run):
            log("!! 训练未就绪，中止实验编排")
            return
    else:
        log("--now：跳过训练等待")

    if not weights.exists():
        log(f"!! 权重不存在: {weights}")
        return
    log(f"使用权重: {weights} "
        f"({weights.stat().st_size / 1024 / 1024:.2f} MB, "
        f"mtime={time.strftime('%H:%M:%S', time.localtime(weights.stat().st_mtime))})")

    table = build_steps(weights, name)
    unknown = [s for s in steps if s not in table]
    if unknown:
        log(f"!! 未知步骤 {unknown}，可选 {list(table)}")
        return

    records = []
    for key in steps:
        desc, cmd = table[key]
        script = CODE_DIR / cmd[0]
        if not script.exists():
            log(f"!! 脚本不存在，跳过: {script}")
            records.append({"step": key, "status": "missing", "script": cmd[0]})
            continue

        log("=" * 74)
        log(f"[{key}] {desc}")
        log("=" * 74)
        t0 = time.time()
        rec = {"step": key, "desc": desc, "script": cmd[0],
               "argv": cmd[1:], "log": str(LOG_DIR / f"{key}.log")}
        try:
            # 用 subprocess 而不是 in-process import：
            # 这些脚本各自会 import torch/ultralytics，进程内串跑容易
            # 出现 CUDA context 残留（尤其量化那步要切 CPU provider）。
            # 用干净子进程最稳。cwd 必须在 02_code，否则 import common 失败。
            p = subprocess.run(
                [PY, *cmd],
                cwd=str(CODE_DIR),
                capture_output=True,
            )
            out = (p.stdout or b"").decode("utf-8", errors="replace")
            err = (p.stderr or b"").decode("utf-8", errors="replace")
            (LOG_DIR / f"{key}.log").write_text(
                out + ("\n--- STDERR ---\n" + err if err.strip() else ""),
                encoding="utf-8")
            rec["status"] = "ok" if p.returncode == 0 else f"exit{p.returncode}"
            rec["seconds"] = round(time.time() - t0, 1)
            log(f"  -> {rec['status']}  用时 {rec['seconds']}s  日志 {key}.log")
        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = repr(e)
            rec["seconds"] = round(time.time() - t0, 1)
            log(f"  !! 失败: {e!r}")
        records.append(rec)

    dump_json({
        "run": run,
        "name": name,
        "weights": str(weights),
        "weights_mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                       time.localtime(weights.stat().st_mtime)),
        "order": steps,
        "results": records,
        "note": ("实验编排日志。各步骤的完整 stdout 在 logs/experiments/<step>.log。"
                 "所有实验必须串行执行，并行会抢 GPU 导致量化延迟指标失真。"),
    }, LOG_DIR / "run_all_summary.json")

    log("=" * 74)
    for r in records:
        log(f"{r['step']:10s} {r.get('status', '?'):10s} {r.get('seconds', '-')}s")
    log("=" * 74)
    log(f"汇总: {LOG_DIR / 'run_all_summary.json'}")


if __name__ == "__main__":
    main()
