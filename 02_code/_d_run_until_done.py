# -*- coding: utf-8 -*-
r"""
_d_run_until_done.py —— D 路【分块续训驱动器】（⚠️ 已弃用，见下）

================================================================
⚠️ 已弃用（2026-09-25）—— 请改用 `logs/_d_watchdog.py`
================================================================
**为什么弃用**：本脚本是「守护 → 驱动器 → 训练」三层链的中间层。
本机约 28 分钟杀「长任务」（纪律 #7），而这条链上**每一层都是长任务**
⇒ 训练被杀的同一时刻，驱动器也在自己的 28 分钟窗口里被杀
⇒ **第 2 块永远起不来**，而且**不报错**（看起来"在跑"）。
（具体表现：本脚本 L108 `p.communicate()` 会一直等到训练子进程结束，
 但它自己和训练子进程共享同一个 28 分钟预算。）

**替代方案**：`logs/_d_watchdog.py`
  它**不 babysit**，只做轻量轮询（每轮只 stat 一个文件）；
  判死就 spawn 训练**但不等待** ⇒ 哨兵被杀也不影响训练，可随时重启。
  已用 `logs/_selfcheck_watchdog.py` 做真/假两态自检（7 项全过）。

本文件保留仅为存档与追溯，**不要在新任务里使用**。
如需手动跑一块，请直接调 `_d_train_downsample.py`。

================================================================
原说明（存档）
================================================================

===== 为什么需要它 =====
本机长任务约 28 分钟被杀（exit −1、无 traceback，纪律 #7）。
B1 跑到 60 epoch 需 ~78 分钟 ⇒ **必须分块**。
但「起训练→等 28 分钟→被杀→续训」若用手动方式，人一走就断。

本脚本把整个「跑到目标轮数」的循环**放在一个进程内**：
  for chunk in range(MAX_CHUNK):
      rc = run(训练 或 --resume)          # 子进程；会被外部杀
      读 results.csv 轮数
      if 轮数 >= target: break
      # 否则循环，下一轮用 --resume
它自己不被杀（子进程被杀后它继续），从而实现**无人值守续训**。

⚠️ 注意：本脚本自身也可能在被杀窗口里 —— 故用 `run_in_background=true` 起它，
   并在每块结束后把进度**写文件**（PowerShell stdout 不回传，纪律 #3）。

用法：
  python 02_code/_d_run_until_done.py --cfg=B1_rust3000 --target=60
  python 02_code/_d_run_until_done.py --cfg=B2_rust1000 --target=60
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINER = ROOT / "02_code" / "_d_train_downsample.py"
LOGF = ROOT / "logs" / "_d_driver.log"


def _log(s: str):
    line = f"[{time.strftime('%H:%M:%S')}] {s}"
    print(line, flush=True)
    with LOGF.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _epochs_done(run_dir: Path) -> int:
    csvp = run_dir / "results.csv"
    if not csvp.exists():
        return 0
    try:
        rows = list(csv.DictReader(csvp.open("r", encoding="utf-8", newline="")))
        return len(rows)
    except Exception:
        return 0


def _best_mAP50(run_dir: Path):
    csvp = run_dir / "results.csv"
    if not csvp.exists():
        return None
    try:
        rows = list(csv.DictReader(csvp.open("r", encoding="utf-8", newline="")))
        vals = [float(r.get("metrics/mAP50(B)", "nan")) for r in rows]
        vals = [v for v in vals if v == v]
        return max(vals) if vals else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, choices=["B1_rust3000", "B2_rust1000"])
    ap.add_argument("--target", type=int, default=60)
    ap.add_argument("--max-chunks", type=int, default=12)
    ap.add_argument("--chunk-timeout", type=int, default=1800,
                    help="单块最长秒数（默认 30 分钟，略大于 28 分钟杀窗口）")
    args = ap.parse_args()

    run_dir = ROOT / "04_results" / "train" / f"d_{args.cfg}"
    py = sys.executable  # 用的是当前解释器（即 D:\下载\python.exe）

    _log("=" * 70)
    _log(f"驱动器启动 cfg={args.cfg} target={args.target} run_dir={run_dir}")
    _log(f"解释器 = {py}")
    _log("=" * 70)

    for k in range(1, args.max_chunks + 1):
        done = _epochs_done(run_dir)
        if done >= args.target:
            _log(f"✅ 已达目标 {done}/{args.target}，退出")
            break

        first = not (run_dir / "weights" / "last.pt").exists()
        cmd = [py, str(TRAINER), f"--cfg={args.cfg}",
               f"--epochs={args.target}"]
        if not first:
            cmd.append("--resume")
        _log(f"--- 第 {k} 块：{'首次' if first else '续训'} 起点 epoch={done} ---")
        _log("CMD: " + " ".join(cmd))

        t0 = time.time()
        try:
            p = subprocess.Popen(cmd, cwd=str(ROOT),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
            # 不设 timeout：让外部 28 分钟杀子进程；这里只等它结束
            _, _ = p.communicate()
            rc = p.returncode
        except Exception as e:
            _log(f"块 {k} 异常: {e}")
            rc = -999
        dt = time.time() - t0

        done2 = _epochs_done(run_dir)
        best = _best_mAP50(run_dir)
        _log(f"块 {k} 结束 rc={rc} 用时 {dt/60:.1f} min，"
             f"epoch {done}→{done2}，当前 best mAP50={best}")
        time.sleep(3)

    done = _epochs_done(run_dir)
    best = _best_mAP50(run_dir)
    _log(f"驱动器收工：最终 epoch={done}/{args.target}，best mAP50={best}")
    return 0 if done >= args.target else 1


if __name__ == "__main__":
    sys.exit(main())
