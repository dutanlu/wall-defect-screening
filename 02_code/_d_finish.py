# -*- coding: utf-8 -*-
r"""
_d_finish.py —— D 路收尾编排：B1/B2 训练完成后，一键「评估 → 分析 → 报告」

===== 为什么要有它 =====
D 路实验的收尾是**固定四步**，手工逐条执行容易漏（本项目已多次踩「漏一步」）：
  1. 用 `evaluate.py` 在**同一 V3 测试集**上评估 B1、B2 的 best.pt；
  2. 确认评估产物齐全（`*_test_metrics.json` + `*_per_class.csv`）；
  3. 跑 `_d_analyze.py` 生成三组对照报告；
  4. 把结论摘要落到 `logs/_D_FINISH_SUMMARY.md`。

★ 严守的两条纪律
  · **不做任何数字的"补写"**：评估失败就报失败，绝不猜；
  · **不覆盖已有评估**：除非显式 `--force`，否则已存在的 metrics 直接复用
    （避免重复评估白白占用 GPU）。

用法：
  python 02_code/_d_finish.py                 # 自动判断该做什么
  python 02_code/_d_finish.py --force         # 强制重跑评估
  python 02_code/_d_finish.py --skip-eval     # 只做分析（评估已就绪时）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "04_results" / "eval"
TRAIN = ROOT / "04_results" / "train"
LOGF = ROOT / "logs" / "_d_finish.log"

# 本脚本负责的配置（对照 v11s640 已评估过，不重复）
CFGS = ["d_B1_rust3000", "d_B2_rust1000"]


def _log(s: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {s}"
    print(line, flush=True)
    try:
        with LOGF.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _eval_ready(cfg: str) -> bool:
    """评估产物是否齐全。"""
    j = EVAL / f"{cfg}_test_metrics.json"
    c = EVAL / f"{cfg}_per_class.csv"
    return j.exists() and c.exists()


def _run(cmd: list[str], timeout: int | None = None) -> int:
    _log("CMD: " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True)
    out = (p.stdout or b"").decode("utf-8", "replace")
    err = (p.stderr or b"").decode("utf-8", "replace")
    # 落盘完整输出，便于事后核查
    tag = Path(cmd[1]).stem if len(cmd) > 1 else "cmd"
    (ROOT / "logs" / f"_d_finish_{tag}.out").write_text(
        out + "\n--- STDERR ---\n" + err, encoding="utf-8")
    tail = "\n".join(out.splitlines()[-12:])
    _log("rc=%d\n%s" % (p.returncode, tail))
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="强制重跑评估")
    ap.add_argument("--skip-eval", action="store_true", help="只做分析")
    args = ap.parse_args()

    py = sys.executable
    _log("=" * 70)
    _log(f"D 路收尾 force={args.force} skip_eval={args.skip_eval}")
    _log("=" * 70)

    # ---------- 0) 前置检查：权重是否都在 ----------
    missing_w = []
    for cfg in CFGS:
        w = TRAIN / cfg / "weights" / "best.pt"
        if not w.exists():
            missing_w.append(str(w))
    if missing_w:
        _log("⚠️ 以下权重还不存在，训练可能尚未完成：")
        for w in missing_w:
            _log("   " + w)
        _log("⇒ 先等训练跑完（哨兵会自动接力）。本次不做任何事。")
        return 2

    # ---------- 1) 评估 ----------
    if not args.skip_eval:
        for cfg in CFGS:
            if _eval_ready(cfg) and not args.force:
                _log(f"[{cfg}] 已有评估产物，跳过（要重跑加 --force）")
                continue
            w = TRAIN / cfg / "weights" / "best.pt"
            _log(f"[{cfg}] 开始评估 {w.name}")
            rc = _run([py, str(ROOT / "02_code" / "evaluate.py"),
                       f"--weights={w}", f"--name={cfg}"])
            if rc != 0:
                _log(f"[{cfg}] ⚠️ 评估 rc={rc}；继续下一项（不中断）")
    else:
        _log("按 --skip-eval 跳过评估")

    # ---------- 2) 确认产物 ----------
    ok = []
    bad = []
    for cfg in CFGS:
        if _eval_ready(cfg):
            ok.append(cfg)
        else:
            bad.append(cfg)
    _log(f"评估产物齐全: {ok}")
    if bad:
        _log(f"评估产物缺失: {bad}")

    # ---------- 3) 分析 ----------
    _log("运行 _d_analyze.py …")
    rc = _run([py, str(ROOT / "02_code" / "_d_analyze.py")])

    # ---------- 4) 摘要 ----------
    lines = ["# D 路收尾摘要", "",
             f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"- 评估产物齐全：{ok}", f"- 评估产物缺失：{bad}",
             f"- `_d_analyze.py` 退出码：{rc}", ""]
    cmpj = EVAL / "d_downsample_comparison.json"
    if cmpj.exists():
        try:
            d = json.loads(cmpj.read_text(encoding="utf-8"))
            lines.append("## 三组总体指标（来自 d_downsample_comparison.json）")
            lines.append("")
            lines.append("| 配置 | 训练 rust | 训练总实例 | mAP50 | mAP50-95 |")
            lines.append("|---|---:|---:|---:|---:|")
            for g in d.get("groups", []):
                m = g.get("metrics") or {}
                lines.append("| %s | %s | %s | %.4f | %.5f |" % (
                    g.get("label"), g.get("train_rust"), g.get("train_total"),
                    m.get("mAP50", float("nan")), m.get("mAP5095", float("nan"))))
            lines.append("")
            lines.append("> caveat: " + str(d.get("caveat", "")))
        except Exception as e:
            lines.append(f"（读 comparison.json 失败：{e!r}）")
    else:
        lines.append("（尚无 `d_downsample_comparison.json`）")
    lines.append("")
    lines.append("详细对照见 `logs/_D_DOWNSAMPLE_REPORT.md`。")
    (ROOT / "logs" / "_D_FINISH_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    _log("已写 logs/_D_FINISH_SUMMARY.md")

    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
