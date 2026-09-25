# -*- coding: utf-8 -*-
"""
_d_analyze.py —— D 路【实验组 B】结果分析：三组逐类 AP 对比

===== 它要回答的问题 =====

长尾降采样（rust 9433 → 3078 / 977）到底有没有让「稀有但关键」的类变好？

对照三组（全部同一 V3 测试集评估，口径不变）：
  · 对照  v11s640              （rust 9433，不平衡比 45.6:1）
  · B1    d_B1_rust3000        （rust 3078，不平衡比 14.9:1）
  · B2    d_B2_rust1000        （rust  977，不平衡比  4.7:1）

===== 判读纪律（沿用项目现行口径）=====

1. **不看总体 mAP50 定胜负** —— 本项目已确认：模型间 mAP50 差 > 0.03 才可称真实。
2. **看逐类 AP50**：降采样若有效，应表现在**稀有类的 AP 上升**（crack / spalling）。
3. **看宏平均**（7 类等权）：它才是"长尾是否被缓解"的直接指标 ——
   因为微平均（= 通常的 mAP50）会被 rust 的绝对数量主导。
4. **r在噪声内要说"在噪声内"**：按二项分布估各类 AP 的 CI 半宽（复用项目既有做法）。

===== 输入 =====

  04_results/eval/<name>_test_metrics.json   （由 evaluate.py 生成）
  04_results/eval/<name>_per_class.csv       （逐类 AP）

若缺失，本脚本会提示先跑评估，而不是自动猜数字。

用法：python _d_analyze.py
输出：04_results/eval/d_downsample_comparison.json
       logs/_D_DOWNSAMPLE_REPORT.md
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "04_results" / "eval"

GROUPS = [
    ("对照 v11s640", "v11s640"),
    ("B1 rust3000", "d_B1_rust3000"),
    ("B2 rust1000", "d_B2_rust1000"),
]
CLASSES = ["crack", "spalling", "efflorescence", "exposed_rebar",
           "rust", "delamination", "moss"]
# 各档训练时的实例数（用于宏平均口径的说明）
RUST_COUNT = {"v11s640": 9433, "d_B1_rust3000": 3078, "d_B2_rust1000": 977}
TOTAL_COUNT = {"v11s640": 12631, "d_B1_rust3000": 6276, "d_B2_rust1000": 4175}


def load_metrics(stem: str):
    """读 <stem>_test_metrics.json；键在 overall 下，名如 'metrics/mAP50(B)'。"""
    p = EVAL / f"{stem}_test_metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    ov = d.get("overall", d)
    def g(*keys):
        for k in keys:
            if k in ov:
                return float(ov[k])
        return None
    return dict(
        mAP50=g("metrics/mAP50(B)", "mAP50"),
        mAP5095=g("metrics/mAP50-95(B)", "mAP50-95"),
        precision=g("metrics/precision(B)", "precision"),
        recall=g("metrics/recall(B)", "recall"),
    )


def load_per_class(stem: str):
    p = EVAL / f"{stem}_per_class.csv"
    if not p.exists():
        return None
    rows = {}
    with p.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = None
            for cand in ("class", "cls", "name", "类别"):
                if cand in r:
                    key = r[cand]
                    break
            if key is None:
                continue
            ap = None
            for cand in ("AP50", "ap50", "mAP50", "ap"):
                if cand in r:
                    try:
                        ap = float(r[cand])
                    except Exception:
                        ap = None
                    break
            if ap is not None:
                rows[key] = ap
    return rows or None


def ci_half(n_pos: int) -> float:
    """二项分布 AP 的粗 CI 半宽（复用项目既有做法：1.96*sqrt(p(1-p)/n)）。

    ⚠️ p 取 **0.75**，与 MEMORY §3.3 那张「不可比区间」表**保持一致**
    （曾用 0.7 导致 spalling 半宽算成 0.137，与已文档化的 0.129 不符 ⇒ 已改回 0.75）。
    只用于「同量级」判断，不是严格统计量。
    """
    if n_pos <= 0:
        return float("nan")
    p = 0.75
    return 1.96 * math.sqrt(p * (1 - p) / n_pos)


def main() -> int:
    lines: list[str] = []
    def say(s=""): lines.append(s)

    data = {}
    for label, stem in GROUPS:
        m = load_metrics(stem)
        pc = load_per_class(stem)
        data[stem] = dict(label=label, metrics=m, per_class=pc)

    missing = [s for s, v in data.items() if v["metrics"] is None]
    if missing:
        say("# D 路降采样诊断 —— 结果分析")
        say()
        say("⚠️ 以下配置**尚无评估结果**，请先跑 evaluate.py：")
        for s in missing:
            say(f"  · {s}")
        say()
        say("命令示例：")
        for s in missing:
            say(f"  python evaluate.py --weights=04_results/train/{s}/weights/best.pt --name={s}")
        txt = "\n".join(lines)
        (ROOT / "logs" / "_D_DOWNSAMPLE_REPORT.md").write_text(txt, encoding="utf-8")
        print(txt)
        return 2

    say("# D 路 · 长尾 rust 降采样诊断")
    say()
    say("> 干预层级对照：**实验组 A = 损失层加权**（已做，7 类全在噪声内）；"
        "**实验组 B = 数据层降采样**（本文）。")
    say()

    # ---- 总体指标 ----
    say("## 一、总体指标（⚠️ 不以 mAP50 定胜负）")
    say()
    say("| 配置 | 训练 rust 实例 | 训练总实例 | 不平衡比 | mAP50 | mAP50-95 | P | R |")
    say("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, stem in GROUPS:
        v = data[stem]; m = v["metrics"]
        rc, tc = RUST_COUNT[stem], TOTAL_COUNT[stem]
        ratio = "—" if stem == "v11s640" else f"{rc/207:.1f}:1"
        if stem == "v11s640":
            ratio = "45.6:1"
        say(f"| {v['label']} | {rc} | {tc} | {ratio} | "
            f"{m['mAP50']:.4f} | {m['mAP5095']:.5f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} |")
    say()

    # ---- 逐类 AP50 ----
    say("## 二、★ 逐类 AP50（降采样是否救回了稀有类）")
    say()
    have_pc = all(data[s]["per_class"] for _, s in GROUPS)
    if not have_pc:
        say("（逐类 CSV 不全，跳过本节）")
    else:
        # 各测试集类的实例数（V3 test：见 MEMORY §3.2），用于估 CI 半宽
        TSET_N = {"crack": 58, "spalling": 43, "efflorescence": 116,
                  "exposed_rebar": 147, "rust": 1149, "delamination": 68, "moss": 65}
        say("| 类别 | 对照 | B1 | B2 | B1−对照 | B2−对照 | ±CI | 判读 |")
        say("|---|---:|---:|---:|---:|---:|---:|---|")
        rows_csv = []
        for c in CLASSES:
            a = data["v11s640"]["per_class"].get(c)
            b1 = data["d_B1_rust3000"]["per_class"].get(c)
            b2 = data["d_B2_rust1000"]["per_class"].get(c)
            if a is None or b1 is None or b2 is None:
                say(f"| {c} | — | — | — | — | — | — | 数据缺失 |")
                continue
            d1, d2 = b1 - a, b2 - a
            half = ci_half(TSET_N.get(c, 43))
            def verdict(d):
                # 双判据：既看项目 0.03 硬阈值，也看该类自身 CI 半宽
                if abs(d) <= half:
                    return "噪声内(≤CI)"
                if abs(d) > 0.03:
                    return "**超 0.03**"
                return "超CI但<0.03"
            say(f"| {c} | {a:.4f} | {b1:.4f} | {b2:.4f} | {d1:+.4f} | {d2:+.4f} | "
                f"±{half:.3f} | B1:{verdict(d1)} / B2:{verdict(d2)} |")
            rows_csv.append(dict(cls=c, base=a, b1=b1, b2=b2, d1=d1, d2=d2,
                                 ci_half=round(half, 4)))
        # 宏平均（7 类等权）—— 长尾是否缓解的直接指标
        def macro(stem):
            pc = data[stem]["per_class"]
            vals = [pc.get(c) for c in CLASSES]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        say()
        say("### ★ 宏平均（7 类等权）—— 长尾缓解的直接指标")
        say()
        say(f"- 对照: **{macro('v11s640'):.4f}**")
        say(f"- B1  : **{macro('d_B1_rust3000'):.4f}**"
            f"（{macro('d_B1_rust3000')-macro('v11s640'):+.4f}）")
        say(f"- B2  : **{macro('d_B2_rust1000'):.4f}**"
            f"（{macro('d_B2_rust1000')-macro('v11s640'):+.4f}）")
        say()
        say("> 为什么看宏平均：常规 mAP50 是**微平均**，被 rust 的绝对数量主导；"
            "降采样的目的正是让**稀有类**被看见，宏平均才对得上这个目的。")
    say()

    # ---- 结果写入 ----
    out = dict(
        groups=[dict(label=data[s]["label"], stem=s,
                     train_rust=RUST_COUNT[s], train_total=TOTAL_COUNT[s],
                     metrics=data[s]["metrics"]) for _, s in GROUPS],
        per_class_note="逐类 AP50 见上表；宏平均 = 7 类等权均值",
        caveat=("★ 固有混杂：B1/B2 的训练样本**总量也变小了**（1995→1425/1248），"
                "故「变差」不能单独归因于 rust 减少，也可能是总量减少。"
                "请结合 B1/B2 的**强度趋势**判读，并在报告中如实声明。"),
    )
    (EVAL / "d_downsample_comparison.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # 逐类 CSV 另存
    if have_pc:
        with (EVAL / "d_downsample_per_class.csv").open("w", newline="",
                                                        encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["cls", "base", "b1", "b2", "d1", "d2",
                                              "ci_half"])
            w.writeheader()
            for r in rows_csv:
                w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in r.items()})

    say("## 三、★ 限制条件（必须写进报告）")
    say()
    say("1. **固有混杂**：B1/B2 训练样本总量同时减少（1995 → 1425 / 1248），"
        "「AP 变化」不能单独归因于 rust 减少。")
    say("2. **val/test 未变**：两组的 val 与 test 均用 V3 正式集 ⇒ 口径与基线一致。")
    say("3. **唯一变量**：`cls_pw=0.0`（未启用类权重）、seed=42、其余超参同 COMMON_ARGS。")
    say()

    txt = "\n".join(lines)
    (ROOT / "logs" / "_D_DOWNSAMPLE_REPORT.md").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
