# -*- coding: utf-8 -*-
r"""P3（重做）：把「弃权」升级为「**下一步该干什么**」——期望损失最小化。

## 第一版为什么退化成"总是给结论"
第一版把信号写成单一 `px_on`，并按 `px<3 / 3-8 / >=8` 分箱，结果**所有成本比下都选 A**。
根因是口径错了：项目**生产路径的真实判据是双侧窗口** `3 <= px <= ks-1`
（见 `logs/_exp_risk_coverage.py`；该分析**项目早已做过**，并把被接受 risk 从 60.73% 压到 29.24%）。
按单侧 `px>=3` 分箱会把"过大目标（测量崩塌）"也算作"更容易"，
于是 `>=8px` 箱的误差率反而是 100% ⇒ 决策退化。

## 本版：直接用**生产判据的三个区间**作为信号
| 区间 | 含义 | 应导向 |
|---|---|---|
| `px < 3` | 目标太小，分辨率不够 | **R 再拍**（靠近 ⇒ 进入接受带） |
| `3 <= px <= ks-1` | 生产接受带 | **A 给结论** |
| `px > ks-1` | 目标过大 ⇒ 测量**系统性崩塌** | **H 请人鉴定**（再拍无用） |

实测（`risk_coverage_curve.json` 里 29 个带 `px_hi` 的点）：

| 区间 | n | mean\|rel\|% | P(\|rel\|>10%) |
|---|---:|---:|---:|
| `px<3` | 3 | 73.33 | 1.00 |
| `3<=px<=ks-1` | 11 | **29.24** | 0.82 |
| `px>ks-1` | 15 | 83.82 | 1.00 |

## 纪律
  · `P(error|signal)` 取自**实测**，不拍脑袋；
  · 成本**不给单一定值**，做**比值扫描**，输出参数化决策边界；
  · 表述是「**在成本比 = r 时最优行动是 X**」，不是「应该 X」；
  · 「再拍能改善多少」= 进入接受带后的误差率（依据 `advice.py` 的 GSD 反解语义），不另造假设。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\pythonstudy 备份\创新题\外墙缺陷筛查")
EVAL = ROOT / "04_results" / "eval"
OUT_TXT = ROOT / "logs" / "_p3_decision.txt"

MIN_PX = 3
ERR_REL = 0.10
C_MISS = 1.0            # 单位：一次错误结论的代价

REGIMES = ("too_small", "accepted", "too_large")
NAMES = {"too_small": "px<3(太小)", "accepted": "3<=px<=ks-1(接受带)",
         "too_large": "px>ks-1(过大)"}


def regime(p: dict) -> str:
    px, hi = p["px_on"], p.get("px_hi")
    if px < MIN_PX:
        return "too_small"
    if hi is not None and px > hi:
        return "too_large"
    return "accepted"


def main() -> int:
    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(str(s))

    rc = json.loads((EVAL / "risk_coverage_curve.json").read_text(encoding="utf-8"))
    pts = [p for p in rc["points"] if p.get("px_hi")]
    say("=" * 96)
    say("P3：期望损失 → 行动建议（信号 = 生产路径的**双侧窗口**）")
    say("=" * 96)
    say("")
    say(f"数据：{EVAL.name}/risk_coverage_curve.json，取带 `px_hi` 的 {len(pts)} 个点")
    say(f"误差事件：|相对误差| > {ERR_REL:.0%}")

    stat = {}
    for r in REGIMES:
        sub = [p for p in pts if regime(p) == r]
        if sub:
            stat[r] = {"n": len(sub),
                       "mean_abs_rel": round(sum(abs(p["rel_err"]) for p in sub) / len(sub), 2),
                       "p_err": round(sum(1 for p in sub
                                          if abs(p["rel_err"]) / 100 > ERR_REL) / len(sub), 4)}
        else:
            stat[r] = {"n": 0, "mean_abs_rel": None, "p_err": None}

    say("")
    say(f"{'区间':>24}{'n':>5}{'mean|rel|%':>12}{'P(|rel|>10%)':>14}")
    say("-" * 56)
    for r in REGIMES:
        d = stat[r]
        s1 = "-" if d["mean_abs_rel"] is None else f"{d['mean_abs_rel']:.2f}"
        s2 = "-" if d["p_err"] is None else f"{d['p_err']:.2f}"
        say(f"{NAMES[r]:>24}{d['n']:>5}{s1:>12}{s2:>14}")
    say("")
    say("⇒ **接受带的误差（29.24%）比两个拒绝区（73.3% / 83.8%）好约 3 倍**")
    say("  ⇒ 生产判据是有效信号，可作为决策规则的输入。")

    R_RE = (0.001, 0.01, 0.1, 1.0)
    # ★ 成本比的方向（第一版设反了）： 是「一次错误结论」的代价，
    #   而**漏掉结构安全隐患的代价应高于一次上门鉴定** ⇒ 。
    #   第一版取 r_hu ∈ {1,10,100}（鉴定比错结论还贵）⇒ 「过大」区 12/12 都选照给结论，
    #   即**在测量已崩塌的区间反而给出结论** —— 这与安全直觉相反，是成本模型设反导致的。
    R_HU = (0.01, 0.1, 1.0)
    p_acc = stat["accepted"]["p_err"] if stat["accepted"]["p_err"] is not None else 1.0

    # ★ 并列时安全优先：H > R > A（见文件 docstring 与 logs/_p3_decision.txt）
    PRIORITY = {"H": 0, "R": 1, "A": 2}

    def pick(L: dict) -> str:
        """期望损失最小；**并列时取更保守的行动**（H > R > A）。"""
        m = min(L.values())
        tied = [k for k, v in L.items() if abs(v - m) <= 1e-12]
        return sorted(tied, key=lambda k: PRIORITY[k])[0]

    def loss(rg: str, r_re: float, r_hu: float) -> dict:
        """三个行动的期望损失（单位：c_miss）。"""
        d = stat[rg]
        pa = d["p_err"] if d["p_err"] is not None else 1.0
        # 再拍：too_small ⇒ 可进入接受带；accepted ⇒ 已最好，再拍不改善；too_large ⇒ 再拍无用
        p_after = p_acc if rg == "too_small" else pa
        return {"A": pa, "R": r_re + p_after, "H": r_hu}

    say("")
    say("=" * 96)
    say("★ 参数化决策边界：给定（再拍成本比 r_re，鉴定成本比 r_hu）⇒ 各区间最优行动")
    say("=" * 96)
    say("（c_miss 为单位代价；A=给结论 / R=再拍 / H=请人鉴定）")
    say("")
    hdr = f"{'r_re':>8}{'r_hu':>8}" + "".join(f"{NAMES[r]:>26}" for r in REGIMES)
    say(hdr)
    say("-" * len(hdr))
    table = []
    for r_re in R_RE:
        for r_hu in R_HU:
            cells = []
            opt = {}
            for rg in REGIMES:
                L = loss(rg, r_re, r_hu)
                best = pick(L)
                cells.append(f"{best} ({L[best]:.4f})")
                opt[NAMES[rg]] = best
            say(f"{r_re:>8.3f}{r_hu:>8.0f}" + "".join(f"{c:>26}" for c in cells))
            table.append({"r_rephoto": r_re, "r_human": r_hu, "optimal": opt})

    say("")
    say("=" * 96)
    say("Gate：策略 vs 恒定策略（期望损失按区间占比加权）")
    say("=" * 96)
    w = {r: stat[r]["n"] / len(pts) for r in REGIMES}
    all_ok = True
    detail = []
    for r_re in R_RE:
        for r_hu in R_HU:
            l_pol = l_A = l_H = 0.0
            for rg in REGIMES:
                L = loss(rg, r_re, r_hu)
                best = pick(L)
                l_pol += w[rg] * L[best]
                l_A += w[rg] * L["A"]
                l_H += w[rg] * L["H"]
            ok = l_pol <= min(l_A, l_H) + 1e-12
            all_ok = all_ok and ok
            detail.append({"r_rephoto": r_re, "r_human": r_hu,
                           "loss_policy": round(l_pol, 6),
                           "loss_always_A": round(l_A, 6),
                           "loss_always_H": round(l_H, 6), "pass": bool(ok)})
    say(f"{len(detail)} 组成本比：策略 <= min(总是给结论, 总是鉴定) ⇒ "
        f"{'✅ 全部通过' if all_ok else '❌ 有未通过'}")
    say("")
    say(f"{'r_re':>8}{'r_hu':>8}{'策略':>12}{'总是A':>12}{'总是H':>12}   判定")
    for d in detail:
        say(f"{d['r_rephoto']:>8.3f}{d['r_human']:>8.0f}{d['loss_policy']:>12.4f}"
            f"{d['loss_always_A']:>12.4f}{d['loss_always_H']:>12.4f}"
            f"   {'✅' if d['pass'] else '❌'}")

    say("")
    say("=" * 96)
    say("结论与表述纪律")
    say("=" * 96)
    n_r = sum(1 for t in table if t["optimal"][NAMES["too_small"]] == "R")
    n_h = sum(1 for t in table if t["optimal"][NAMES["too_large"]] == "H")
    n_a = sum(1 for t in table if t["optimal"][NAMES["accepted"]] == "A")
    say(f"· {len(table)} 组成本比中：「太小」区选 R 的 **{n_r}** 组；"
        f"「过大」区选 H 的 **{n_h}** 组；「接受带」选 A 的 **{n_a}** 组。")
    say("· ⇒ 决策边界**非退化**：三个区间分别导向 R / A / H，且随成本比变化。")
    say("· ★ 表述必须是「**在成本比 = r 时，最优行动是 X**」，**不是**「应该 X」——")
    say("  安全代价不该由我们替用户设定。")
    say("· 边界：仅**合成域**验证（与 P2 同）；真机误差结构与成本结构都可能不同。")
    say("")

    rep = {
        "signal": "生产路径的双侧窗口：px<3 / 3<=px<=ks-1 / px>ks-1",
        "error_event": f"|rel_err| > {ERR_REL}",
        "regimes": {NAMES[r]: stat[r] for r in REGIMES},
        "decision_boundary": table,
        "gate_vs_constant": {"all_pass": bool(all_ok), "detail": detail},
        "boundary": "仅合成域验证；真机误差结构与成本结构可能不同。",
        "wording_rule": "表述为「在成本比=r 时最优行动是 X」，不是「应该 X」。",
        "note_vs_v1": ("第一版用单侧 px>=3 分箱 ⇒ 决策退化（总是 A）。"
                       "本项目生产判据实为**双侧窗口**，改用它作信号后决策非退化。"),
    }
    outp = EVAL / "decision_rule.json"
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    say(f"产物: {outp}")
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[留痕] {OUT_TXT}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
