# -*- coding: utf-8 -*-
r"""P2：给测量值配「**带覆盖保证的区间**」（split conformal + Mondrian 分箱）。

## 要解决什么
现在系统对一个缺陷只报**一个点值**（如「宽度 0.35mm」）。本项把它升级为
「0.35mm，**90% 区间 [lo, hi]**」，且该区间的覆盖率**在独立集上验证过**。

## 为什么用 conformal 而不是「均值±2σ」
conformal 给的是**有限样本覆盖保证**（不依赖误差的正态假设），只需"可交换性"。
而本项目的测量误差**已知是系统性偏正的**（§7.6.2：细目标 +20%~+100%），
非对称分布 ⇒ 必须用**带符号残差**，不能用对称的 σ 区间。

## 关键设计
  · 非一致性分数 = **带符号残差** `r = ŵ − w_true`（不是绝对值）
    ⇒ 区间 `[ŵ − q_hi, ŵ − q_lo]`（`q_lo`, `q_hi` 是 r 的 α/2 与 1−α/2 分位数）
  · **Mondrian 分箱**：按 `px_on_target` 分箱。依据是 §7.6/§7.7 的发现
    「误差强烈依赖目标占像素数」⇒ 不分箱会导致条件覆盖严重失衡。
    分箱边界用 `measure.py` 已有的两个门槛（min_px=3、comfort=8）⇒ 与系统语义一致。
  · 分位数用 **numpy** 自实现（本机 `scipy.stats` 被应用控制策略阻断 DLL）。
  · `px_on_target < 3` 的样本**不给区间**（沿用 `window_ok` 的弃权语义）。

## ★ 诚实边界（必须随结论写）
校准集与测试集**都是合成的** ⇒ 只能声称「**在合成域上覆盖率得到验证**」，
**不得**声称「真机 90% 保证」（合成→真机不满足可交换性）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\pythonstudy 备份\创新题\外墙缺陷筛查")
sys.path.insert(0, str(ROOT / "02_code"))
sys.path.insert(0, str(ROOT / "logs"))

import _verify_measure_accuracy as V          # noqa: E402
from gsd import Calibration                    # noqa: E402
from measure import measure_instance           # noqa: E402

EVAL = ROOT / "04_results" / "eval"
OUT_TXT = ROOT / "logs" / "_p2_conformal.txt"

MIN_PX = 3        # 与 gsd.assess_interpretability 的 min_px 一致
COMFORT_PX = 8    # 与 quantify 门槛一致（8px）


def gen_case(d_mm: float, width_mm: float, seed: int, angle_deg: float = 0.0):
    """生成一个合成裂缝样本，返回 (w_true, w_hat, gsd, px_on_target, seed)。

    ★ 线长按 GSD 自适应：`length_mm = 1500 * GSD` ⇒ 无论远近，线长恒约 1500px，
      不会被 3200x2200 的画布裁掉（第一版固定 900mm，在 d=600 时变成 4500px 被裁）。
    """
    Hp2i = V.plane_to_img_homography(angle_deg, d=d_mm)
    mmpp = V.mmpp_at(Hp2i, 0.0, 0.0)
    length_mm = 1500.0 * mmpp
    img = V.render_canvas(angle_deg, seed=seed)
    pts = V.line_pts(0.0, 0.0, width_mm, length_mm, 0.0)
    V.draw_line_band(img, Hp2i, 0.0, 0.0, width_mm, length_mm, 40.0, 0.0)
    bb = V.bbox_of_world(Hp2i, pts)
    calib = Calibration(mm_per_px=mmpp, method="truth", confidence="high")
    m = measure_instance(img, bb, "crack", calib)
    px = width_mm / mmpp
    if m is None or m.width_mean_px <= 0:
        return {"w_true": width_mm, "w_hat": None, "gsd": mmpp,
                "px_on_target": px, "seed": seed, "d_mm": d_mm}
    return {"w_true": width_mm, "w_hat": m.width_mean_mm, "gsd": mmpp,
            "px_on_target": px, "seed": seed, "d_mm": d_mm}


def build_corpus(tag: str, n_seed: int, seed0: int) -> list[dict]:
    """跨 GSD（用距离扫）与跨宽度生成语料。"""
    rows = []
    for k in range(n_seed):
        seed = seed0 + k
        # 距离扫 ⇒ 扫 GSD；宽度扫 ⇒ 扫 px_on_target
        # ★ 距离扫宽（GSD 0.20~2.17 mm/px）× 宽度扫宽 ⇒ 三个 px 箱都有数据
        for d_mm in (600.0, 1000.0, 1500.0, 2500.0, 4000.0, 6500.0):
            for wmm in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0):
                rows.append(gen_case(d_mm, wmm, seed))
    return rows


def conformal_q(scores: np.ndarray, level: float) -> float:
    """split conformal 的有限样本分位数（numpy 自实现）。

    取 `ceil((n+1)*level)/n` 位置的分位数（method='higher'），
    这是保证边缘覆盖 ≥ level 的标准做法。
    """
    n = len(scores)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * level))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def bin_of(px: float) -> str:
    if px < MIN_PX:
        return "A_abstain(<3px)"
    if px < COMFORT_PX:
        return "B_screen(3-8px)"
    return "C_quantify(>=8px)"


def main() -> int:
    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(str(s))

    ALPHA = 0.10          # 目标覆盖率 90%
    say("=" * 92)
    say(f"P2：测量值的 conformal 区间（目标覆盖率 {1 - ALPHA:.0%}）")
    say("=" * 92)
    say("")
    say("生成校准集与测试集（**独立 seed**）…")
    cal_rows = build_corpus("cal", n_seed=10, seed0=1000)
    test_rows = build_corpus("test", n_seed=6, seed0=9000)
    say(f"  校准集 {len(cal_rows)} 条 / 测试集 {len(test_rows)} 条")

    def usable(rows):
        return [r for r in rows
                if r["w_hat"] is not None and r["px_on_target"] >= MIN_PX]

    cal = usable(cal_rows)
    tst = usable(test_rows)
    say(f"  可用（分割成功且 px≥{MIN_PX}）：校准 {len(cal)} / 测试 {len(tst)}")
    n_abst = sum(1 for r in cal_rows + test_rows
                 if r["w_hat"] is None or r["px_on_target"] < MIN_PX)
    say(f"  弃权（px<{MIN_PX} 或分割失败）：{n_abst} 条 —— 沿用 `window_ok` 语义，不给区间")
    say("")

    # ---- 分箱统计残差 ----
    say("按 px_on_target 分箱统计带符号残差 r = ŵ − w_true：")
    say(f"{'箱':>20}{'n':>6}{'r 中位':>10}{'r 均值':>10}{'r 90%':>10}{'r 10%':>10}")
    say("-" * 68)
    bins = {}
    for r in cal:
        bins.setdefault(bin_of(r["px_on_target"]), []).append(r["w_hat"] - r["w_true"])
    qtab = {}
    for b in sorted(bins):
        a = np.array(bins[b])
        qlo = conformal_q(a, ALPHA / 2)
        qhi = conformal_q(a, 1 - ALPHA / 2)
        qtab[b] = {"n": len(a), "q_lo": qlo, "q_hi": qhi}
        say(f"{b:>20}{len(a):>6}{np.median(a):>10.4f}{a.mean():>10.4f}"
            f"{np.percentile(a, 90):>10.4f}{np.percentile(a, 10):>10.4f}")
    say("")
    say("⇒ 各箱的 q_lo / q_hi（用于区间 [ŵ − q_hi, ŵ − q_lo]）：")
    for b in sorted(qtab):
        say(f"   {b:>20}  q_lo={qtab[b]['q_lo']:+.4f}  q_hi={qtab[b]['q_hi']:+.4f}  (n={qtab[b]['n']})")

    # ---- 在独立测试集上验覆盖率 ----
    say("")
    say("=" * 92)
    say("★ 独立测试集上的覆盖率验证")
    say("=" * 92)
    cover_bin = {}
    for r in tst:
        b = bin_of(r["px_on_target"])
        if b not in qtab:
            continue
        lo = r["w_hat"] - qtab[b]["q_hi"]
        hi = r["w_hat"] - qtab[b]["q_lo"]
        d = cover_bin.setdefault(b, {"n": 0, "cov": 0, "width": []})
        d["n"] += 1
        d["cov"] += int(lo <= r["w_true"] <= hi)
        d["width"].append(hi - lo)
    say(f"{'箱':>20}{'n':>6}{'覆盖率':>10}{'目标':>8}{'区间宽度中位':>14}   判定")
    say("-" * 74)
    tot_n = tot_c = 0
    for b in sorted(cover_bin):
        d = cover_bin[b]
        c = d["cov"] / d["n"]
        tot_n += d["n"]
        tot_c += d["cov"]
        ok = "✅" if c >= 1 - ALPHA else "❌"
        say(f"{b:>20}{d['n']:>6}{c:>10.1%}{1 - ALPHA:>8.0%}"
            f"{np.median(d['width']):>14.4f}   {ok}")
    say("-" * 74)
    overall = tot_c / tot_n if tot_n else float("nan")
    say(f"{'总体（Mondrian）':>20}{tot_n:>6}{overall:>10.1%}{1 - ALPHA:>8.0%}")

    # ---- 对照：不分箱（全局分位数）----
    allr = np.array([r["w_hat"] - r["w_true"] for r in cal])
    g_lo = conformal_q(allr, ALPHA / 2)
    g_hi = conformal_q(allr, 1 - ALPHA / 2)
    cov_g = sum(1 for r in tst
                if (r["w_hat"] - g_hi) <= r["w_true"] <= (r["w_hat"] - g_lo))
    say(f"{'对照：不分箱':>20}{len(tst):>6}{cov_g / len(tst) if tst else float('nan'):>10.1%}"
        f"{1 - ALPHA:>8.0%}")
    say("")
    say("⇒ 若「不分箱」也达标，则 Mondrian 的价值体现在**各箱的条件覆盖**上（上表逐箱列）。")

    # ---- 结论与边界 ----
    say("")
    say("=" * 92)
    say("结论与边界")
    say("=" * 92)
    pass_overall = overall >= 1 - ALPHA
    say(f"· 总体覆盖率 {overall:.1%} ≥ 目标 {1 - ALPHA:.0%} ："
        f"{'✅ 通过' if pass_overall else '❌ 未通过'}")
    say("· ★ **边界（必须随结论写）**：校准集与测试集**都是合成**的 ⇒")
    say("  只能声称「**在合成域上覆盖率得到验证**」，**不得**声称「真机 90% 保证」")
    say("  （合成 → 真机不满足 conformal 所需的可交换性）。")
    say("· 与弃权机制的衔接：`px_on_target < 3` 的样本**不给区间**，直接弃权（沿用 `window_ok`）。")
    say("")

    rep = {
        "alpha": ALPHA, "target_coverage": 1 - ALPHA,
        "n_cal": len(cal), "n_test": len(tst), "n_abstain": n_abst,
        "quantiles_by_bin": qtab,
        "coverage_by_bin": {b: {"n": d["n"], "coverage": d["cov"] / d["n"],
                                "median_width_mm": round(float(np.median(d["width"])), 4)}
                            for b, d in cover_bin.items()},
        "overall_coverage_mondrian": overall,
        "overall_coverage_no_binning": (cov_g / len(tst)) if tst else None,
        "pass_overall": bool(pass_overall),
        "boundary": "仅**合成域**上验证；不得声称真机 90% 保证（不满足可交换性）。",
    }
    outp = EVAL / "conformal_measure.json"
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    say(f"产物: {outp}")
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[留痕] {OUT_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
