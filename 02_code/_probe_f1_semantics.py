# -*- coding: utf-8 -*-
"""
_probe_f1_semantics.py —— 隔离验证「F1 多尺度黑帽」到底在改什么

_assess_blindzone_impact.py 发现 F1 让 90% 实例的 wmax 改变、中位 +50%。
必须先弄清这是「修了盲区」还是「引入过度分割」，再决定能不能用。

做法：合成已知宽度的暗带（复用 §7.6 的渲染思路，但简化到最小可判），
在同一 ROI 上跑三种策略，比较测出的宽度：
  cur  : ks = max(7, 短边//12)          —— 现有
  f1   : 多尺度 {7,13,21} 取响应最大     —— 候选
  f2   : ks 封顶 31                     —— 候选

若 F1 在**应在盲区内的宽度**上给出正确值，则 F1 是修 bug；
若 F1 在**本该正确**的宽度上也偏离，则是引入了新偏差。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA_DIR  # noqa: E402
from measure import _skeleton_length_px, _thin_skeleton  # noqa: E402
from _assess_blindzone_impact import segment_capped, segment_multiscale  # noqa: E402

ROOT = Path(DATA_DIR).resolve().parent
OUT_DIR = Path(ROOT) / "logs" / "_assess_blindzone"


def make_band_roi(rh: int, rw: int, w_px: int, val: float = 40.0,
                  bg: float = 205.0, seed: int = 11) -> np.ndarray:
    """亮背景 + 水平居中竖直暗带（宽 w_px）。返回 BGR。"""
    rng = np.random.default_rng(seed)
    g = np.full((rh, rw), bg, np.float32)
    g += rng.normal(0, 3.0, g.shape)
    g = np.clip(g, 0, 255)
    x0 = (rw - w_px) // 2
    g[:, x0:x0 + w_px] = val
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def widths(mask):
    a = float(cv2.countNonZero(mask))
    if a <= 0:
        return 0.0, 0.0, 0.0
    sk = _thin_skeleton(mask)
    L = _skeleton_length_px(sk)
    if L <= 0:
        return 0.0, 0.0, a
    d = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    p = d[sk > 0]
    if p.size == 0:
        return L, 0.0, a
    return L, float(np.percentile(2.0 * p, 95)), a


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    # ROI 尺寸覆盖「ks 被下限兜住」与「ks 被公式抬高」两区
    for (rh, rw) in ((100, 400), (300, 400), (400, 400), (600, 600)):
        ks_cur = max(7, (min(rh, rw) // 12) | 1)
        for w_px in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40):
            if w_px >= rw - 20:
                continue
            roi = make_band_roi(rh, rw, w_px)
            m_cur, _ = __import__("measure").segment_defect(roi, "crack")
            m_f1, _ = segment_multiscale(roi, "crack")
            m_f2, _ = segment_capped(roi, "crack")
            _, w_cur, _ = widths(m_cur)
            _, w_f1, _ = widths(m_f1)
            _, w_f2, _ = widths(m_f2)

            def rel(v):
                return None if v <= 0 else round((v - w_px) / w_px, 4)

            rows.append({
                "roi": f"{rh}x{rw}", "ks_cur": ks_cur, "w_true_px": w_px,
                "w_cur": round(w_cur, 3), "w_f1": round(w_f1, 3),
                "w_f2": round(w_f2, 3),
                "rel_cur": rel(w_cur), "rel_f1": rel(w_f1), "rel_f2": rel(w_f2),
                "in_blind_cur": bool(w_px >= ks_cur),
            })

    with open(OUT_DIR / "f1_semantics.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    L = []
    L.append("=" * 92)
    L.append("F1/F2 语义隔离验证 —— 合成暗带，真值即绘制宽度")
    L.append("=" * 92)
    L.append(f"{'ROI':>9} {'ks':>3} {'真值':>4} | {'cur':>7} {'rel':>8} | "
             f"{'f1':>7} {'rel':>8} | {'f2':>7} {'rel':>8} | 盲区?")
    L.append("-" * 92)
    for r in rows:
        L.append(f"{r['roi']:>9} {r['ks_cur']:>3} {r['w_true_px']:>4} | "
                 f"{r['w_cur']:>7.2f} {str(r['rel_cur']):>8} | "
                 f"{r['w_f1']:>7.2f} {str(r['rel_f1']):>8} | "
                 f"{r['w_f2']:>7.2f} {str(r['rel_f2']):>8} | "
                 f"{'YES' if r['in_blind_cur'] else ''}")
    L.append("-" * 92)

    # 统计：在「现策略正确(±20%)」的样本上，F1 是否也正确
    def count_ok(key):
        return sum(1 for r in rows
                   if r[key] is not None and abs(r[key]) <= 0.20)

    L.append(f"落在 ±20% 的样本数：cur {count_ok('rel_cur')} / "
             f"f1 {count_ok('rel_f1')} / f2 {count_ok('rel_f2')}  （共 {len(rows)}）")

    ok_cur = [r for r in rows
              if r["rel_cur"] is not None and abs(r["rel_cur"]) <= 0.20]
    if ok_cur:
        broke = [r for r in ok_cur
                 if r["rel_f1"] is None or abs(r["rel_f1"]) > 0.20]
        L.append(f"其中 cur 正确、但 F1 变错：{len(broke)} 个 -> "
                 f"{[ (r['roi'], r['w_true_px'], r['rel_f1']) for r in broke ]}")
    blind = [r for r in rows if r["in_blind_cur"]]
    if blind:
        fixed = [r for r in blind
                 if r["rel_f1"] is not None and abs(r["rel_f1"]) <= 0.20]
        L.append(f"盲区内样本 {len(blind)} 个，F1 修好 {len(fixed)} 个")

    L.append("")
    L.append("产出: " + str(OUT_DIR / "f1_semantics.csv"))
    L.append("=" * 92)
    (OUT_DIR / "f1_semantics.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
