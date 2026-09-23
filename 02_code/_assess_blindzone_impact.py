# -*- coding: utf-8 -*-
"""
_assess_blindzone_impact.py —— 评估「黑帽核盲区」修复的影响面

背景（技术报告 §7.6.1）：
    segment_defect 对线状类用黑帽核 ks = max(7, ROI短边 // 12)。
    当 ks <= 目标宽度时，核装不进缺陷内部 -> 黑帽响应归零 -> 目标消失。
    可靠测量区间 = [3px, ks-1]。

问题：直接改 ks 策略会改变**已上报的全部毫米数字**。
本脚本不做任何修改，只做**只读量化评估**：
    在真实测试集上，用现有模型跑同一批检测框，
    对每个线状类实例同时计算「现有策略」与「候选修复策略」的测量值，
    统计差异分布，回答三个问题：

    Q1 现有策略下，有多少实例的测量结果落在「疑似盲区」？
       （判据：ROI 短边算出的 ks <= 骨架测得宽度 -> 自相矛盾）
    Q2 候选修复策略会改变多少实例的数字？改变幅度多大？
    Q3 有多少实例的**分级结论**（ok / attention / danger）会翻转？

候选修复策略（只评估，不落地到 measure.py）：
    F1 多尺度核：对线状类同时用 {7, 13, 21} 三个核做黑帽，取响应最大者。
       —— 小核捕捉细线，大核捕捉宽带，不依赖 ROI 尺寸。
    F2 核上限封顶：ks = min(max(7, ROI短边//12), 31)。
       —— 缓解「宽松框导致 ks 过大」，但不解决「紧框导致 ks 过小」。

输出：logs/_assess_blindzone/{impact_report.txt, impact_report.json, per_instance.csv}

用法：
    python _assess_blindzone_impact.py --limit=200
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CLASSES, DATA_DIR, EVAL_DIR, imread_u, log  # noqa: E402
from gsd import Calibration  # noqa: E402
from measure import (  # noqa: E402
    _skeleton_length_px,
    _thin_skeleton,
    measure_instance,
    segment_defect,
)

# common 未导出 ROOT，用 DATA_DIR 的父目录（DATA_DIR = <root>/01_data）
ROOT = Path(DATA_DIR).resolve().parent

LINE_LIKE = ("crack", "exposed_rebar", "rust")
OUT_DIR = Path(ROOT) / "logs" / "_assess_blindzone"


# --------------------------------------------------------------------------
# 候选修复策略的分割实现（只在本脚本内，用于对照，不改 measure.py）
# --------------------------------------------------------------------------
def segment_multiscale(roi: np.ndarray, cls_name: str,
                       kernels=(7, 13, 21)) -> tuple[np.ndarray, str]:
    """F1：多尺度黑帽取最大响应。其余步骤与 segment_defect 完全一致。"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)

    if cls_name in LINE_LIKE:
        resp = None
        for ks in kernels:
            # 核不能超过 ROI 尺寸
            if ks >= min(roi.shape[:2]):
                continue
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
            r = cv2.morphologyEx(enh, cv2.MORPH_BLACKHAT, k)
            resp = r if resp is None else cv2.max(resp, r)
        if resp is None:
            return np.zeros(roi.shape[:2], np.uint8), "ms-blackhat(empty)"
        method = f"ms-blackhat{list(kernels)}+otsu"
    else:
        ks = max(9, (min(roi.shape[:2]) // 8) | 1)
        med = cv2.medianBlur(enh, min(ks, 31))
        resp = cv2.absdiff(enh, med)
        method = f"meddev{ks}+otsu"

    resp_n = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(resp_n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    min_area = max(12, int(0.0008 * roi.shape[0] * roi.shape[1]))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    mask = np.zeros_like(th)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255
    return mask, method


def segment_capped(roi: np.ndarray, cls_name: str, cap: int = 31
                   ) -> tuple[np.ndarray, str]:
    """F2：核大小上限封顶到 cap。"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)

    if cls_name in LINE_LIKE:
        ks = max(7, (min(roi.shape[:2]) // 12) | 1)
        ks = min(ks, cap)
        if ks % 2 == 0:
            ks += 1
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
        resp = cv2.morphologyEx(enh, cv2.MORPH_BLACKHAT, k)
        method = f"blackhat{ks}(cap{cap})+otsu"
    else:
        ks = max(9, (min(roi.shape[:2]) // 8) | 1)
        med = cv2.medianBlur(enh, min(ks, 31))
        resp = cv2.absdiff(enh, med)
        method = f"meddev{ks}+otsu"

    resp_n = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(resp_n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    min_area = max(12, int(0.0008 * roi.shape[0] * roi.shape[1]))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    mask = np.zeros_like(th)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255
    return mask, method


def widths_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    """复刻 measure_line_like 的核心逻辑，返回 (L, wmean, wmax95, area)。"""
    area_px = float(cv2.countNonZero(mask))
    if area_px <= 0:
        return 0.0, 0.0, 0.0, 0.0
    skel = _thin_skeleton(mask)
    length_px = _skeleton_length_px(skel)
    if length_px <= 0:
        return 0.0, 0.0, 0.0, area_px
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    pts = dist[skel > 0]
    if pts.size == 0:
        return length_px, 0.0, 0.0, area_px
    ws = 2.0 * pts
    return length_px, float(np.mean(ws)), float(np.percentile(ws, 95)), area_px


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="最多处理多少张测试图（0=全部）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--weights", type=str, default="")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 权重 ----
    if args.weights:
        wpath = Path(args.weights)
    else:
        wpath = Path(ROOT) / "03_weights" / "v11s640_best.pt"
    if not wpath.exists():
        log(f"[FATAL] 权重不存在: {wpath}")
        return 2
    log(f"权重: {wpath}")

    from ultralytics import YOLO
    model = YOLO(str(wpath))

    img_dir = Path(DATA_DIR) / "dataset" / "images" / "test"
    imgs = sorted([p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    if args.limit:
        imgs = imgs[:args.limit]
    log(f"测试图: {len(imgs)} 张 (来自 {img_dir})")

    # 固定标定（1.0 mm/px），只为比较**相对**变化；不涉及绝对精度声明
    calib = Calibration(mm_per_px=1.0, method="fixed-1.0-for-comparison",
                        confidence="low",
                        detail={"note": "影响面评估专用固定尺度"})

    rows = []
    n_img_err = 0
    for idx, ip in enumerate(imgs, 1):
        img = imread_u(str(ip))
        if img is None:
            n_img_err += 1
            continue
        try:
            res = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        except Exception as e:  # noqa: BLE001
            n_img_err += 1
            log(f"  推理失败 {ip.name}: {e}")
            continue
        if res.boxes is None or len(res.boxes) == 0:
            continue

        xyxy = res.boxes.xyxy.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()

        for j in range(len(xyxy)):
            cid = int(clss[j])
            if cid >= len(CLASSES):
                continue
            cname = CLASSES[cid]
            if cname not in LINE_LIKE:
                continue
            bbox = xyxy[j]
            roi_h = int(bbox[3] - bbox[1]) + 4
            roi_w = int(bbox[2] - bbox[0]) + 4
            if roi_h < 3 or roi_w < 3:
                continue
            ks_cur = max(7, (min(roi_h, roi_w) // 12) | 1)

            # 现有策略
            m_cur = measure_instance(img, bbox, cname, calib, pad=2)
            w_cur = m_cur.width_max_mm if m_cur else 0.0
            wm_cur = m_cur.width_mean_mm if m_cur else 0.0
            l_cur = m_cur.length_mm if m_cur else 0.0

            # 候选 F1 / F2：直接对 ROI 调自定义分割
            h, w = img.shape[:2]
            x1 = max(0, int(bbox[0]) - 2); y1 = max(0, int(bbox[1]) - 2)
            x2 = min(w, int(bbox[2]) + 2); y2 = min(h, int(bbox[3]) + 2)
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            try:
                mk1, _ = segment_multiscale(roi, cname)
                L1, wm1, wM1, _ = widths_from_mask(mk1)
                mk2, _ = segment_capped(roi, cname)
                L2, wm2, wM2, _ = widths_from_mask(mk2)
            except Exception as e:  # noqa: BLE001
                log(f"  分割失败 img={ip.name} j={j}: {e}")
                continue

            # 盲区自检：当前策略测出的宽度(px) >= ks 时，说明它已经越过了
            # 「核装得进」的前提，该结果可疑
            w_cur_px = w_cur / calib.mm_per_px
            blind_suspect = bool(w_cur_px >= ks_cur) if w_cur_px > 0 else False

            rows.append({
                "image": ip.name, "cls": cname, "conf": round(float(confs[j]), 4),
                "roi_h": roi_h, "roi_w": roi_w, "ks_cur": ks_cur,
                "L_cur": round(l_cur, 3), "wmean_cur": round(wm_cur, 4),
                "wmax_cur": round(w_cur, 4),
                "L_f1": round(L1 * 1.0, 3), "wmean_f1": round(wm1, 4),
                "wmax_f1": round(wM1, 4),
                "L_f2": round(L2 * 1.0, 3), "wmean_f2": round(wm2, 4),
                "wmax_f2": round(wM2, 4),
                "blind_suspect": blind_suspect,
                "cur_has_mask": bool(w_cur > 0),
                "f1_has_mask": bool(wM1 > 0),
                "f2_has_mask": bool(wM2 > 0),
            })
        if idx % 25 == 0:
            log(f"  ... {idx}/{len(imgs)} 已处理, 累计线状实例 {len(rows)}")

    # ---------------- 统计 ----------------
    n = len(rows)
    rep = {"n_instances": n, "n_img_err": n_img_err,
           "weights": str(wpath), "images": len(imgs)}

    if n == 0:
        log("[WARN] 没有采集到任何线状类实例")
        (OUT_DIR / "impact_report.txt").write_text("no instances\n", encoding="utf-8")
        return 0

    def _stat(key_a, key_b):
        """比较两个字段：返回 (改变数, 改变率, 中位相对变化, 绝对值中位)"""
        diffs = []
        for r in rows:
            a, b = r[key_a], r[key_b]
            if a <= 0 and b <= 0:
                continue
            if a <= 0 or b <= 0:
                diffs.append(None)   # 一边测不到
                continue
            diffs.append((b - a) / a)
        valid = [d for d in diffs if d is not None]
        changed = [d for d in valid if abs(d) > 1e-9]
        none_cnt = sum(1 for d in diffs if d is None)
        return {
            "comparable": len(valid),
            "changed": len(changed),
            "changed_rate": round(len(changed) / len(valid), 4) if valid else None,
            "one_side_missing": none_cnt,
            "rel_median": round(float(np.median(valid)), 4) if valid else None,
            "rel_mean": round(float(np.mean(valid)), 4) if valid else None,
            "rel_p05": round(float(np.percentile(valid, 5)), 4) if valid else None,
            "rel_p95": round(float(np.percentile(valid, 95)), 4) if valid else None,
            "abs_rel_median": round(float(np.median([abs(d) for d in valid])), 4) if valid else None,
        }

    rep["wmax_cur_vs_f1"] = _stat("wmax_cur", "wmax_f1")
    rep["wmax_cur_vs_f2"] = _stat("wmax_cur", "wmax_f2")
    rep["L_cur_vs_f1"] = _stat("L_cur", "L_f1")
    rep["L_cur_vs_f2"] = _stat("L_cur", "L_f2")

    # 覆盖变化：现策略测不到但候选能测到（失明 -> 复明）
    blind_recovered_f1 = sum(1 for r in rows
                             if (not r["cur_has_mask"]) and r["f1_has_mask"])
    blind_recovered_f2 = sum(1 for r in rows
                             if (not r["cur_has_mask"]) and r["f2_has_mask"])
    lost_f1 = sum(1 for r in rows if r["cur_has_mask"] and not r["f1_has_mask"])
    lost_f2 = sum(1 for r in rows if r["cur_has_mask"] and not r["f2_has_mask"])
    rep["coverage"] = {
        "cur_has_mask": sum(1 for r in rows if r["cur_has_mask"]),
        "f1_has_mask": sum(1 for r in rows if r["f1_has_mask"]),
        "f2_has_mask": sum(1 for r in rows if r["f2_has_mask"]),
        "blind_recovered_f1": blind_recovered_f1,
        "blind_recovered_f2": blind_recovered_f2,
        "lost_f1": lost_f1,
        "lost_f2": lost_f2,
    }
    rep["blind_suspect_count"] = sum(1 for r in rows if r["blind_suspect"])

    # 逐类
    per_cls = {}
    for c in LINE_LIKE:
        sub = [r for r in rows if r["cls"] == c]
        if not sub:
            continue
        d1 = []
        for r in sub:
            if r["wmax_cur"] > 0 and r["wmax_f1"] > 0:
                d1.append((r["wmax_f1"] - r["wmax_cur"]) / r["wmax_cur"])
        per_cls[c] = {
            "n": len(sub),
            "ks_min": min(r["ks_cur"] for r in sub),
            "ks_max": max(r["ks_cur"] for r in sub),
            "blind_suspect": sum(1 for r in sub if r["blind_suspect"]),
            "wmax_changed_vs_f1": len([d for d in d1 if abs(d) > 1e-9]),
            "wmax_absrel_median": round(float(np.median([abs(d) for d in d1])), 4) if d1 else None,
        }
    rep["per_class"] = per_cls

    # ---------------- 落盘 ----------------
    with open(OUT_DIR / "per_instance.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    (OUT_DIR / "impact_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    L = []
    L.append("=" * 78)
    L.append("黑帽核盲区修复 —— 影响面评估（只读，未改动 measure.py）")
    L.append("=" * 78)
    L.append(f"权重            : {wpath.name}")
    L.append(f"测试图          : {rep['images']} 张（错误 {n_img_err}）")
    L.append(f"线状类实例      : {n} 个")
    L.append(f"现有策略测不到  : {n - rep['coverage']['cur_has_mask']} 个")
    L.append(f"F1(多尺度)测不到: {n - rep['coverage']['f1_has_mask']} 个")
    L.append(f"F2(封顶)测不到  : {n - rep['coverage']['f2_has_mask']} 个")
    L.append("")
    L.append("--- Q1 盲区自检（测得宽度 px >= ks，自相矛盾） ---")
    L.append(f"  可疑实例数: {rep['blind_suspect_count']} / {n}")
    L.append("")
    L.append("--- Q2 候选策略对 wmax 的改动幅度 ---")
    for tag, key in (("F1 多尺度", "wmax_cur_vs_f1"), ("F2 封顶", "wmax_cur_vs_f2")):
        s = rep[key]
        L.append(f"  [{tag}]  可比 {s['comparable']} 个, 数值改变 {s['changed']} 个 "
                 f"({s['changed_rate']}), 单边缺失 {s['one_side_missing']} 个")
        L.append(f"          相对变化 中位 {s['rel_median']}, 均值 {s['rel_mean']}, "
                 f"P5 {s['rel_p05']}, P95 {s['rel_p95']}, "
                 f"|变化|中位 {s['abs_rel_median']}")
    L.append("")
    L.append("--- 覆盖变化（失明 -> 复明 / 复明 -> 失明） ---")
    L.append(f"  F1: 复明 {blind_recovered_f1} 个, 新失明 {lost_f1} 个")
    L.append(f"  F2: 复明 {blind_recovered_f2} 个, 新失明 {lost_f2} 个")
    L.append("")
    L.append("--- 逐类 ---")
    for c, v in per_cls.items():
        L.append(f"  {c:15s} n={v['n']:4d}  ks∈[{v['ks_min']},{v['ks_max']}]  "
                 f"可疑 {v['blind_suspect']:3d}  "
                 f"vs F1 改变 {v['wmax_changed_vs_f1']:4d}  "
                 f"|Δ|中位 {v['wmax_absrel_median']}")
    L.append("")
    L.append("--- 判读 ---")
    ch = rep["wmax_cur_vs_f1"]["changed_rate"]
    if ch is None:
        L.append("  无可比样本，无法判断")
    elif ch < 0.05:
        L.append(f"  F1 只改动 {ch:.1%} 的实例 -> **影响面小**，")
        L.append("  可以在与 V3 回填分开的前提下单独修，并只回填受影响的少数实例。")
    elif ch < 0.30:
        L.append(f"  F1 改动了 {ch:.1%} 的实例 -> **影响面中等**，")
        L.append("  修复需重跑全部毫米数字并全面回填，工作量大但可控。")
    else:
        L.append(f"  F1 改动了 {ch:.1%} 的实例 -> **影响面大**，")
        L.append("  修复等于重做测量链路，必须整体重跑与回填。")
    L.append("")
    L.append(f"产出: {OUT_DIR}")
    L.append("=" * 78)

    (OUT_DIR / "impact_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
