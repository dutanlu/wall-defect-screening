# -*- coding: utf-8 -*-
"""
_a_collect_features.py —— A 路【学习式弃权器】数据采集

===== 目标 =====
为「可学习弃权器 vs 手写几何判据 vs 纯 softmax 置信度」三方对决，
采集一张**逐实例特征表**，字段含三类信号源：

  ① 模型侧（softmax 基线用）  : conf
  ② 几何侧（手写判据用）      : ks_cur, px_on(=wmean_cur), window_ok, roi_h/w
  ③ 多尺度侧（本文新信号）    : F1 多尺度核下的 wmean/L，与 cur 的相对差

===== 标签怎么来的（真值无关，可复算）=====
本实验**不需要人工标注**。弃权的"真值"用**方法间一致性**定义：

    measure_ok = |wmean_cur - wmean_f1| / max(wmean_cur, wmean_f1, eps) < TAU_AGREE

直觉：黑帽核 ks 由 ROI 短边决定（ks = max(7, short//12)）。
当 ks 装不进目标内部时黑帽响应归零 ⇒ 测出的宽度是"塌陷值"，不可信。
F1（多尺度核 {7,13,21} 取响应最大者）**不受单核尺寸限制**，是更稳的参照。
两者差异大 ⇒ cur 的测量不可信 ⇒ **应弃权**。

⚠️ 局限（必须写进报告）：F1 也可能错。故这不是"绝对真值"，
   而是"**共识真值**"（consensus ground truth）。

输出：
  logs/_a_abstainer/features.csv
  logs/_a_abstainer/collect_report.json
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

from common import CLASSES, DATA_DIR, DATASET_DIR, imread_u, log  # noqa: E402
from gsd import Calibration  # noqa: E402
from measure import measure_instance, _skeleton_length_px, _thin_skeleton  # noqa: E402

ROOT = Path(DATA_DIR).resolve().parent
OUT_DIR = ROOT / "logs" / "_a_abstainer"

TAU_AGREE = 0.15
LINE_LIKE = {"crack", "exposed_rebar", "rust"}


def _multiscale_measure(image, bbox, cls_name):
    """F1 多尺度核参照：线状类取 {7,13,21} 三核黑帽中响应最大者再测量。"""
    if cls_name not in LINE_LIKE:
        return None
    x1, y1, x2, y2 = bbox
    roi = image[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    best = None
    for ks in (7, 13, 21):
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker)
        _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n = int((mask > 0).sum())
        if best is None or n > best[0]:
            best = (n, mask)
    if best is None or best[0] == 0:
        return None
    skel = _thin_skeleton(best[1] > 0)
    # ⚠️ `_thin_skeleton()` 返回的是 **uint8 图像（0/255）**，不是布尔掩码。
    #    直接 `dist[skel]` 会用 255 当下标 ⇒ IndexError: index 255 out of bounds。
    #    （2026-09-25 实跑踩到；本句原本就会让整个采集崩掉。）
    skel_mask = skel > 0
    L = float(_skeleton_length_px(skel))
    dist = cv2.distanceTransform((best[1] > 0).astype(np.uint8), cv2.DIST_L2, 5)
    vals = dist[skel_mask]
    if vals.size == 0:
        return None
    return dict(L_px=L, wmean_px=float(vals.mean()) * 2.0,
                wmax_px=float(vals.max()) * 2.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--weights", default=str(ROOT / "03_weights" / "v11s640_best.pt"))
    args = ap.parse_args()

    from ultralytics import YOLO

    imgs = sorted((DATASET_DIR / "images" / "test").glob("*"))
    imgs = [p for p in imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if args.limit:
        imgs = imgs[: args.limit]
    log(f"A 路采集：{len(imgs)} 张测试图，权重 {Path(args.weights).name}")

    model = YOLO(args.weights)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, n_img_err = [], 0
    _measure_err = 0            # measure_instance 异常计数（★ 不静默吞掉）
    for i, p in enumerate(imgs, 1):
        img = imread_u(p)
        if img is None:
            n_img_err += 1
            continue
        res = model.predict(img, conf=args.conf, verbose=False, device=0)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        # ⚠️ Calibration 的字段是 (mm_per_px, method, confidence, detail)，
        #    **没有 source**；且 method / confidence 均无默认值 ⇒ 必须全传。
        #    （本实验只关心 **像素** 级特征，mm_per_px 取 1.0 当占位，
        #      不参与任何以 mm 为单位的判据 —— 见文件头「特征表字段」说明。）
        calib = Calibration(mm_per_px=1.0, method="placeholder",
                            confidence="low", detail={"note": "仅用于像素级特征采集"})
        for bi in range(len(res.boxes)):
            xyxy = res.boxes.xyxy[bi].cpu().numpy().astype(int)
            cls_id = int(res.boxes.cls[bi].cpu().item())
            conf = float(res.boxes.conf[bi].cpu().item())
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
            try:
                # ⚠️ measure_instance 的真实签名是 (image, bbox_xyxy, cls_name, calib, pad=2)
                #    —— **没有 `image_quality_ok` 参数**（曾误传，会 TypeError）。
                #    质量门控不在本函数里；本实验只需像素级测量结果。
                m = measure_instance(img, tuple(xyxy), cls_name, calib)
            except Exception as e:      # noqa: BLE001
                # ★ 纪律：绝不静默吞掉异常。宽泛捕获会把「签名写错」伪装成「数据缺失」，
                #   本项目已因此误判过一次。⇒ 计入 _measure_err 并在末尾报警。
                m = None
                _measure_err += 1
                if _measure_err <= 3:   # 只打前 3 条，避免刷屏
                    log(f"  ⚠️ measure_instance 异常 ×{_measure_err}: "
                        f"{type(e).__name__}: {e}  （{cls_name} {xyxy.tolist()}）")
            ms = _multiscale_measure(img, tuple(xyxy), cls_name)

            wmean_cur = getattr(m, "width_mean_px", 0.0) if m else 0.0
            if ms is not None and wmean_cur > 0:
                rel = abs(wmean_cur - ms["wmean_px"]) / max(wmean_cur, ms["wmean_px"], 1e-6)
                lab = int(rel < TAU_AGREE)
            else:
                rel, lab = float("nan"), -1

            rows.append(dict(
                image=p.name, cls=cls_name, conf=round(conf, 4),
                roi_h=int(xyxy[3] - xyxy[1]), roi_w=int(xyxy[2] - xyxy[0]),
                roi_short=max(1, min(int(xyxy[3] - xyxy[1]), int(xyxy[2] - xyxy[0]))),
                ks_cur=int(getattr(m, "ks", 0)) if m else 0,
                window_ok=int(bool(getattr(m, "window_ok", True))) if m else 0,
                L_cur=round(float(getattr(m, "length_px", 0.0)) if m else 0.0, 3),
                wmean_cur=round(wmean_cur, 4),
                wmax_cur=round(float(getattr(m, "width_max_px", 0.0)) if m else 0.0, 4),
                L_f1=round(ms["L_px"], 3) if ms else "",
                wmean_f1=round(ms["wmean_px"], 4) if ms else "",
                wmax_f1=round(ms["wmax_px"], 4) if ms else "",
                rel_diff=round(rel, 4) if ms is not None and wmean_cur > 0 else "",
                label_ok=lab,
                method=getattr(m, "method", "") if m else "",
            ))
        if i % 50 == 0:
            log(f"  进度 {i}/{len(imgs)}，累计实例 {len(rows)}")

    if not rows:
        log("!! 零实例，检查权重/阈值"); return 1

    csv_path = OUT_DIR / "features.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)

    from collections import Counter
    lab = [r for r in rows if r["label_ok"] >= 0]
    n_ok = sum(1 for r in lab if r["label_ok"] == 1)

    # ★ 逐类拆解「可监督 / 正例」——A 路的关键风险是「正例太稀有」，
    #   必须知道正例集中在哪些类，才能在触发门禁时做出知情决策
    #   （eg. 若正例全来自 rust，则改口径只影响 rust 的结论，须如实说明）。
    per_class_lab: dict[str, dict[str, int]] = {}
    for r in rows:
        d = per_class_lab.setdefault(r["cls"], {"inst": 0, "labeled": 0, "pos": 0})
        d["inst"] += 1
        if r["label_ok"] >= 0:
            d["labeled"] += 1
        if r["label_ok"] == 1:
            d["pos"] += 1

    rep = dict(
        n_instances=len(rows), n_images=len(imgs), n_img_err=n_img_err,
        n_measure_err=_measure_err,
        weights=Path(args.weights).name, conf_thr=args.conf, tau_agree=TAU_AGREE,
        per_class=dict(Counter(r["cls"] for r in rows)),
        per_class_labeled=per_class_lab,
        n_labeled=len(lab), n_label_ok=n_ok,
        label_ok_rate=round(n_ok / len(lab), 4) if lab else None,
        note="label_ok = cur 与 F1多尺度 相对差 < TAU_AGREE（共识真值，非绝对真值）",
    )
    (OUT_DIR / "collect_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"写出 {csv_path}（{len(rows)} 实例，可监督 {len(lab)}，ok={n_ok}）")
    log("  逐类（实例/可监督/正例）：")
    for c, d in sorted(per_class_lab.items(), key=lambda kv: -kv[1]["inst"]):
        log(f"    {c:16s} {d['inst']:5d} / {d['labeled']:5d} / {d['pos']:4d}")
    if len(lab) and n_ok < 20:
        log(f"⚠️ 正例仅 {n_ok} 个（<20）⇒ `_a_abstainer_duel.py` 会拒绝运行（exit=6）。"
            f" 需在报告里如实说明换口径（扩样本/放宽 TAU/不二值化）后再跑。")
    if _measure_err:
        log(f"⚠️ 注意：{_measure_err} 次 measure_instance 异常（已按缺失处理，"
            f"若比例高须查签名/数据，不要当成正常）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
