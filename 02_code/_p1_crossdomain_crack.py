# -*- coding: utf-8 -*-
r"""I1：BFDD 裂缝的**零样本跨源评测**（不训练、不改 V3）。

## 为什么这条不受 P1 那个混淆影响
P1 比较的是「不同**输入模态**」，而 BFDD 的标签由可见光判读定义 ⇒ 对红外不公。
本项比较的是「**同一个模型、同一个模态（RGB），换一个数据源**」⇒ 不存在该混淆：
  · 模型 = 主力 `v11s640_best.pt`（在 V3 上训练）
  · 输入 = BFDD 的 **RGB** 图（不同采集者/设备/城市/分辨率/任务）
  · 用途 = **纯评测**，不训练、不微调、不改 V3 的任何已上报数字

## 标注语义差异（必须声明）
BFDD 是**语义分割**掩膜，转成检测框后与 V3 的检测框标注**语义不同**：
细长裂缝网的**外接框**会明显大于「一个裂缝实例」的框。
实测（test 前 30 图）：连通域中位 7 个/图，外接框尺寸中位 4.4%×7.8% 全图，
仅 11.7% 的连通域有边超过半幅。
⇒ 因此**除严格 AP50 外，另报一个免 IoU 阈值的「命中率」**，两个口径并列。

## 判据（事先定死）
  · 严格口径：class 0（crack）的 **AP50**，与 V3 域内 `crack` AP50 = **0.7283** 对比
  · 宽松口径：GT 框被任一 class-0 预测框**覆盖到**（交叠面积 / GT 面积 ≥ 0.1）的比例
  · **不训练、不改 V3**；结果无论升降都如实报告
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\pythonstudy 备份\创新题\外墙缺陷筛查")
BFDD = ROOT / "01_data" / "raw" / "_public_datasets" / "bfdd"
SRC = BFDD / "_extract" / "Dataset_1x"
WRK = BFDD / "_yolo" / "crackdet"
EVAL = ROOT / "04_results" / "eval"
LOGS = ROOT / "logs"

# V3 的类名与顺序（必须与主力权重完全一致，否则类索引对不上）
V3_NAMES = ["crack", "spalling", "efflorescence", "exposed_rebar",
            "rust", "delamination", "moss"]
CRACK_ID = 0
MIN_AREA = 24          # 连通域最小面积（与 _bfdd_masks_to_yolo 保持一致）

import cv2                      # noqa: E402


def rd(p: Path, f=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), f)


def build_labels(split: str) -> int:
    """把 v=1 的连通域转成 YOLO 检测标注（class 0）。"""
    (WRK / "images" / split).mkdir(parents=True, exist_ok=True)
    (WRK / "labels" / split).mkdir(parents=True, exist_ok=True)
    # BFDD 的划分文件名是 `train.txt` / `test.txt`（不是 YOLO 惯例的 val.txt）
    fname = {"val": "test", "train": "train"}[split]
    stems = [s.strip() for s in (SRC / f"{fname}.txt").read_text().split() if s.strip()]
    nb = 0
    for s in stems:
        lab = rd(SRC / "Label" / f"{s}.png", cv2.IMREAD_UNCHANGED)
        if lab is None:
            continue
        H, W = lab.shape[:2]
        m = (lab == 1).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lines = []
        for c in cnts:
            if cv2.contourArea(c) < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            lines.append(f"{CRACK_ID} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        (WRK / "labels" / split / f"{s}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        dst = WRK / "images" / split / f"{s}.jpg"
        if not dst.exists():
            try:
                os.link(SRC / "RGB" / f"{s}.JPG", dst)
            except OSError:
                import shutil
                shutil.copy2(SRC / "RGB" / f"{s}.JPG", dst)
        nb += len(lines)
    return nb


def metrics_hitrate(model, split: str, imgsz: int, conf: float) -> dict:
    """宽松口径：GT 框被 class-0 预测覆盖（交叠/GT 面积 ≥ 0.1）的比例。"""
    stems = sorted(p.stem for p in (WRK / "images" / split).glob("*.jpg"))
    n_gt = n_hit = 0
    for s in stems:
        img = rd(WRK / "images" / split / f"{s}.jpg")
        H, W = img.shape[:2]
        gts = []
        lp = WRK / "labels" / split / f"{s}.txt"
        for ln in lp.read_text(encoding="utf-8").split("\n"):
            p = ln.split()
            if len(p) != 5:
                continue
            _, cx, cy, bw, bh = p
            cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H
            gts.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        r = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        preds = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for k, c in enumerate(cls):
                if c == CRACK_ID:
                    preds.append(xyxy[k])
        for g in gts:
            n_gt += 1
            ga = max(0.0, (g[2] - g[0])) * max(0.0, (g[3] - g[1]))
            if ga <= 0:
                continue
            for p in preds:
                ix = max(0.0, min(g[2], p[2]) - max(g[0], p[0]))
                iy = max(0.0, min(g[3], p[3]) - max(g[1], p[1]))
                if (ix * iy) / ga >= 0.1:
                    n_hit += 1
                    break
    return {"n_gt_boxes": n_gt, "n_hit": n_hit,
            "hit_rate": (n_hit / n_gt) if n_gt else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    weights = ROOT / "03_weights" / "v11s640_best.pt"
    assert weights.exists(), f"缺少主力权重 {weights}"

    print("=" * 88)
    print("I1：BFDD 裂缝的零样本跨源评测（**不训练、不改 V3**）")
    print("=" * 88, flush=True)

    nb_val = build_labels("val")
    print(f"BFDD test 152 图 → 裂缝检测框 {nb_val} 个（class 0）", flush=True)

    # 域内基准（权威来源）
    ind = json.loads((EVAL / "v11s640_test_metrics.json").read_text(encoding="utf-8"))
    in_dom = next(p for p in ind["per_class"] if p["class_id"] == CRACK_ID)
    print(f"V3 域内 crack：AP50 = {in_dom['ap50']:.4f}  "
          f"(P={in_dom['precision']:.4f} R={in_dom['recall']:.4f} n={in_dom['instances']})\n",
          flush=True)

    yml = WRK / "bfdd_crack.yaml"
    yml.write_text(
        f"path: {WRK.as_posix()}\n"
        f"train: images/val\n"           # 只为 val() 能跑起来；只用 val
        f"val: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(V3_NAMES)),
        encoding="utf-8")

    from ultralytics import YOLO
    model = YOLO(str(weights))

    print("跑严格口径 val() …", flush=True)
    m = model.val(data=str(yml), split="val", imgsz=args.imgsz, device=args.device,
                  workers=0, plots=False, verbose=False)
    # ★ DetMetrics 没有 .ap50/.p/.r；逐类要用 class_result(i) 或 all_ap + ap_class_index
    ap50 = float(np.asarray(m.box.map50))
    idx = [int(x) for x in np.asarray(m.box.ap_class_index).reshape(-1)]
    all_ap = np.asarray(m.box.all_ap)                      # (n_classes, 10) IoU 0.5:0.05:0.95
    per_class_ap50 = [float(x) for x in all_ap[:, 0]]
    print(f"  全类 mAP50 = {ap50:.4f}；参与评测的类索引 = {idx}", flush=True)
    print(f"  逐类 AP50 = {[round(x, 4) for x in per_class_ap50]}", flush=True)
    if CRACK_ID not in idx:
        raise SystemExit(f"!! class {CRACK_ID}(crack) 未参与评测，idx={idx}")
    pos = idx.index(CRACK_ID)
    cross_ap50 = per_class_ap50[pos]
    p_c, r_c, _ap50c, _apc = m.box.class_result(pos)
    print(f"  ⇒ 跨源 crack：AP50 = {cross_ap50:.4f}  P = {p_c:.4f}  R = {r_c:.4f}",
          flush=True)

    print("\n跑宽松口径（命中率）…", flush=True)
    hit = metrics_hitrate(model, "val", args.imgsz, args.conf)
    print(f"  GT 框 {hit['n_gt_boxes']} 个，命中 {hit['n_hit']} 个 "
          f"⇒ 命中率 {hit['hit_rate']:.1%}", flush=True)

    delta = cross_ap50 - in_dom["ap50"]
    rep = {
        "task": "BFDD crack 零样本跨源评测（不训练、不改 V3）",
        "weights": str(weights),
        "imgsz": args.imgsz, "conf": args.conf,
        "n_bfdd_val_images": 152, "n_bfdd_crack_boxes": nb_val,
        "in_domain": {"source": "04_results/eval/v11s640_test_metrics.json",
                      "class": "crack", "ap50": in_dom["ap50"],
                      "precision": in_dom["precision"], "recall": in_dom["recall"],
                      "instances": in_dom["instances"], "images": in_dom["images"]},
        "cross_domain": {"class": "crack (BFDD v1)", "ap50": cross_ap50,
                         "precision": p_c, "recall": r_c,
                         "hit_rate": hit["hit_rate"], "n_gt_boxes": hit["n_gt_boxes"],
                         "n_hit": hit["n_hit"]},
        "delta_ap50": delta,
        "per_class_ap50_on_bfdd_gt": {V3_NAMES[i]: round(v, 6)
                                      for i, v in enumerate(per_class_ap50)},
        "caveats": [
            "BFDD 是**语义分割**掩膜转检测框，与 V3 的检测框标注**语义不同**"
            "（细长裂缝网的外接框偏大）⇒ 除严格 AP50 外另报免 IoU 阈值的命中率",
            "BFDD 为**他人公开数据**（CC BY 4.0，东南大学），采集为无人机单次航测战役（4 天）",
            "本项目**无真实无人机数据**；本项是**零样本跨源评测**，不是无人机实测",
            "只评测，**未训练、未微调**，V3 的已上报数字未受影响",
        ],
    }
    outp = EVAL / "bfdd_crossdomain_crack.json"
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print("=" * 88)
    print(f"V3 域内 crack AP50 = {in_dom['ap50']:.4f}")
    print(f"BFDD 跨源 crack AP50 = {cross_ap50:.4f}   Δ = {delta:+.4f} "
          f"({delta / in_dom['ap50'] * 100:+.1f}%)")
    print(f"宽松口径命中率 = {hit['hit_rate']:.1%}（免 IoU 阈值）")
    print(f"产物: {outp}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
