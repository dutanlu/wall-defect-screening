# -*- coding: utf-8 -*-
r"""P1 探针：在 BFDD 上按**单一输入模态**训一个多类分割模型，并给出逐类掩膜 IoU。

## 为什么从零训练
本地无预训练 `*-seg.pt`，且本机 `github.com` 不可达（ultralytics 的权重托管在 GitHub Releases）。
对本任务反而更干净：**无 COCO 预训练带来的跨域泄露疑虑**。
三种模态**同架构 / 同超参 / 同 seed / 同样本 / 同 epoch**。

## 为什么自己算 IoU 而不用 `val()` 的 mAP
预登记判据写的是 **IoU**（`logs/_bfdd_hollow_prereg.md`）。改成 mAP 会破坏可证伪性。
故这里按「预测掩膜 vs 真值掩膜」逐类算并集交并比；`val()` 的 seg 指标另存作旁证。

## 用法
  python _p1_train_probe.py --modality rgb --epochs 80
  python _p1_train_probe.py --modality ir  --epochs 80
  python _p1_train_probe.py --modality irgray --epochs 80
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\pythonstudy 备份\创新题\外墙缺陷筛查")
BFDD = ROOT / "01_data" / "raw" / "_public_datasets" / "bfdd"
YOLO_ROOT = BFDD / "_yolo"
RUN_ROOT = BFDD / "_runs"
SRC = BFDD / "_extract" / "Dataset_1x"
EVAL = ROOT / "04_results" / "eval"
LOGS = ROOT / "logs"

sys.path.insert(0, str(ROOT / "02_code"))
import cv2                      # noqa: E402

MODS = {"rgb": "jpg", "ir": "png", "irgray": "png", "fuse": "png", "ctrl": "png"}


def read_img(p: Path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), flags)


def per_class_iou(model, modality: str, split: str = "val", conf: float = 0.25,
                  imgsz: int = 512) -> dict:
    """逐类掩膜 IoU：把该类所有实例的预测掩膜并起来，与真值掩膜比并集/交集。"""
    ext = MODS[modality]
    img_dir = YOLO_ROOT / modality / "images" / split
    stems = sorted(p.stem for p in img_dir.glob(f"*.{ext}"))
    inter = np.zeros(6, dtype=np.int64)     # 索引 1..5 用
    union = np.zeros(6, dtype=np.int64)
    n_pred = np.zeros(6, dtype=np.int64)
    n_gt = np.zeros(6, dtype=np.int64)

    for i, s in enumerate(stems, 1):
        img = read_img(img_dir / f"{s}.{ext}")
        lab = read_img(SRC / "Label" / f"{s}.png", cv2.IMREAD_UNCHANGED)
        if img is None or lab is None:
            continue
        h, w = lab.shape[:2]
        r = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]

        pred_union = {v: np.zeros((h, w), dtype=bool) for v in range(1, 6)}
        if r.masks is not None and len(r.masks) > 0:
            ms = r.masks.data.cpu().numpy()
            cs = r.boxes.cls.cpu().numpy().astype(int)
            for k, c in enumerate(cs):
                v = c + 1
                if not (1 <= v <= 5):
                    continue
                m = cv2.resize(ms[k].astype(np.float32), (w, h),
                               interpolation=cv2.INTER_NEAREST) > 0.5
                pred_union[v] |= m
                n_pred[v] += 1
        for v in range(1, 6):
            gt = (lab == v)
            pd = pred_union[v]
            inter[v] += int(np.count_nonzero(gt & pd))
            union[v] += int(np.count_nonzero(gt | pd))
            n_gt[v] += int(np.count_nonzero(gt))
        if i % 40 == 0:
            print(f"    [iou {i}/{len(stems)}]", flush=True)

    out = {}
    for v in range(1, 6):
        out[f"v{v}"] = {
            "iou": (float(inter[v]) / float(union[v])) if union[v] > 0 else None,
            "gt_px": int(n_gt[v]), "pred_boxes": int(n_pred[v]),
        }
    return {"n_images": len(stems), "per_class": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=list(MODS))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0,
                    help="数据加载进程数。本项目纪律：workers 是**概率性静默失败源**"
                         "（见记忆 §1.7），开启后必须读 args.yaml 核实三个值，"
                         "并确认 results.csv 在增长。默认 0（稳，但慢）。")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--name", default="", help="输出名后缀（默认用模态名）")
    ap.add_argument("--weights", default="",
                    help="初始化权重路径。空 = 从零训 yolo11n-seg.yaml。\n"
                         "★ 实测：**从零训 100 轮远不够**（v1 裂缝预测框 0~1 个、mAP50(M) 仅 0.009）；\n"
                         "  本机 github.com 不可达，但 **hf-mirror.com 可下到官方预训练 seg 权重**\n"
                         "  （`Ultralytics/YOLO11/resolve/main/yolo11n-seg.pt`，6.18 MB）。\n"
                         "  三模态用**同一份初始化** ⇒ 对比仍然公平。")
    ap.add_argument("--tag", default="", help="产物文件名后缀（用于区分不同初始化）")
    args = ap.parse_args()

    name = args.name or args.modality
    data_yaml = YOLO_ROOT / f"{args.modality}.yaml"
    assert data_yaml.exists(), f"缺少数据集 yaml: {data_yaml}"
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    t0 = time.perf_counter()
    # ★ 先定 init_desc，再打印 —— 第一版把打印写在定义之前，直接 NameError
    if args.weights:
        init_desc = f'预训练初始化：{args.weights}'
        model = YOLO(args.weights)
    else:
        init_desc = '从零训练：yolo11n-seg.yaml（不加载预训练权重）'
        model = YOLO('yolo11n-seg.yaml')

    print("=" * 78)
    print(f"P1 探针训练 | 模态={args.modality} | epochs={args.epochs} | "
          f"imgsz={args.imgsz} | batch={args.batch} | seed={args.seed}")
    print(f"初始化: {init_desc}", flush=True)
    print("=" * 78, flush=True)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,                 # 本项目纪律：workers 是概率性静默失败源
        seed=args.seed,
        deterministic=True,
        project=str(RUN_ROOT),
        name=name,
        exist_ok=True,
        plots=False,
        verbose=True,
    )
    train_sec = time.perf_counter() - t0

    run_dir = RUN_ROOT / name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    # ★ 本项目纪律：**训练是静默失败**，判据看落盘，不看退出码
    assert (run_dir / "args.yaml").exists(), f"缺少 args.yaml ⇒ 训练未真正开始：{run_dir}"
    assert (run_dir / "results.csv").exists(), f"缺少 results.csv ⇒ 训练未产出：{run_dir}"
    assert best.exists() or last.exists(), f"缺少权重 ⇒ 训练未完成：{run_dir}"
    w = best if best.exists() else last
    print(f"\n训练完成，用时 {train_sec/60:.1f} min，权重 {w}")

    model2 = YOLO(str(w))
    iou = per_class_iou(model2, args.modality, "val", conf=args.conf, imgsz=args.imgsz)

    # 旁证：ultralytics 官方 val 指标
    try:
        m = model2.val(data=str(data_yaml), split="val", imgsz=args.imgsz,
                       device=args.device, workers=0, plots=False, verbose=False)
        segmaps = getattr(getattr(m, "seg", None), "maps", None)
        val_seg_maps = ([float(x) for x in np.asarray(segmaps).reshape(-1)]
                        if segmaps is not None else None)
    except Exception as e:
        print(f"  [warn] val() 失败（仅作旁证，不影响判据）：{type(e).__name__}: {e}")
        val_seg_maps = None

    rep = {
        "modality": args.modality,
        "init": init_desc,
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "seed": args.seed, "conf": args.conf,
        "train_minutes": round(train_sec / 60, 2),
        "weights": str(w),
        "run_dir": str(run_dir),
        "iou": iou,
        "val_seg_maps_per_class": val_seg_maps,
        "note": ("yolo11n-seg 从零训练；IoU 由本项目自算（预测掩膜并集 vs 真值掩膜），"
                 "因为预登记判据写的是 IoU 而非 mAP。val_seg_maps 仅作旁证。"),
    }
    outp = EVAL / f"bfdd_probe_{args.modality}{args.tag}.json"
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n== 逐类掩膜 IoU ==")
    print(f"{'值':>4}{'IoU':>10}{'真值像素':>12}{'预测框数':>10}")
    for v in range(1, 6):
        d = iou["per_class"][f"v{v}"]
        iou_s = "None" if d["iou"] is None else f"{d['iou']:.4f}"
        print(f"{v:>4}{iou_s:>10}{d['gt_px']:>12,}{d['pred_boxes']:>10,}")
    print(f"\n产物: {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
