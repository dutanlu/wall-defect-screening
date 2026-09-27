# -*- coding: utf-8 -*-
r"""4 通道（RGB+IR）探针：训一个 `[B,G,R,IR]` 输入的 5 类分割模型，与既有模态同口径对比。

## 与 `_p1_train_probe.py` 的关系
沿用**完全相同的判据与超参**（同 imgsz=512 / batch=8 / seed=0 / epochs=80 /
同预训练来源 `logs/_yolo11n-seg.pt`），只换输入通道数 ⇒ 单变量。
IoU 也沿用同一套自算实现（预测掩膜并集 vs 真值掩膜），不换成 mAP。

## 与既有四个模态的关键差别
`fuse`/`ctrl` 都是**丢掉 B 通道**换上一路灰度；本探针是**保留 R,G,B 再加 IR**，
所以它回答的是那个此前**问不出口**的问题：
  「**在完整保留 RGB 的前提下，外加一路红外是否有增益？**」

## 判读规则（**跑之前声明**，与既有对比保持一致）
· `rgbir > rgb`  ⇒ 保留 RGB 时外加 IR **仍有增益** ⇒ IR 携带独立信息（强证据）
· `rgbir ≈ rgb`  ⇒ 无证据表明 IR 有增益
· `rgbir < rgb`  ⇒ 外加 IR **反而有害**（可能是 4 通道带来的优化困难，不是 IR 无用）
⚠️ 仍**不能**据此证明「IR 能看见空鼓」：BFDD 掩膜由可见光判读定义，
   本探针只回答「IR 通道是否携带独立信息」这一**必要非充分**条件。

## 用法
  python _bfdd_rgbir_probe.py                      # 默认 80 轮，膨胀模式 mean
  python _bfdd_rgbir_probe.py --inflate zero       # 换初始化做敏感性
  python _bfdd_rgbir_probe.py --epochs 4 --name rgbir_smoke   # 冒烟
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
import cv2                              # noqa: E402
import _bfdd_rgbir_patch as PATCH       # noqa: E402

MODALITY = "rgbir"
EXT = "png"


def read_img(p: Path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), flags)


def build_inflated_weights(src_pt: Path, dst_pt: Path, mode: str, nc: int = 5) -> dict:
    """读 3 通道预训练 ckpt → 膨胀首层到 4 通道 → 另存一个新 ckpt 供 YOLO() 加载。

    为什么必须另存而不是直接改 :class:`YOLO` 的内部模型：
    `YOLO(pt).train()` 会从 ckpt 重建模型，运行期对 `model.model` 的修改会被覆盖。
    ⇒ 必须在 **ckpt 层面**膨胀，让加载者拿到的就是 4 通道模型。
    """
    import torch

    sd = torch.load(str(src_pt), map_location="cpu", weights_only=False)
    model = sd.get("model")
    assert model is not None, f"ckpt 里没有 'model'：{src_pt}"
    old_shape = tuple(model.model[0].conv.weight.shape)
    info = PATCH.inflate_first_conv(model, mode=mode, verbose=True)

    # EMA 也要同步膨胀，否则训练开始时 EMA/模型通道数不一致
    ema = sd.get("ema")
    if ema is not None and hasattr(ema, "model"):
        try:
            PATCH.inflate_first_conv(ema, mode=mode, verbose=False)
        except Exception as e:
            print(f"  [warn] EMA 膨胀失败（{type(e).__name__}），置空以免不一致：{e}")
            sd["ema"] = None

    sd["model"] = model
    torch.save(sd, str(dst_pt))
    info["src_shape"] = old_shape
    info["dst_shape"] = tuple(model.model[0].conv.weight.shape)
    return info


def per_class_iou(model, split: str = "val", conf: float = 0.25,
                  imgsz: int = 512) -> dict:
    """与 `_p1_train_probe.per_class_iou` **同一实现**（4 通道读取）。"""
    img_dir = YOLO_ROOT / MODALITY / "images" / split
    stems = sorted(p.stem for p in img_dir.glob(f"*.{EXT}"))
    inter = np.zeros(6, dtype=np.int64)
    union = np.zeros(6, dtype=np.int64)
    n_pred = np.zeros(6, dtype=np.int64)
    n_gt = np.zeros(6, dtype=np.int64)

    for i, s in enumerate(stems, 1):
        # ★ 关键：4 通道必须 IMREAD_UNCHANGED，否则被截断成 3 通道、实验变伪
        img = read_img(img_dir / f"{s}.{EXT}", cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            raise AssertionError(f"读到的不是 4 通道：{s} shape="
                                 f"{None if img is None else img.shape}")
        lab = read_img(SRC / "Label" / f"{s}.png", cv2.IMREAD_UNCHANGED)
        if lab is None:
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
    return {"n_images": len(stems), "channel_read_ok": 4, "per_class": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4,
                    help="沿用既有模态的 workers=4（保持单变量）；"
                         "★ workers 是概率性静默失败源，启动后必须核 args.yaml")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--inflate", default="mean", choices=["mean", "zero", "repeat"])
    ap.add_argument("--name", default="rgbir")
    ap.add_argument("--src_weights", default=str(LOGS / "_yolo11n-seg.pt"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    data_yaml = YOLO_ROOT / f"{MODALITY}.yaml"
    assert data_yaml.exists(), f"缺少数据集 yaml: {data_yaml}（先跑 _bfdd_build_rgbir.py）"
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)

    # 1) 安装 4 通道 patch（必须在 YOLO 读取数据前）
    PATCH.install(verbose=True)

    # 2) ckpt 层面膨胀首层
    dst_pt = LOGS / f"_rgbir_{args.inflate}_init.pt"
    infl = build_inflated_weights(Path(args.src_weights), dst_pt, args.inflate)

    from ultralytics import YOLO
    import torch

    print("=" * 78)
    print(f"4 通道探针 | modality={MODALITY} | epochs={args.epochs} | "
          f"imgsz={args.imgsz} | batch={args.batch} | seed={args.seed} | "
          f"workers={args.workers}")
    print(f"初始化: {args.src_weights} → 首层膨胀 {infl['src_shape']} → {infl['dst_shape']} "
          f"(mode={args.inflate})")
    print("=" * 78, flush=True)

    t0 = time.perf_counter()
    model = YOLO(str(dst_pt))
    # 3) 训练前自检：数据/模型都真是 4 通道
    PATCH.assert_ready(model.model, data_yaml, verbose=True)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, seed=args.seed,
        deterministic=True, project=str(RUN_ROOT), name=args.name,
        exist_ok=True, plots=False, verbose=True,
    )
    train_sec = time.perf_counter() - t0

    run_dir = RUN_ROOT / args.name
    best, last = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"
    assert (run_dir / "args.yaml").exists(), f"缺 args.yaml ⇒ 未真正开始：{run_dir}"
    assert (run_dir / "results.csv").exists(), f"缺 results.csv ⇒ 未产出：{run_dir}"
    assert best.exists() or last.exists(), f"缺权重 ⇒ 未完成：{run_dir}"
    w = best if best.exists() else last
    n_ep = sum(1 for _ in open(run_dir / "results.csv", encoding="utf-8")) - 1
    print(f"\n训练完成，用时 {train_sec/60:.1f} min，落盘 {n_ep} 轮，权重 {w}")

    # 4) 用同一 patch 的 predict 路径做逐类 IoU
    model2 = YOLO(str(w))
    got = model2.model.model[0].conv.weight.shape[1]
    assert got == 4, f"重载后首层不是 4 通道（={got}）⇒ 权重膨胀未持久化"
    iou = per_class_iou(model2, "val", conf=args.conf, imgsz=args.imgsz)

    rep = {
        "modality": MODALITY,
        "channel_order": "BGRA = [B, G, R, IR_gray]",
        "init": f"{args.src_weights} + first-conv inflation(mode={args.inflate})",
        "inflate_info": infl,
        "epochs": args.epochs, "epochs_landed": n_ep,
        "imgsz": args.imgsz, "batch": args.batch, "seed": args.seed,
        "workers": args.workers, "conf": args.conf,
        "train_minutes": round(train_sec / 60, 2),
        "weights": str(w), "run_dir": str(run_dir),
        "iou": iou,
        "note": ("真 4 通道 [R,G,B,IR]；首层由 3 通道预训练权重膨胀而来"
                 "（前 3 通道逐元素保留，第 4 通道按 mode 初始化）。"
                 "IoU 与 _p1_train_probe.py 同一实现（预测掩膜并集 vs 真值掩膜）。"),
    }
    outp = EVAL / f"bfdd_probe_{MODALITY}{args.tag}.json"
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n== 逐类掩膜 IoU ==")
    print(f"{'值':>4}{'IoU':>10}{'真值像素':>12}{'预测框数':>10}")
    for v in range(1, 6):
        d = iou["per_class"][f"v{v}"]
        iou_s = "None" if d["iou"] is None else f"{d['iou']:.4f}"
        print(f"{v:>4}{iou_s:>10}{d['gt_px']:>12,}{d['pred_boxes']:>10,}")
    print(f"\n产物: {outp}")

    PATCH.dump_record(LOGS / "_bfdd_rgbir_patch_record.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
