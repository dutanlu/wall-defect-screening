# -*- coding: utf-8 -*-
"""
_d_train_downsample.py —— D 路【实验组 B】训练入口（rust 降采样）

===== 为什么独立于 train.py =====
train.py 的 CONFIGS 是「正式实验登记表」，D 路是**诊断性实验**（可能得负结论），
塞进去会污染那张表。故本脚本独立，但**超参严格复制 COMMON_ARGS**
（epochs=100 / patience=25 / lr0=0.01 / mosaic=0.5 / seed=42 / deterministic=True …），
保证与基线 v11s640 的**唯一变量 = 训练数据分布**。

⚠️ 与基线的已知差异（必须如实记录，不能假装没有）：
  · 基线 v11s640 的训练集 = V3 全量 1995 图；
    B1/B2 的训练集 = 降采样后 1425 / 1248 图 ⇒ **训练样本量本身也变小了**，
    因此「B 组变差」不能单独归因于「rust 减少」，也可能是「总量减少」。
    ⇒ 这是本实验的**固有混杂**，分析时必须写成限制条件。
    （若要完全解耦需额外做「随机删同量非-rust 图」的对照组，成本×2；
      本脚本用 B1/B2 两档**强度趋势**来部分缓解：若 AP 随 rust 减少单调恶化，
      则「样本量」与「rust 占比」两种解释都能成立；若呈非单调/反直觉，则信息量更大。）

用法（**必须分块跑，本机约 28 分钟会被杀**）：
  python _d_train_downsample.py --cfg=B1_rust3000
  python _d_train_downsample.py --cfg=B2_rust1000
  python _d_train_downsample.py --cfg=B1_rust3000 --resume   # 续训

落盘位置：04_results/train/d_<cfg>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "02_code"
DOWN = ROOT / "01_data" / "_d_downsample"
TRAIN_OUT = ROOT / "04_results" / "train"
WEIGHTS = ROOT / "03_weights"

# 复制自 train.py 的 COMMON_ARGS（唯一变量 = 数据分布，故全部保持一致）
BASE_ARGS = dict(
    epochs=100,
    patience=25,
    optimizer="auto",
    lr0=0.01,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=3.0,
    mosaic=0.5,
    close_mosaic=10,
    fliplr=0.5,
    flipud=0.0,
    degrees=5.0,
    scale=0.5,
    translate=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    copy_paste=0.0,
    # ★ 本机内存纪律：workers=0（基线 args.yaml 也是 0 ⇒ 同时消除 A/B 变量差异）
    workers=0,
    val=True,
    plots=True,
    save=True,
    exist_ok=True,
    verbose=True,
    seed=42,
    deterministic=True,
    imgsz=640,
    batch=24,
    device=0,
    project=str(TRAIN_OUT),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, choices=["B1_rust3000", "B2_rust1000"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 epochs（默认 100）")
    args = ap.parse_args()

    yaml_path = DOWN / args.cfg / f"{args.cfg}.yaml"
    if not yaml_path.exists():
        print(f"[ERR] 数据集 yaml 不存在: {yaml_path}\n      先跑 _d_build_downsample.py")
        return 2

    name = f"d_{args.cfg}"
    run_dir = TRAIN_OUT / name

    from ultralytics import YOLO

    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            print(f"[ERR] 找不到续训点: {last}")
            return 2
        print(f"[{name}] --resume 从 {last}")
        model = YOLO(str(last))
        model.train(resume=True)
        return 0

    if run_dir.exists() and any(run_dir.glob("weights/*.pt")):
        print(f"[{name}] ⚠️ 已存在训练产物（{run_dir}）。")
        print("         续训请加 --resume；重跑请先改名该目录（勿直接删，保留证据）。")
        return 3

    kw = dict(BASE_ARGS)
    if args.epochs is not None:
        kw["epochs"] = args.epochs
    kw["data"] = str(yaml_path)
    kw["name"] = name
    kw["model"] = str(WEIGHTS / "yolo11s.pt")

    print("=" * 74)
    print(f"[{name}] D 路实验组 B —— rust 降采样")
    print(f"  data   = {kw['data']}")
    print(f"  epochs = {kw['epochs']}  imgsz={kw['imgsz']}  batch={kw['batch']}"
          f"  workers={kw['workers']}  seed={kw['seed']}")
    print(f"  out    = {run_dir}")
    print("=" * 74)

    model = YOLO(kw.pop("model"))
    model.train(**kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
