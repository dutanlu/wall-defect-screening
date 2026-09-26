# -*- coding: utf-8 -*-
r"""构造 **融合模态** 数据集 `fuse = [R, G, IR_gray]`（BGR 序即 `[IR_gray, G, R]`）。

## 为什么需要它
P1 的预登记判据（`Δ(v)=IoU_IR−IoU_RGB`）有一个**结构性混淆**：
BFDD 的掩膜是**人工标注**的，标注者很可能是在**可见光图**上判读的
⇒ 标签编码的是「可见光可见的证据」，用 IR 去预测它**结构性处于劣势**。
实测 Δ(v) **五个类全为负**（−0.098 ~ −0.320），正是这个混淆的表现。

**互补性检验**对上述混淆免疫：
  · 若 `IoU_fuse > IoU_RGB` ⇒ 在**保留大部分 RGB 信息**的前提下加入 IR **仍有增益**
    ⇒ 说明 IR 提供了 RGB 没有的信息（对那个类而言）。
  · 若 `IoU_fuse ≤ IoU_RGB` ⇒ **不构成结论**（因为换掉了 B 通道，损失可能被掩盖）。

## 通道选择
用 `[R, G, IR_gray]`（把 B 通道换成 IR 灰度）而不是 4 通道：
  4 通道会破坏第一层卷积 ⇒ **无法加载预训练权重**；而三模态用同一预训练初始化是公平性的前提。
B 通道对这类灰白色立面的信息相对冗余，所以"换掉 B"是代价最小的选择。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\pythonstudy 备份\创新题\外墙缺陷筛查")
BFDD = ROOT / "01_data" / "raw" / "_public_datasets" / "bfdd"
SRC = BFDD / "_extract" / "Dataset_1x"
YOLO = BFDD / "_yolo"

import cv2                      # noqa: E402


def rd(p: Path, f=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), f)


def main() -> int:
    ext_img = "jpg"
    for split in ("train", "val"):
        (YOLO / "fuse" / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO / "fuse" / "labels" / split).mkdir(parents=True, exist_ok=True)
        stems = sorted(p.stem for p in (YOLO / "rgb" / "images" / split).glob(f"*.{ext_img}"))
        n = 0
        for s in stems:
            dst = YOLO / "fuse" / "images" / split / f"{s}.png"
            if not dst.exists():
                bgr = rd(SRC / "RGB" / f"{s}.JPG")
                ir = rd(SRC / "IR" / f"{s}.png")
                if bgr is None or ir is None:
                    continue
                if ir.shape[:2] != bgr.shape[:2]:
                    ir = cv2.resize(ir, (bgr.shape[1], bgr.shape[0]))
                g = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
                out = bgr.copy()
                out[:, :, 0] = g                      # B 通道 ← IR 灰度
                cv2.imencode(".png", out)[1].tofile(str(dst))
            # 标签直接复用（硬链接省空间；只用不改，无截断写问题）
            lsrc = YOLO / "rgb" / "labels" / split / f"{s}.txt"
            ldst = YOLO / "fuse" / "labels" / split / f"{s}.txt"
            if not ldst.exists():
                try:
                    os.link(lsrc, ldst)
                except OSError:
                    ldst.write_bytes(lsrc.read_bytes())
            n += 1
        print(f"  {split}: {n} 张")

    (YOLO / "fuse.yaml").write_text(
        f"path: {(YOLO / 'fuse').as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: cls1\n  1: cls2\n  2: cls3\n  3: cls4\n  4: cls5\n",
        encoding="utf-8")
    print(f"\n写出 {YOLO / 'fuse.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
