# -*- coding: utf-8 -*-
r"""在 `fuse` 之外再构造 **对照模态 `ctrl`**，把「第三个通道是灰度图」这个结构固定住。

## 为什么需要这个对照
`fuse = [R, G, IR_gray]` 的结果**全部 ≤ RGB**，但它**不可解释**：
第三个通道被换掉了，损失可能来自「丢掉 B」而不是「IR 没用」。

**对照设计**：`ctrl = [R, G, gray(RGB)]`
  · **通道结构与 fuse 完全相同**（第 3 通道都是一张灰度图）；
  · **唯一差别是灰度的来源**（RGB 亮度 vs 红外）。
⇒ `fuse − ctrl` 就隔离出「**红外灰度相对可见光灰度的信息增益**」。

## 判读规则（**事先声明**）
  · `fuse > ctrl` ⇒ 红外灰度比可见光灰度更有用 ⇒ 支持「IR 携带 RGB 没有的信息」
  · `fuse ≈ ctrl`（|Δ| 小于两类间噪声量级）⇒ 无证据表明 IR 有独有信息
  · `fuse < ctrl` ⇒ 红外灰度不如可见光灰度
⚠️ 本对照仍**不能**直接回答「IR 能否看见空鼓」（标签仍由可见光判读所定义），
   它回答的是「**IR 通道是否携带独立信息**」这一前置问题。
"""
from __future__ import annotations

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


def build(name: str, third_of) -> None:
    """third_of(bgr) -> uint8 单通道灰度，放到 B 通道。"""
    for split in ("train", "val"):
        (YOLO / name / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO / name / "labels" / split).mkdir(parents=True, exist_ok=True)
        stems = sorted(p.stem for p in (YOLO / "rgb" / "images" / split).glob("*.jpg"))
        for s in stems:
            dst = YOLO / name / "images" / split / f"{s}.png"
            if not dst.exists():
                bgr = rd(SRC / "RGB" / f"{s}.JPG")
                if bgr is None:
                    continue
                out = bgr.copy()
                out[:, :, 0] = third_of(bgr)
                cv2.imencode(".png", out)[1].tofile(str(dst))
            lsrc = YOLO / "rgb" / "labels" / split / f"{s}.txt"
            ldst = YOLO / name / "labels" / split / f"{s}.txt"
            if not ldst.exists():
                try:
                    os.link(lsrc, ldst)
                except OSError:
                    ldst.write_bytes(lsrc.read_bytes())
        print(f"  {name}/{split}: {len(stems)} 张")
    (YOLO / f"{name}.yaml").write_text(
        f"path: {(YOLO / name).as_posix()}\n"
        f"train: images/train\nval: images/val\n"
        f"names:\n  0: cls1\n  1: cls2\n  2: cls3\n  3: cls4\n  4: cls5\n",
        encoding="utf-8")


def gray_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def main() -> int:
    print("构造 ctrl = [R, G, gray(RGB)]（通道结构与 fuse 相同，仅灰度来源不同）")
    build("ctrl", gray_rgb)
    print(f"\n写出 {YOLO / 'ctrl.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
