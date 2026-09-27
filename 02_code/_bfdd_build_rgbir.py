# -*- coding: utf-8 -*-
r"""构造 **真 4 通道** 数据集 `rgbir = [R, G, B, IR_gray]`（落盘为 4 通道 PNG）。

## 与既有 `fuse` / `ctrl` 的关系（为什么必须再做一个）
| 数据集 | 通道 | 损失 | 能回答什么 |
|---|---|---|---|
| `rgb`    | R,G,B       | —            | 基线 |
| `fuse`   | R,G,**IR**  | **丢掉 B**   | IR 换掉 B 后是否更好（**有混淆**） |
| `ctrl`   | R,G,**gray**| **丢掉 B**   | 与 fuse 同结构、只换灰度来源 ⇒ 隔离 IR 增益 |
| **`rgbir`** | R,G,B,**IR** | **无损失** | ★ **保留全部 RGB + 加 IR** ⇒ 无混淆 |

`fuse − ctrl` 只能回答「IR 灰度是否优于可见光灰度」，仍**不能**回答
「在保留全部 RGB 的前提下，外加一路 IR 是否有增益」——
因为两者都丢掉了 B。`rgbir` 正是为回答后者而生，也是
`logs/_bfdd_modality_compare.txt` 自己写下的「必须做但预算内未解决」的那一项。

## 落盘格式
4 通道 PNG（RGBA，A 通道承载 IR）。**必须用 PNG**：JPEG 不支持 4 通道。
`cv2.imencode('.png', img4)` 可写 4 通道；OpenCV 的 `cvtColor` 不参与本流程，
避免 4 通道被误转。

⚠️ **本脚本只造数据，不改模型、不改 ultralytics**。
   通道读取与首层权重的处理在 `_bfdd_rgbir_patch.py`（运行期 patch）。
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
NAME = "rgbir"

import cv2                      # noqa: E402


def rd(p: Path, f=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), f)


def main() -> int:
    stats = {}
    for split in ("train", "val"):
        (YOLO / NAME / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO / NAME / "labels" / split).mkdir(parents=True, exist_ok=True)
        stems = sorted(p.stem for p in (YOLO / "rgb" / "images" / split).glob("*.jpg"))
        n_new = n_skip = n_link = n_copy = 0
        for s in stems:
            dst = YOLO / NAME / "images" / split / f"{s}.png"
            if not dst.exists():
                bgr = rd(SRC / "RGB" / f"{s}.JPG")          # HxWx3 BGR
                ir = rd(SRC / "IR" / f"{s}.png", cv2.IMREAD_GRAYSCALE)
                if bgr is None or ir is None:
                    continue
                if ir.shape[:2] != bgr.shape[:2]:
                    ir = cv2.resize(ir, (bgr.shape[1], bgr.shape[0]))
                # 通道顺序：YOLO/OpenCV 用 BGR ⇒ 存 [B,G,R,IR]
                # 这样 _patch 里转 RGB 时前 3 通道语义与原生一致，
                # 第 4 通道固定为 IR。绝不假设读者会重排。
                out = np.dstack([bgr, ir[..., None]]).astype(np.uint8)
                assert out.shape[2] == 4, out.shape
                ok, buf = cv2.imencode(".png", out)
                if not ok:
                    raise RuntimeError(f"PNG 编码失败：{s}")
                buf.tofile(str(dst))
                n_new += 1
            else:
                n_skip += 1

            lsrc = YOLO / "rgb" / "labels" / split / f"{s}.txt"
            ldst = YOLO / NAME / "labels" / split / f"{s}.txt"
            if not ldst.exists():
                try:
                    os.link(lsrc, ldst)      # 硬链接省空间；只用不改
                    n_link += 1
                except OSError:
                    ldst.write_bytes(lsrc.read_bytes())
                    n_copy += 1
        stats[split] = dict(stems=len(stems), new=n_new, skip=n_skip,
                            label_link=n_link, label_copy=n_copy)
        print(f"  {split}: 共 {len(stems)} 张  新建 {n_new}  已存在 {n_skip}  "
              f"标签硬链接 {n_link} / 复制 {n_copy}")

    # 写出 yaml（channels: 4 是本方案的开关；ultralytics 原生不认这个键，
    # 由 _bfdd_rgbir_patch.py 读取 —— 原生会忽略未知键，安全）
    yml = YOLO / f"{NAME}.yaml"
    yml.write_text(
        f"path: {(YOLO / NAME).as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"channels: 4\n"
        f"names:\n  0: cls1\n  1: cls2\n  2: cls3\n  3: cls4\n  4: cls5\n",
        encoding="utf-8")
    print(f"\n写出 {yml}")

    # 自检：抽样读回，确认真的是 4 通道且第 4 通道 == IR
    n_ok = n_bad = 0
    for p in sorted((YOLO / NAME / "images" / "val").glob("*.png"))[:20]:
        im = rd(p, cv2.IMREAD_UNCHANGED)
        ir_ref = rd(SRC / "IR" / f"{p.stem}.png", cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape[2] != 4:
            n_bad += 1
            continue
        if ir_ref is not None and np.array_equal(im[:, :, 3], ir_ref):
            n_ok += 1
        else:
            n_bad += 1
    print(f"自检：抽样 20 张，第 4 通道 == IR 灰度 {n_ok}/20 通过"
          f"{'' if n_bad == 0 else f'（{n_bad} 张异常）'}")

    out = BFDD / "_rgbir_build.json"
    out.write_text(json.dumps(
        {"name": NAME, "channel_order": "BGRA = [B, G, R, IR_gray]",
         "ir_source": "IR/*.png -> cv2.IMREAD_GRAYSCALE",
         "rgb_source": "RGB/*.JPG",
         "splits": stats, "selfcheck_pass": n_ok, "selfcheck_total": 20,
         "note": "真 4 通道；与 rgb/fuse/ctrl 共享标签（硬链接）与同一划分。"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {out}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
