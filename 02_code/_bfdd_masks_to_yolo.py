# -*- coding: utf-8 -*-
r"""BFDD：把像素掩膜转成 YOLO-seg 多边形标签，并搭出三种模态的数据集目录。

## 为什么用「3 个多类模型」而不是「5 值 × 3 模态 = 15 个二分类探针」
原预登记写的是后者。改为前者，理由（已同步进 `logs/_bfdd_hollow_prereg.md`）：
  · 15 次训练成本过高（每次 686 图），且每个二分类只用一个值的前景，样本极不平衡；
  · 一次多类训练即可**同时**得到 5 个值的逐类 IoU，是文献里比较模态的标准协议；
  · **判据逻辑不变**：H1 仍是 `Δ(v) = IoU_IR(v) − IoU_RGB(v)` 的 argmax。

## 三种模态
  rgb    : 原始 RGB（JPG）           —— 「可见光能看见多少」
  ir     : 原始红外（PNG，8-bit 伪彩）—— 「红外能看见多少」
  irgray : 红外先灰度化再回 3 通道     —— H4 用，排除伪彩映射造成的假象

## 空间策略
838 对图像（RGB 各约 300 KB、IR 约 100 KB）**用硬链接**，不复制；
只有 irgray 需要真正生成（约 85 MB）。

## 输出
`01_data/raw/_public_datasets/bfdd/_yolo/<modality>/{images,labels}/{train,val}/`
+ `<modality>.yaml`
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

BFDD = Path(r"D:\pythonstudy 备份\创新题\外墙缺陷筛查\01_data\raw\_public_datasets\bfdd")
SRC = BFDD / "_extract" / "Dataset_1x"
OUT = BFDD / "_yolo"
LOGS = Path(r"D:\pythonstudy 备份\创新题\外墙缺陷筛查\logs")

MIN_COMPONENT_AREA = 24      # 小于此像素数的连通域不生成多边形（噪声/碎片）
APPROX_EPS_RATIO = 0.002     # approxPolyDP 的 epsilon / 周长

import cv2                      # noqa: E402


def read_img(p: Path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), flags)


def link_or_copy(src: Path, dst: Path) -> str:
    """优先硬链接；失败则复制。返回 'link' | 'copy'。"""
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def atomic_write_text(p: Path, text: str) -> None:
    """原子写：先写同目录临时文件再 `os.replace`。

    ★ 为什么不用 `Path.write_text`：第一版在全量重跑时，val 的第一个标签文件上
      抛了 `PermissionError [Errno 13]`（文件当时完全可写，属**瞬时锁**）。
      原子替换既避开"边写边被读"的窗口，也保证不会留下半截文件。
      另：**标签不再跨模态硬链接**（三个模态各写一份，标签很小），
      从根本上消除"对多链接文件做截断写"这个不确定性。
    """
    tmp = p.with_name(p.name + ".tmp~")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def mask_to_lines(lab: np.ndarray) -> tuple[list[str], dict]:
    """把一个多值掩膜转成 YOLO-seg 标签行（每行一个多边形）。"""
    h, w = lab.shape[:2]
    lines: list[str] = []
    stat: dict = {}
    for v in range(1, 6):
        m = (lab == v).astype(np.uint8)
        px = int(m.sum())
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_poly = n_pts = n_skip = 0
        for c in cnts:
            if cv2.contourArea(c) < MIN_COMPONENT_AREA:
                n_skip += 1
                continue
            eps = APPROX_EPS_RATIO * cv2.arcLength(c, True)
            ap = cv2.approxPolyDP(c, max(eps, 0.5), True).reshape(-1, 2)
            if len(ap) < 3:
                n_skip += 1
                continue
            xy = ap.astype(np.float64)
            xy[:, 0] = np.clip(xy[:, 0] / w, 0.0, 1.0)
            xy[:, 1] = np.clip(xy[:, 1] / h, 0.0, 1.0)
            flat = " ".join(f"{a:.6f}" for a in xy.reshape(-1))
            lines.append(f"{v - 1} {flat}")
            n_poly += 1
            n_pts += len(ap)
        stat[v] = {"px": px, "n_poly": n_poly, "n_pts": n_pts, "n_skip": n_skip}
    return lines, stat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    args = ap.parse_args()

    stems_tr = [s.strip() for s in (SRC / "train.txt").read_text(encoding="utf-8").split() if s.strip()]
    stems_te = [s.strip() for s in (SRC / "test.txt").read_text(encoding="utf-8").split() if s.strip()]
    if args.limit:
        stems_tr = stems_tr[:args.limit]
        stems_te = stems_te[:args.limit]
    print(f"train {len(stems_tr)} / val {len(stems_te)}")

    mods = ["rgb", "ir", "irgray"]
    for m in mods:
        for split in ("train", "val"):
            (OUT / m / "images" / split).mkdir(parents=True, exist_ok=True)
            (OUT / m / "labels" / split).mkdir(parents=True, exist_ok=True)

    cls_px = {v: 0 for v in range(1, 6)}
    cls_poly = {v: 0 for v in range(1, 6)}
    modes = {"link": 0, "copy": 0, "exists": 0}

    for split, stems in (("train", stems_tr), ("val", stems_te)):
        for i, s in enumerate(stems, 1):
            lab = read_img(SRC / "Label" / f"{s}.png", cv2.IMREAD_UNCHANGED)
            if lab is None:
                print(f"  !! 掩膜读不到: {s}")
                continue
            lines, st = mask_to_lines(lab)
            for v in range(1, 6):
                cls_px[v] += st[v]["px"]
                cls_poly[v] += st[v]["n_poly"]
            # 三种模态**各写一份标签**（标签很小，避免跨模态硬链接带来的截断写不确定性）
            lbl_text = "\n".join(lines) + ("\n" if lines else "")
            for m in mods:
                atomic_write_text(OUT / m / "labels" / split / f"{s}.txt", lbl_text)

            # 图像
            modes[link_or_copy(SRC / "RGB" / f"{s}.JPG",
                               OUT / "rgb" / "images" / split / f"{s}.jpg")] += 1
            modes[link_or_copy(SRC / "IR" / f"{s}.png",
                               OUT / "ir" / "images" / split / f"{s}.png")] += 1
            # irgray 必须真生成
            gdst = OUT / "irgray" / "images" / split / f"{s}.png"
            if not gdst.exists():
                ir = read_img(SRC / "IR" / f"{s}.png", cv2.IMREAD_COLOR)
                g = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
                cv2.imencode(".png", cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))[1].tofile(str(gdst))
            if i % 100 == 0:
                print(f"  [{split} {i}/{len(stems)}]")

    # 生成 data yaml
    for m in mods:
        (OUT / f"{m}.yaml").write_text(
            f"path: {(OUT / m).as_posix()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n"
            f"  0: cls1\n  1: cls2\n  2: cls3\n  3: cls4\n  4: cls5\n",
            encoding="utf-8")

    rep = {"n_train": len(stems_tr), "n_val": len(stems_te),
           "class_px": cls_px, "class_poly": cls_poly, "imglink": modes,
           "out": str(OUT)}
    (LOGS / "_bfdd_dataset_build.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print("== 每类像素与多边形数（全量）==")
    for v in range(1, 6):
        print(f"  值{v}: 像素 {cls_px[v]:>10,}   多边形 {cls_poly[v]:>6,}")
    print(f"\n图像链接统计: {modes}")
    print(f"输出: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
