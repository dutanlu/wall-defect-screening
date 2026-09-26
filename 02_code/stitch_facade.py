# -*- coding: utf-8 -*-
"""
stitch_facade.py —— 整立面拼接（多段照片 → 一张整立面图 + 能力评估）

===== 为什么是独立模块 + CLI，而不是先改界面 =====
「多图上传」若加到 `06_deploy/app.py`，会改变**可见界面** ⇒ 按项目纪律 #14
必须重录「实机运行视频」。为把「能力交付」与「界面变更」解耦，本模块先以
**纯后端 + 命令行**形式交付：

  · 能力已完整、可跑、可验证（CLI 可用）；
  · UI 入口留作明确标注的「已设计、未接入」项；
  · 这样即使 UI 来不及接，整立面测绘能力本身**已经是交付物**。

===== 它做了什么 =====
    多段照片 → stitch_segments（ORB+RANSAC+羽化）→ 整立面图
             → 统一尺度标定（砖缝周期 / 已知标定物）→ 整立面能力评估

===== 与无人机测绘的关系 =====
本模块是「无人机测绘」的**地面版底座**：手机手持分段拍摄与无人机航拍
在「多段 → 拼接 → 统一尺度」这条链上完全一致。
差别只在 **POS 先验**（无人机有、手机没有），设计见
`07_report/无人机POS先验接入设计论证_20260925.md`。

⚠️ 诚实边界：
  · 本模块**不产生**任何航拍相关声明；
  · 拼接结果**不改变**单图测量的毫米数与分级口径；
  · 若某段匹配失败，会**明确报出是哪一段**（不静默跳过）。

用法：
  python stitch_facade.py --dir 无 --images a.jpg b.jpg c.jpg [--out logs/_stitch_demo]
  python stitch_facade.py --images ... --pitch-mm 250 --brick-axis x
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from common import imread_u, imwrite_u, log  # noqa: E402
import rectify as R  # noqa: E402
from gsd import calibrate_by_brick_period, screening_capability  # noqa: E402


def stitch_and_assess(images: list, out_dir: Path,
                      pitch_mm: float = 250.0, axis: str = "x") -> dict:
    """核心：拼接 + 尺度 + 能力评估。返回结果 dict（也用于 UI 复用）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"n_input": len(images)}

    st = R.stitch_segments(images)
    res["stitch"] = st.to_dict()
    log(f"[stitch] ok={st.ok} conf={st.confidence} n_used={st.n_used}/{st.n_input}"
        f" method={st.method}")
    log(f"[stitch] message: {st.message}")
    if st.advice:
        log(f"[stitch] advice : {st.advice}")

    if st.image is None:
        res["ok"] = False
        res["reason"] = "拼接未产出图像"
        return res

    canvas = st.image
    res["canvas_w"] = int(canvas.shape[1])
    res["canvas_h"] = int(canvas.shape[0])
    # ⚠️ 必须用 `imwrite_u`，不能用 `cv2.imwrite`：
    #   本机实测 `cv2.imwrite` 对**含非 ASCII 的路径**返回 False 且**不抛异常**
    #   （本工程路径含中文）⇒ 原写法下整立面图**永远写不出来，而且完全无声**：
    #   因为不抛异常，外层 `except Exception` 形同虚设，连 warn 都不会打。
    #   2026-09-26 修：改用 imwrite_u 并**显式检查返回值**。
    _png = out_dir / "stitched_facade.png"
    if imwrite_u(_png, canvas):
        res["canvas_png"] = str(_png)
    else:
        res["canvas_png"] = None
        log(f"[warn] 整立面图落盘失败（imwrite_u 返回 False）：{_png}")

    # ---- 统一尺度：优先砖缝周期（整立面必然含砖缝）----
    calib = calibrate_by_brick_period(canvas, brick_pitch_mm=pitch_mm, axis=axis)
    # ⚠️ Calibration 的字段是 (mm_per_px, method, confidence, detail)——
    #    不是 source（曾写错，getattr 会静默返回 None）。
    res["calib"] = dict(
        mm_per_px=getattr(calib, "mm_per_px", None),
        method=getattr(calib, "method", None),
        confidence=getattr(calib, "confidence", None),
    )
    if getattr(calib, "mm_per_px", None):
        mmpp = calib.mm_per_px
        res["cover_m"] = round(canvas.shape[1] * mmpp / 1000.0, 3)
        log(f"[scale] {mmpp:.4f} mm/px（{calib.method}）"
            f" ⇒ 整立面覆盖 {res['cover_m']:.2f} m 宽")
        cap = screening_capability(calib)
        res["capability"] = cap
        n_use = sum(1 for r in cap["capability"].values() if r["usable"])
        res["n_usable"] = n_use
        res["n_total"] = len(cap["capability"])
        log(f"[capability] 该配置下 {n_use}/{res['n_total']} 类目标可判读")
    else:
        log("[scale] 未能从砖缝标定尺度（画面可能不含可辨砖缝）"
            " ⇒ 跳过能力评估")

    res["ok"] = True
    (out_dir / "stitch_facade.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True,
                    help="按拍摄顺序给出的分段照片路径")
    ap.add_argument("--out", default=str(CODE.parent / "logs" / "_stitch_demo"))
    ap.add_argument("--pitch-mm", type=float, default=250.0,
                    help="砖缝周期（mm），默认 250（本项目墙面口径）")
    ap.add_argument("--axis", default="x", choices=["x", "y"])
    args = ap.parse_args()

    paths = [Path(p) for p in args.images]
    missing = [p for p in paths if not p.exists()]
    if missing:
        log("!! 以下文件不存在：")
        for p in missing:
            log("   " + str(p))
        return 2
    if len(paths) < 2:
        log("!! 至少需要 2 张分段照片")
        return 2

    imgs = []
    for p in paths:
        im = imread_u(p)
        if im is None:
            log(f"!! 读图失败: {p}")
            return 2
        imgs.append(im)
    log(f"读入 {len(imgs)} 段，尺寸 {[f'{i.shape[1]}x{i.shape[0]}' for i in imgs]}")

    res = stitch_and_assess(imgs, Path(args.out), args.pitch_mm, args.axis)
    (Path(args.out) / "stitch_facade.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    log(f"产物 -> {args.out}")
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
