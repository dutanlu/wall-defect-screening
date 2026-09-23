# -*- coding: utf-8 -*-
"""
exp_robustness.py —— 图像质量鲁棒性实验（支撑「可判读性」主张）

===== 为什么需要这个实验 =====

创新点「可判读性判定」目前只考虑了**几何因素**（GSD → 目标占几个像素）。
但真实巡检照片的失效模式还有第二类：**图像质量退化** ——
  低照度（背阴面、清晨/黄昏拍摄）
  失焦模糊（手持抖动、对焦到墙上错误深度）
  过曝（正午强光下白墙）
  雨天（镜头水膜、整体低对比）

如果不测这三类，评委会问：「你说系统知道自己够不够格，
但你只考虑了距离，光线差的时候分辨率够了不也是白搭？」

本实验的作用：**证明几何合格 ≠ 图像可用**，
从而把「可判读性」从一维（GSD）升级为二维（GSD × 图像质量）。
这是创新点的加固，不是补丁。

===== 实验设计 =====

对测试集做 4 组「退化 + 检测」对照，每组用同一个训练好的模型评估：

  1. 低照度  —— 线性降亮 + 加高斯噪声（模拟高 ISO 噪声）
  2. 失焦模糊 —— 高斯模糊，半径递增
  3. 过曝    —— 线性提亮 + 截断（模拟高光溢出）
  4. 低对比（雨天）—— 压缩动态范围到中间灰区

每组按严重程度分档（轻/中/重），记录 mAP 变化。
产出一张「质量维度 → 精度衰减」的表，作为报告里的第二张实验表。

===== 为什么用「图像变换 + 重新评估」而不是收集真实劣质照片 =====

诚实说明：本机没有条件去拍同一栋楼在 4 种光照下的成组照片。
用可控退化变换的好处是**因果清晰**（唯一变量就是那一个质量参数），
代价是**与真实 degrade 分布有偏差**（真实噪声不是高斯的）。
报告里必须写明这是受控仿真，结论是「趋势性」而非「绝对值」。

用法：
  python exp_robustness.py --weights=04_results/train/v8s640/weights/best.pt
  python exp_robustness.py --weights=... --split=test
  python exp_robustness.py --weights=... --levels=0.25,0.5,1.0

输出：
  04_results/eval/robustness_<name>.json
  04_results/eval/robustness_<name>.csv
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from common import (
    DATASET_DIR,
    DATASET_YAML_NAME,
    DATASET_YAML_NAME as _YAML,
    EVAL_DIR,
    TRAIN_DIR,
    dump_json,
    ensure_dirs,
    imread_u,
    imwrite_u,
    list_images,
    log,
    resolve_weight,
    write_dataset_yaml,
)

IMGSZ = 640


# --------------------------------------------------------------------------
# 参数解析
# --------------------------------------------------------------------------
def _arg(name: str, default: str | None = None) -> str | None:
    prefix = f"--{name}="
    for a in sys.argv[1:]:
        if a.startswith(prefix):
            return a[len(prefix):]
    return default


# --------------------------------------------------------------------------
# 四类退化变换
# --------------------------------------------------------------------------
def degrade_lowlight(img: np.ndarray, level: float) -> np.ndarray:
    """
    低照度：降亮 + 加噪。

    level 越大越严重。亮度降到 1/(1+2*level) 左右，
    噪声 sigma 随 level 线性上升（高 ISO 的物理表现）。
    """
    gain = 1.0 / (1.0 + 2.0 * level)
    out = img.astype(np.float32) * gain
    sigma = 12.0 * level
    if sigma > 0:
        noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = out + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_blur(img: np.ndarray, level: float) -> np.ndarray:
    """失焦模糊：高斯核半径随 level 增大（level=1 约 9px 核）。"""
    k = int(round(9 * level))
    if k < 1:
        return img
    k = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def degrade_overexpose(img: np.ndarray, level: float) -> np.ndarray:
    """过曝：提亮 + 截断，模拟高光溢出把细节压平。"""
    gain = 1.0 + 1.5 * level
    out = img.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_lowcontrast(img: np.ndarray, level: float) -> np.ndarray:
    """
    低对比（模拟雨天镜头水膜 / 雾霾）：
    把动态范围向中灰压缩，并叠一层很淡的雾。
    """
    gray_mid = 128.0
    alpha = 1.0 / (1.0 + 1.5 * level)      # 对比度压缩
    out = (img.astype(np.float32) - gray_mid) * alpha + gray_mid
    # 叠雾：向白色靠拢一点，模拟散射光
    haze = 40.0 * level
    out = out + (255.0 - out) * (haze / 255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


DEGRADATIONS = {
    "lowlight": ("低照度", degrade_lowlight),
    "blur": ("失焦模糊", degrade_blur),
    "overexpose": ("过曝", degrade_overexpose),
    "lowcontrast": ("低对比(雨天)", degrade_lowcontrast),
}


# --------------------------------------------------------------------------
# 构造退化数据集（仅改图，标签不动 —— 因为缺陷位置没变）
# --------------------------------------------------------------------------
def build_degraded_split(src_img_dir: Path, src_lbl_dir: Path,
                         dst_root: Path, fn, level: float) -> int:
    """
    把某个 split 的图像按 fn 退化后写到 dst_root/images/<split>，
    标签原样复制到 dst_root/labels/<split>。

    注意：这里退化的是 **检测输入**，所以评估结果反映的是
    「同一批标注、同一批缺陷，只在图像质量上退化」的精度变化 ——
    因果干净。
    """
    dst_img = dst_root / "images" / src_img_dir.name
    dst_lbl = dst_root / "labels" / src_lbl_dir.name
    ensure_dirs(dst_img, dst_lbl)

    imgs = list_images(src_img_dir)
    n = 0
    for p in imgs:
        img = imread_u(p)
        if img is None:
            continue
        out = fn(img, level)
        if not imwrite_u(dst_img / p.name, out):
            continue
        lp = src_lbl_dir / f"{p.stem}.txt"
        if lp.exists():
            shutil.copy2(lp, dst_lbl / lp.name)
        n += 1
    return n


# --------------------------------------------------------------------------
def main() -> None:
    weights = _arg("weights", str(TRAIN_DIR / "v8s640" / "weights" / "best.pt"))
    split = _arg("split", "test")
    levels_raw = _arg("levels", "0.25,0.5,1.0")
    levels = [float(x) for x in (levels_raw or "0.25,0.5,1.0").split(",") if x.strip()]

    wp = Path(resolve_weight(weights or ""))
    if not wp.exists():
        log(f"!! 权重不存在: {wp}")
        return
    name = wp.parent.parent.name if wp.parent.name == "weights" else wp.stem

    src_img = DATASET_DIR / "images" / split
    src_lbl = DATASET_DIR / "labels" / split
    if not src_img.exists():
        log(f"!! 源 split 不存在: {src_img}")
        return

    from ultralytics import YOLO

    ensure_dirs(EVAL_DIR)
    # 退化数据集放在 ASCII 临时目录（避免 ultralytics 在中文路径上的已知坑）
    work = Path(tempfile.gettempdir()) / "wall_defect_robust"
    ensure_dirs(work)

    rows: list[dict] = []

    # 基线：原图不变
    log("=" * 70)
    log("[baseline] 原始图像（无退化）")
    log("=" * 70)
    model = YOLO(str(wp))
    try:
        m0 = model.val(data=str(DATASET_DIR / _YAML), split=split, imgsz=IMGSZ,
                       batch=16, device="0", workers=0, conf=0.001, iou=0.6,
                       plots=False, verbose=False)
        base = {"deg": "baseline", "deg_cn": "原始(基线)", "level": 0.0,
                "precision": round(float(m0.box.mp), 4),
                "recall": round(float(m0.box.mr), 4),
                "mAP50": round(float(m0.box.map50), 4),
                "mAP50_95": round(float(m0.box.map), 4)}
        rows.append(base)
        log(f"  P={base['precision']} R={base['recall']} "
            f"mAP50={base['mAP50']} mAP50-95={base['mAP50_95']}")
    except Exception as e:
        log(f"  !! 基线评估失败: {e!r}")
        return

    # 逐退化 × 逐档位
    for deg_key, (deg_cn, fn) in DEGRADATIONS.items():
        for lv in levels:
            log("-" * 70)
            log(f"[{deg_cn}] level={lv}")
            sub = work / f"{deg_key}_{lv}"
            if sub.exists():
                shutil.rmtree(sub, ignore_errors=True)
            dst_img_root = sub
            n = build_degraded_split(src_img, src_lbl, dst_img_root, fn, lv)
            if n == 0:
                log("  !! 未生成退化图像，跳过")
                continue

            # 写一个只指向退化图的 yaml。
            #
            # 布局约定（必须与 build_degraded_split 一致）：
            #   <sub>/images/<split>/*.jpg
            #   <sub>/labels/<split>/*.txt
            #
            # ⚠️ 这里踩过一个坑：write_dataset_yaml 的 train/val/test 是
            # **相对 root 的路径**，ultralytics 之后会按
            # `<root>/<该值>` 去找图。若写成 `split`（如 "test"），
            # 它会去找 `<sub>/test` 而不是 `<sub>/images/test`，
            # 直接报 "images not found"。所以必须写 "images/<split>"。
            y = sub / _YAML
            rel = f"images/{split}"
            write_dataset_yaml(y, train=rel, val=rel, test=rel, root=sub)

            try:
                mm = YOLO(str(wp)).val(
                    data=str(y),
                    split="test",
                    imgsz=IMGSZ, batch=16, device="0", workers=0,
                    conf=0.001, iou=0.6, plots=False, verbose=False)
                row = {
                    "deg": deg_key, "deg_cn": deg_cn, "level": lv, "n_images": n,
                    "precision": round(float(mm.box.mp), 4),
                    "recall": round(float(mm.box.mr), 4),
                    "mAP50": round(float(mm.box.map50), 4),
                    "mAP50_95": round(float(mm.box.map), 4),
                    "d_mAP50": round(float(mm.box.map50) - base["mAP50"], 4),
                    "d_mAP50_95": round(float(mm.box.map) - base["mAP50_95"], 4),
                }
                rows.append(row)
                log(f"  P={row['precision']} R={row['recall']} "
                    f"mAP50={row['mAP50']} (Δ{row['d_mAP50']:+}) "
                    f"mAP50-95={row['mAP50_95']} (Δ{row['d_mAP50_95']:+})")
            except Exception as e:
                log(f"  !! 评估失败: {e!r}")

    # 落盘
    out_json = EVAL_DIR / f"robustness_{name}.json"
    dump_json({
        "weights": str(wp),
        "split": split,
        "imgsz": IMGSZ,
        "levels": levels,
        "note": ("受控退化仿真：对同一批标注图像施加单一质量退化后重新评估。"
                 "因果清晰但噪声模型与真实退化有偏差，结论应作趋势性证据引用。"),
        "rows": rows,
    }, out_json)

    out_csv = EVAL_DIR / f"robustness_{name}.csv"
    # ⚠️ 基线行与退化行的字段集**不同**（基线没有 n_images / d_mAP50 / d_mAP50_95），
    # 直接 DictWriter(fieldnames=rows[0].keys()) 会在基线行**碰巧排在第一**时
    # 写退化行报 "dict contains fields not in fieldnames"，然后**丢掉整张表**。
    # 本项目已实测踩到：12 组评估全部成功，仅因最后一步写 CSV 崩了，
    # 结果一个数字都没落盘。故这里取**并集**并显式排序。
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)

    log("=" * 70)
    log(f"结果: {out_json}")
    log(f"      {out_csv}")
    log("=" * 70)
    # 摘要
    for r in rows[1:]:
        log(f"{r['deg_cn']:14s} lv={r['level']:<4} "
            f"mAP50={r['mAP50']:.3f} (Δ{r['d_mAP50']:+.3f})  "
            f"mAP50-95={r['mAP50_95']:.3f} (Δ{r['d_mAP50_95']:+.3f})")


if __name__ == "__main__":
    main()
