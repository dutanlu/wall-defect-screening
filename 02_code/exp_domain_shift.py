# -*- coding: utf-8 -*-
"""
exp_domain_shift.py —— 距离域偏移实验（核心创新点 1 的定量论据）

===== 实验要回答的问题 =====

公开数据集都是**近距离**采集的（拍砖墙、拍梁柱，几米内），
而本系统的部署场景是**远距离**筛查（地面仰拍，20~50m）。
这中间存在一个**域偏移（domain shift）**：

    训练分布：近距 / 小 GSD（每像素代表很小的一块墙面）
    部署分布：远距 / 大 GSD（每像素代表很大的一块墙面）

如果直接拿近距训练的模型去跑远距照片，性能会衰减多少？
**这个问题在现有外墙缺陷检测研究里几乎没人定量回答。**

===== 实验设计（为什么这样设计）=====

我们没有真实的「同场景不同距离」配对照片，所以用**降采样**做受控模拟：
把测试集图像按不同倍率缩小，再还原到模型输入尺寸 ——
这在光学上等价于「用更低的分辨率/更远的距离拍摄同一场景」。

    原图 GSD = g  →  缩小 k 倍后 GSD = k·g

这样做的优点是**严格受控**：内容完全一致，唯一变量是有效分辨率，
因此性能变化可以直接归因于 GSD，排除了光照/角度/场景差异等混淆因素。
> 局限（必须在报告中声明）：降采样无法模拟真实远距拍摄的
> 大气扰动、运动模糊、透视畸变。因此本实验给出的是**性能衰减的下界**
> （真实衰减会更严重）。诚实声明这一点比夸大结论更有说服力。

===== 关键产出 =====

对每个 GSD 水平，输出三件事：
  1. **检测层面**：mAP50 / mAP50-95 / 各阈值下的召回
  2. **可判读性层面**：判决等级（measurement / screening / infeasible）
  3. **交叉点**：模型性能开始显著劣化的 GSD，是否与 gsd.py 的理论门槛一致？

**第 3 点是本实验最有价值的地方**：它验证「可判读性自检」这个创新机制
不是拍脑袋的经验值，而是**与模型实际性能衰减点吻合的物理判据**。

用法：
  python exp_domain_shift.py --weights=..\\03_weights\\v8s640_best.pt
  python exp_domain_shift.py --weights=... --scales=1,0.75,0.5,0.35,0.25,0.15
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from common import (
    ABLATION_DIR,
    CLASSES,
    CLASS_CN,
    DATASET_DIR,
    DATASET_YAML_NAME,
    argv_flag,
    dump_json,
    ensure_dirs,
    imread_u,
    imwrite_u,
    log,
)
from gsd import MIN_RELIABLE_PX, assess_interpretability, calibrate_by_object

# 参考 GSD：假设原始测试图为「近距离采集」的基准。
# 取 1.0 mm/px 是因为：砖缝周期 250mm 在图中约占 250px，符合数据集实际观感
# （数据集图片多为 400~1200px 宽的近距特写）。这个值只影响绝对刻度，
# 不影响「性能随 GSD 衰减」这一结论的相对趋势。
BASE_GSD_MM_PER_PX = 1.0


def build_degraded_dataset(src_images: Path, dst_images: Path,
                           scale: float) -> int:
    """
    把源图像按 scale 降采样后再放大回原尺寸（模拟远距拍摄的低分辨率）。

    为什么要「降采样再放大」而不是直接缩小：
    模型要求固定输入尺寸。若只缩小，Ultralytics 会在推理时把它放大回去，
    相当于我们无法控制「有效信息量」。先缩小（丢信息）再放大（恢复尺寸），
    这样进入模型的像素数不变，但**承载的真实信息量已经被削减**了，
    精确对应「距离变远 → 单位面积像素变少」。

    取整到偶数尺寸：JPEG/模型对奇数尺寸有时会有 1px 抖动，属无谓噪声。
    """
    dst_images.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in src_images.glob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    n = 0
    for p in imgs:
        img = imread_u(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        if scale >= 0.999:
            out = img
        else:
            sw = max(16, int(round(w * scale)))
            sh = max(16, int(round(h * scale)))
            small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
            # 再放大回原尺寸（用 LINEAR，模拟光学模糊的平滑效果）
            out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        imwrite_u(dst_images / p.name, out)
        n += 1
    return n


def main() -> None:
    weights = argv_flag("weights", "")
    run_name = argv_flag("name", "v8s640")
    scales_s = argv_flag("scales", "1,0.75,0.5,0.35,0.25,0.15")
    imgsz = int(argv_flag("imgsz", "640"))
    device = argv_flag("device", "0")
    out_root = Path(argv_flag("out", str(ABLATION_DIR / "domain_shift")))

    if not weights:
        log("用法: python exp_domain_shift.py --weights=<best.pt> [--name=<run>]")
        return
    w = Path(weights)
    if not w.exists():
        log(f"!! 权重不存在: {w}")
        return

    scales = [float(s) for s in scales_s.split(",")]

    src_test = DATASET_DIR / "images" / "test"
    if not src_test.exists():
        log(f"!! 测试集不存在: {src_test}")
        return
    n_test = len([p for p in src_test.glob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    log(f"测试集: {src_test}  ({n_test} 张)")

    from ultralytics import YOLO
    model = YOLO(str(w))

    ensure_dirs(out_root)
    records = []

    for scale in scales:
        eff_gsd = BASE_GSD_MM_PER_PX / scale          # 有效 GSD：越小越近，越大越远

        # ---- 可判读性判定（理论侧）----
        calib = calibrate_by_object(
            pixel_width=1000.0,                        # 归一化占位，只为构造 Calibration
            real_width_mm=1000.0 * eff_gsd,
            object_name="等效GSD",
        )
        interp_crack = assess_interpretability(calib, 0.30)   # 裂缝 0.3mm
        interp_blob = assess_interpretability(calib, 10.0)    # 1cm 剥落

        tag = f"s{scale:g}".replace(".", "p")
        ds_dir = out_root / f"data_{tag}"
        n_made = build_degraded_dataset(src_test, ds_dir, scale)

        # ---- 推理评估（实测侧）----
        # 直接把退化后的图建成一个临时 YOLO 数据集（复用原 labels）
        lab_src = DATASET_DIR / "labels" / "test"
        lab_dst = out_root / f"labels_{tag}"
        lab_dst.mkdir(parents=True, exist_ok=True)
        for t in lab_src.glob("*.txt"):
            (lab_dst / t.name).write_bytes(t.read_bytes())

        yaml_path = out_root / f"ds_{tag}.yaml"
        yaml_path.write_text(
            "# 由 exp_domain_shift.py 自动生成（距离域偏移实验的临时数据集）\n"
            f"path: {out_root.resolve().as_posix()}\n"
            f"train: data_{tag}\n"          # 占位，val 才是真用的
            f"val: data_{tag}\n"
            f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)),
            encoding="utf-8")

        # labels 需与 images 同级目录名对应：Ultralytics 用 images->labels 替换
        # 因此把 labels 放到 data_{tag} 的同级 labels 位置
        canon_lab = out_root / "labels" / f"data_{tag}"
        canon_lab.mkdir(parents=True, exist_ok=True)
        for t in lab_src.glob("*.txt"):
            (canon_lab / t.name).write_bytes(t.read_bytes())

        # 重写 yaml 指向规范的 images 目录结构
        img_root = out_root / "images"
        img_root.mkdir(parents=True, exist_ok=True)
        split_img = img_root / f"data_{tag}"
        split_img.mkdir(parents=True, exist_ok=True)
        for p in ds_dir.glob("*"):
            if p.is_file():
                (split_img / p.name).write_bytes(p.read_bytes())

        yaml_path.write_text(
            "# 由 exp_domain_shift.py 自动生成（距离域偏移实验）\n"
            f"path: {out_root.resolve().as_posix()}\n"
            f"train: images/data_{tag}\n"
            f"val: images/data_{tag}\n"
            "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)),
            encoding="utf-8")

        log(f"[scale={scale:g}] 有效GSD={eff_gsd:.2f}mm/px  退化图 {n_made} 张  "
            f"裂缝0.3mm={interp_crack.level}  1cm剥落={interp_blob.level}")

        rec = {
            "scale": scale,
            "effective_gsd_mm_per_px": round(eff_gsd, 4),
            "crack_0_3mm_px": round(0.30 / eff_gsd, 3),
            "blob_10mm_px": round(10.0 / eff_gsd, 3),
            "interpretability_crack": interp_crack.level,
            "interpretability_blob": interp_blob.level,
            "n_images": n_made,
            "status": "pending",
        }
        try:
            m = model.val(
                data=str(yaml_path), split="val", imgsz=imgsz, batch=16,
                device=device, workers=0, plots=False, verbose=False,
                conf=0.001, iou=0.6,
            )
            rd = m.results_dict or {}
            rec["metrics"] = {k: round(float(v), 5)
                              for k, v in rd.items() if isinstance(v, (int, float))}
            # 每类 AP，用于看「哪一类先崩」
            per = {}
            try:
                for n, ci in enumerate(m.box.ap_class_index):
                    ci = int(ci)
                    nm = CLASSES[ci] if ci < len(CLASSES) else f"cls{ci}"
                    per[nm] = {
                        "ap50": round(float(m.box.ap50[n]), 4),
                        "ap50_95": round(float(m.box.ap[n]), 4),
                    }
            except Exception:
                pass
            rec["per_class"] = per
            rec["status"] = "ok"
            log(f"    mAP50={rec['metrics'].get('metrics/mAP50(B)')} "
                f"mAP50-95={rec['metrics'].get('metrics/mAP50-95(B)')}")
        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = repr(e)
            log(f"    !! 评估失败: {e!r}")
        records.append(rec)

    # ---- 落盘 ----
    dump_json({
        "weights": str(w),
        "run": run_name,
        "base_gsd_mm_per_px": BASE_GSD_MM_PER_PX,
        "min_reliable_px": MIN_RELIABLE_PX,
        "split": "test",
        "design_note": ("用降采样模拟远距拍摄：内容完全一致，唯一变量是有效分辨率，"
                        "因此性能变化可直接归因于 GSD。局限：无法模拟大气扰动/"
                        "运动模糊/透视畸变，故本结果是衰减下界。"),
        "records": records,
    }, out_root / "domain_shift_result.json")

    # CSV 供报告直接引用
    csv_path = out_root / "domain_shift_table.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["scale", "有效GSD(mm/px)", "0.3mm裂缝占px", "1cm剥落占px",
                     "可判读性(裂缝)", "可判读性(剥落)",
                     "mAP50", "mAP50-95", "P", "R"])
        for r in records:
            m = r.get("metrics") or {}
            wr.writerow([
                f"{r['scale']:g}", r["effective_gsd_mm_per_px"],
                r["crack_0_3mm_px"], r["blob_10mm_px"],
                r["interpretability_crack"], r["interpretability_blob"],
                m.get("metrics/mAP50(B)", ""), m.get("metrics/mAP50-95(B)", ""),
                m.get("metrics/precision(B)", ""), m.get("metrics/recall(B)", ""),
            ])

    # ---- 汇总表 ----
    log("=" * 84)
    log("距离域偏移实验结果（测试集）")
    log("=" * 84)
    log(f"{'scale':>6} {'GSD(mm/px)':>11} {'0.3mm(px)':>10} {'1cm(px)':>8} "
        f"{'裂缝判读':>10} {'mAP50':>8} {'mAP50-95':>9}")
    for r in records:
        m = r.get("metrics") or {}
        log(f"{r['scale']:>6g} {r['effective_gsd_mm_per_px']:>11.2f} "
            f"{r['crack_0_3mm_px']:>10.2f} {r['blob_10mm_px']:>8.1f} "
            f"{r['interpretability_crack']:>10} "
            f"{str(m.get('metrics/mAP50(B)', '-')):>8} "
            f"{str(m.get('metrics/mAP50-95(B)', '-')):>9}")
    log("=" * 84)
    log(f"结果: {out_root}")
    log("要点：找 mAP50 开始显著下降的 GSD，与 gsd.py 的理论门槛对照 —— ")
    log("      两者吻合即证明「可判读性自检」是有物理依据的判据，而非经验拍脑袋。")


if __name__ == "__main__":
    main()
