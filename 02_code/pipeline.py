# -*- coding: utf-8 -*-
"""
pipeline.py —— 端到端推理：图像 → 检测 → 标定 → 可判读性 → 量化 → 分级

这是整个系统的组装点，也是答辩演示的主入口。

完整流程：
  1. YOLO 检测        —— 找出「在哪里、是什么」（bbox + 类别 + 置信度）
  2. 尺度标定          —— 得到 mm/px（标定物 / 砖缝周期 / 相机参数）
  3. 【核心创新】可判读性判定
                      —— 系统自检「这张图够不够格」，不够格就拒绝给定量结论
  4. OpenCV 量化       —— 出毫米级长度/宽度/面积
  5. 规范分级          —— GB 50010 / JGJ 125 → severity
  6. 汇总              —— 房屋危险性等级 + 是否需专业鉴定 + 建议

用法示例：
  # 单张图，用标定物标定（A4 纸在图上占 420px）
  python pipeline.py --image=wall.jpg --model=best.pt --calib-object-px=420

  # 整目录，用相机参数估 GSD（20m 外主摄）
  python pipeline.py --dir=photos/ --model=best.pt --distance=20

  # 用砖缝周期自动标定
  python pipeline.py --image=wall.jpg --model=best.pt --calib-brick

  # 斜拍照片：先做正射校正，再检测与量化
  #   校正成功且「校正后两轴自检」通过时，会自动改用校正给出的
  #   全画面统一尺度 —— 它比相机参数 / 裸砖缝周期更可信（rectify.py §6.5）。
  python pipeline.py --image=wall.jpg --model=best.pt --rectify

  # 强制施加校正（即使判定已接近正对），用于对照实验
  python pipeline.py --image=wall.jpg --model=best.pt --rectify=force
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from common import (
    CLASSES,
    EVAL_DIR,
    RESULT_DIR,
    VIS_DIR,
    argv_flag,
    default_weight,
    dump_json,
    ensure_dirs,
    imread_u,
    imwrite_u,
    list_images,
    log,
    resolve_weight,
)
from gsd import (
    Calibration,
    assess_image_quality,
    assess_interpretability,
    calibrate_by_brick_period,
    calibrate_by_camera,
    calibrate_by_object,
    screening_capability,
)
from grade import (
    LINE_CLASSES,
    apply_abstention,
    building_risk_level,
    classify_unjudgeable,
    grade_all,
)
from measure import draw_measurement, measure_instance
from rectify import rectify as rectify_image


# --------------------------------------------------------------------------
def _flag(name: str) -> bool:
    """
    解析布尔开关，同时支持 `--rectify` 与 `--rectify=1` 两种写法。

    为什么不直接复用 argv_flag：它只匹配 `--key=value`，裸写 `--rectify`
    会被**静默忽略** —— 而这恰恰是最危险的一种失败：使用者以为开了校正，
    实际仍按斜拍原图算毫米数，且报告里看不出来。
    """
    if f"--{name}" in sys.argv[1:]:
        return True
    v = argv_flag(name)
    if v is None:
        return False
    return str(v).strip().lower() not in ("", "0", "false", "no", "off")


def get_calibration(args: dict, image: np.ndarray,
                    rect: dict | None = None) -> Calibration:
    """
    按优先级确定标定：标定物 > 校正后统一尺度 > 砖缝周期 > 相机参数。

    第 2 条 `rectify_metric` 是本次新增的路径：`rectify.py` 只在**校正后
    两轴各自量出的 mm/px 一致**（相对偏差 <= MAX_AXIS_RESIDUAL）时才给出
    `scale_mm_per_px`，所以它是一条「**已验证**的全画面统一尺度」；
    而裸砖缝周期默认假定全画面尺度一致、并未验证，故排在其后。
    它仍弱于「标定物」—— 那是画面里的直接物理参照物。
    """
    if args.get("calib_object_px"):
        c = calibrate_by_object(
            pixel_width=float(args["calib_object_px"]),
            real_width_mm=float(args.get("calib_object_mm", 210.0)),
            object_name=args.get("calib_object_name", "标定物"),
        )
        log(f"标定(标定物)：{c.mm_per_px:.4f} mm/px  [{c.confidence}]")
        return c

    # 正射校正自带的全画面统一尺度（仅「校正后两轴自检」通过时非 None）
    if rect and rect.get("scale_mm_per_px"):
        conf = rect.get("confidence")
        if conf not in ("high", "medium", "low"):
            conf = "low"
        c = Calibration(
            mm_per_px=float(rect["scale_mm_per_px"]),
            method="rectify_metric",
            confidence=conf,
            detail={
                "source": "rectify.py 正射校正（校正后两轴自检通过）",
                "rectify_method": rect.get("method"),
                "uniform_scale": bool(rect.get("uniform_scale")),
                "axis_residual": (rect.get("detail") or {}).get("axis_residual"),
                "note": ("全画面同一尺度，故整张图可用一个 GSD；"
                         "未校正的斜拍图 GSD 随位置变化，不能这样用。"),
            },
        )
        log(f"标定(校正后统一尺度)：{c.mm_per_px:.4f} mm/px  [{c.confidence}]")
        return c

    if args.get("calib_brick"):
        c = calibrate_by_brick_period(
            image,
            brick_pitch_mm=float(args.get("brick_pitch_mm", 250.0)),
        )
        if c is not None:
            log(f"标定(砖缝周期)：{c.mm_per_px:.4f} mm/px "
                f"周期={c.detail['period_px']}px  [{c.confidence}]")
            return c
        log("砖缝周期标定失败（未检出周期结构），回退到相机参数法")

    c = calibrate_by_camera(
        distance_m=float(args.get("distance", 20.0)),
        image_width_px=image.shape[1],
        focal_35mm=float(args.get("focal", 24.0)),
    )
    log(f"标定(相机参数)：{c.mm_per_px:.4f} mm/px  [{c.confidence}]")
    return c


def run_one(image_path: Path, model, args: dict, image=None) -> dict:
    """处理单张图像，返回结构化结果。

    `image` 可选：**已解码的图像数组**；给定时**跳过磁盘读取**。
    加这个形参的原因：`video_screen` 逐帧「写盘 → 读盘」只为了适配本函数
    「路径进」的契约，纯属 I/O 浪费。调用方若已在内存里有帧，可直接传入。
    默认 `None` ⇒ 行为与改动前**逐字节一致**（不影响 `pipeline --dir`
    与 `batch_screen` 两个既有入口）。

    ⚠️ `str(image_path)` 仍会写进结果的 `image` 字段，故调用方应传一个
       有意义的标识路径（视频路径下它是帧的逻辑名，未必真实落盘）。
    """
    img = imread_u(image_path) if image is None else image
    if img is None:
        log(f"  读取失败: {image_path.name}")
        return {"image": str(image_path), "status": "read_failed"}

    h, w = img.shape[:2]
    conf_thr = float(args.get("conf", 0.25))
    imgsz = int(args.get("imgsz", 640))

    # ---- 0) 可选：斜拍正射校正 ----
    # 必须放在检测**之前**：校正改变几何（因而也改变像素尺度）。
    # 若先检测再校正，bbox 会整体错位 —— 这是本模块长期游离在主链路
    # 之外时最容易被忽略的坑。
    rect = None
    if args.get("rectify") or args.get("rectify_force"):
        rres = rectify_image(img, force=bool(args.get("rectify_force")))
        rect = rres.to_dict()
        rect["image_used"] = "rectified" if rres.applied else "original"
        if rres.applied:
            img = rres.image
            h, w = img.shape[:2]
            log(f"  正射校正：{rres.method} / {rres.confidence} —— "
                f"后续检测与量化都在校正后的 {w}x{h} 上做")
        else:
            log(f"  正射校正未施加（confidence={rres.confidence}）：{rres.message}")

    # ---- 1) YOLO 检测 ----
    results = model.predict(img, conf=conf_thr, imgsz=imgsz, verbose=False)
    r = results[0]
    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        clsid = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            name = CLASSES[clsid[i]] if clsid[i] < len(CLASSES) else f"id{clsid[i]}"
            dets.append({"bbox": xyxy[i].tolist(), "cls": name,
                         "conf": float(confs[i])})

    # ---- 2) 标定 ----
    calib = get_calibration(args, img, rect=rect)

    # ---- 3) 可判读性判定（核心创新）----
    # 对「裂纹定级」和「块状筛查」分别判一次，因为二者目标尺寸差两个数量级
    interp_crack = assess_interpretability(calib, 0.30)   # 最严：裂缝宽 0.3mm
    interp_blob = assess_interpretability(calib, 10.0)    # 筛查：1cm 剥落
    capability = screening_capability(calib)

    # ---- 4) 量化 -----
    measurements = []
    for d in dets:
        m = measure_instance(img, d["bbox"], d["cls"], calib)
        if m is not None:
            m.__dict__["conf"] = d["conf"]
            measurements.append(m)

    # ---- 5) 分级 ----
    # JGJ 125 两级危险点判据开关（自检报告 P1-6：此前从未被传入过）。
    # 默认全关 = 只走 GB 50010，与接线前逐字节一致。
    # 用法：--jgj125=main-rebar  /  --jgj125=slab-tension  /  --jgj125=main-rebar,slab-tension
    _jgj = str(args.get("jgj125", "") or "").strip().lower().replace("，", ",")
    _jgj_parts = {t.strip() for t in _jgj.split(",") if t.strip()}
    is_main_rebar_zone = bool(_jgj_parts & {"main-rebar", "rebar", "主筋"})
    is_slab_tension = bool(_jgj_parts & {"slab-tension", "slab", "板受拉"})
    env_class = args.get("env", "二类环境（露天/潮湿）")
    grades = grade_all(measurements, env_class=env_class,
                       is_main_rebar_zone=is_main_rebar_zone,
                       is_slab_tension=is_slab_tension)

    # ---- 5b) 弃权传染到输出层（2026-09-25 修复）----
    # ★★ 这是一个**真实存在的缺陷修复**，不是可选增强：
    #   原先此处直接 `building_risk_level(grades)`，完全不看可判读性
    #   ⇒ 标定兜底给出 GSD≈67mm/px 时，一条裂缝被量成「宽 401.79mm」
    #   ⇒ 判 danger ⇒ 风险报 **C（局部危房）**，而同图 interpretability
    #   明明说 `insufficient`。**自相矛盾的输出比不输出更危险**。
    #   报告 §2.3 把「拒答传染到输出层」列为设计亮点，但该逻辑此前
    #   只存在于 `06_deploy/app.py` 的展示路径 ⇒ 报告所述与 API 行为
    #   不一致。现在把判定抽到 `grade.classify_unjudgeable` /
    #   `grade.apply_abstention` 的**单一共享实现**，两个入口都调它，
    #   保证只有一套口径。
    _quality = assess_image_quality(img)
    _flags = classify_unjudgeable(measurements, interp_crack, interp_blob,
                                  quality=_quality)

    def _advise_for(cls_name: str) -> str:
        """按类给「靠近到多少米才能判读」的具体补拍建议（增强项）。"""
        from advice import advise_distance
        tgt = 0.30 if cls_name in LINE_CLASSES else 10.0
        adv = advise_distance(tgt, image_width_px=int(img.shape[1]),
                              lens_label="主摄 24mm")
        if adv.feasible:
            return (f"【补拍建议】要判 {tgt:g}mm 目标需 GSD ≤ "
                    f"{adv.required_gsd:.3f} mm/px ⇒ 请靠近至 "
                    f"**≤ {adv.max_distance_m:.2f} m**。")
        return (f"【补拍建议】用当前镜头判 {tgt:g}mm 目标即使贴到 "
                f"{adv.max_distance_m*100:.0f}cm 也不够 ⇒ 请改用更长焦段或"
                f"提高画面分辨率。")

    risk, n_unjudgeable = apply_abstention(
        grades, measurements, _flags, calib=calib, quality=_quality,
        advise_fn=_advise_for)

    # ---- 6) 可视化 ----
    vis = img.copy()
    # 7 类调色板（BGR）。颜色选择原则：同类缺陷色相接近（裂缝红、剥落橙、
    # 泛碱蓝、锈迹深橙），便于报告里一眼分辨。
    # ⚠️ 每新增一类必须同步补这里，否则 .get 会静默回退成裂缝红，
    #    导致报告配图把苔藓画成裂缝 —— 6→7 类扩容时已实际漏过一次。
    colors = {
        "crack": (0, 0, 255), "spalling": (0, 140, 255),
        "efflorescence": (180, 120, 0), "exposed_rebar": (255, 0, 180),
        "rust": (0, 100, 255), "delamination": (0, 200, 100),
        "moss": (60, 180, 75),          # 苔藓：绿色系（生物附着）
    }
    for m in measurements:
        sev = "ok"
        for g in grades:
            if tuple(g.bbox_xyxy) == tuple(m.bbox_xyxy):
                sev = g.severity
                break
        col = colors.get(m.cls_name, (0, 0, 255))
        if sev == "danger":
            col = (0, 0, 255)
        vis = draw_measurement(vis, m, col)

    # 角标：把「可判读性」结论直接印在图上，这就是答辩要展示的「系统知道自己几斤几两」
    banner = [
        f"GSD = {calib.mm_per_px:.3f} mm/px ({calib.method})",
        *([f"rectified: {rect['method']} [{rect['confidence']}]"]
          if (rect or {}).get("applied") else []),
        f"crack 0.3mm -> {interp_crack.pixels_on_target:.2f}px  "
        f"[{interp_crack.level.upper()}]",
        f"blob 10mm  -> {interp_blob.pixels_on_target:.2f}px  "
        f"[{interp_blob.level.upper()}]",
        f"risk level: {risk['level']}",
    ]
    for i, t in enumerate(banner):
        y = 22 + i * 22
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA)

    # ★ 2026-09-26：标注图改为**可选落盘**（默认关，用 `--save-annotated` 开启）。
    # 为什么：批量与视频场景下逐图写 annotated.jpg 属纯 I/O 浪费 ——
    #   它们要的是聚合结论，不是每张配图。
    # 顺带把返回值也检查上：`cv2.imwrite` 对含非 ASCII 的路径会**静默返回 False**，
    #   本工程统一走 `imwrite_u`（内部 imencode + tofile）。
    annotated_path = None
    if args.get("save_annotated"):
        save_dir = Path(args.get("out", str(VIS_DIR)))
        ensure_dirs(save_dir)
        _ann = save_dir / f"{image_path.stem}_annotated.jpg"
        if imwrite_u(_ann, vis):
            annotated_path = _ann
        else:
            log(f"  [warn] 标注图落盘失败：{_ann}")

    return {
        "image": str(image_path),
        "size": [w, h],
        "status": "ok",
        "n_detections": len(dets),
        "calibration": {
            "mm_per_px": round(calib.mm_per_px, 5),
            "method": calib.method,
            "confidence": calib.confidence,
            "detail": calib.detail,
        },
        "interpretability": {
            "crack_0.3mm": interp_crack.to_dict(),
            "blob_10mm": interp_blob.to_dict(),
            "can_report_mm": interp_crack.level == "measurement",
            "can_screen": interp_blob.feasible,
        },
        "capability": capability["capability"],
        # 正射校正记录：None 表示本次未启用（见报告 §6.5）
        "rectify": rect,
        "image_used": (rect or {}).get("image_used", "original"),
        # ★ 显式合并 conf：Measurement.to_dict() 走 asdict()，只序列化
        #   dataclass 声明字段，而 conf 是运行时塞进 __dict__ 的额外字段
        #   ⇒ 不显式取出来，下游（含 video_screen 的跨帧聚合）拿到的
        #   置信度恒为 0.0。app.py 的 payload 有同样处理，两处口径须一致。
        #   本次新增（2026-09-25）——视频聚合需按置信度挑代表帧，
        #   缺了它就会把所有帧的代表都选成「第一个」。
        "measurements": [
            {**m.to_dict(), "conf": round(float(m.__dict__.get("conf", 0.0)), 4)}
            for m in measurements
        ],
        "grades": [g.to_dict() for g in grades],
        "risk": risk,
        # 未开启 --save-annotated 时为 None。**不要**写成不存在的路径 ——
        # 否则下游 JSON 会残留指向空文件的引用（video_screen.py:498 会透传本字段）。
        "annotated": (str(annotated_path) if annotated_path else None),
    }


def main() -> None:
    # 默认权重统一走 common.default_weight()（单一实现）。
    # ★ 2026-09-25 修正：原先硬写 v8s640，**早已不是主力**（主力 2026-09-20 起
    #   切到 v11s640），且路径落在训练工作目录 ⇒ 交付包里必然不存在。
    model_path = argv_flag("model", default_weight("v11s640"))
    single = argv_flag("image")
    folder = argv_flag("dir")
    save_json = argv_flag("json", "")
    args = {
        "conf": argv_flag("conf", "0.25"),
        "imgsz": argv_flag("imgsz", "640"),
        "distance": argv_flag("distance", "20"),
        "focal": argv_flag("focal", "24"),
        "calib_object_px": argv_flag("calib-object-px"),
        "calib_object_mm": argv_flag("calib-object-mm", "210"),
        "calib_object_name": argv_flag("calib-object-name", "A4短边210mm"),
        "calib_brick": argv_flag("calib-brick") is not None,
        "brick_pitch_mm": argv_flag("brick-pitch-mm", "250"),
        "env": argv_flag("env", "二类环境（露天/潮湿）"),
        "jgj125": argv_flag("jgj125", ""),
        "out": argv_flag("out", str(VIS_DIR)),
        # 标注图默认不落盘（批量/视频下逐图写盘是纯 I/O 浪费）
        "save_annotated": _flag("save-annotated"),
        # 斜拍正射校正（rectify.py / 报告 §6.5）
        "rectify": _flag("rectify"),
        "rectify_force": (str(argv_flag("rectify", "")).strip().lower() == "force"),
    }

    resolved = resolve_weight(model_path)
    log(f"模型: {resolved}")
    if not Path(resolved).exists():
        log("!! 模型不存在。请先训练（train.py）或用 --model 指定权重")
        return

    from ultralytics import YOLO
    model = YOLO(resolved)

    if single:
        targets = [Path(single)]
    elif folder:
        targets = list_images(folder)
    else:
        log("请指定 --image=<文件> 或 --dir=<目录>")
        return

    log(f"待处理 {len(targets)} 张")
    records = []
    for i, p in enumerate(targets, 1):
        log(f"[{i}/{len(targets)}] {p.name}")
        rec = run_one(p, model, args)
        records.append(rec)
        if rec.get("status") == "ok":
            log(f"  检测 {rec['n_detections']} 处；"
                f"GSD={rec['calibration']['mm_per_px']}mm/px；"
                f"裂缝0.3mm判读={rec['interpretability']['crack_0.3mm']['level']}；"
                f"风险={rec['risk']['level']}")
            if rec.get("rectify"):
                r0 = rec["rectify"]
                log(f"  正射校正：applied={r0['applied']} method={r0['method']} "
                    f"conf={r0['confidence']} uniform_scale={r0['uniform_scale']}"
                    f" -> 用图 {rec['image_used']}")
            if not rec["interpretability"]["can_report_mm"]:
                log("  -> 系统判定：本图不足以给出毫米级裂缝结论，仅输出筛查意见")

    out_json = Path(save_json) if save_json else (EVAL_DIR / "pipeline_results.json")
    dump_json({"model": resolved, "args": args, "results": records}, out_json)
    log(f"结果已写入 {out_json}")

    # 汇总
    ok = [r for r in records if r.get("status") == "ok"]
    if ok:
        n_mm = sum(1 for r in ok if r["interpretability"]["can_report_mm"])
        log("=" * 66)
        log(f"总计 {len(ok)} 张处理成功；其中 {n_mm} 张达到毫米级报告门槛"
            f"（{n_mm / len(ok):.0%}）")
        log(f"拒绝给定量结论的：{len(ok) - n_mm} 张 —— 这正是本系统诚实的价值所在")
        n_rect = sum(1 for r in ok
                     if r["calibration"]["method"] == "rectify_metric")
        if n_rect:
            log(f"其中 {n_rect} 张用了「校正后统一尺度」标定"
                f"（证据强度仅次于标定物）")
        log("=" * 66)


if __name__ == "__main__":
    main()
