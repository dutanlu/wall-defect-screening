# -*- coding: utf-8 -*-
"""
app.py —— 外墙缺陷筛查系统 Web 演示（06_deploy）

===== 为什么要有这个 =====

答辩演示如果只在命令行跑 `python pipeline.py`，评委会问「这东西给谁用」。
本项目的定位是「给街道/物业的日常巡检人员用的筛查工具」，
所以必须有一个「上传照片 → 看结论」的界面。

技术选型：**Gradio**。
  为什么不用 Flask/FastAPI + 前端：演示场景不需要用户系统/并发，
  Gradio 一个文件就能跑出可交互 UI，且天然支持「拖拽上传图片」和
  「画布标注返回」，把工程重心留在算法而不是胶水代码上。

===== 界面设计原则（对齐本项目的核心主张）=====

界面上必须显著展示「系统对自己的能力判断」——
这正是我们区别于「又一个裂缝检测器」的地方。因此结果区固定包含：

  1. GSD 与可判读性结论（大字提示：本图是否够格给毫米级结论）
  2. 房屋危险性等级 A/B/C/D + 「是否需请专业机构」
  3. 每处缺陷的量化值（毫米/平方米）与对应国标判据
  4. 明确的免责声明：本结果是筛查排序，不是法定鉴定

===== 命令 =====

  python app.py                      # 默认 localhost:7860
  python app.py --port=7860 --share  # 生成公网临时链接（演示给评委用）

⚠️ 关于 --share：Gradio 的 share 走的是国外隧道（gradio.live），
   本机网络环境下 CONNECT 隧道会 502。**演示前务必先测通**，
   若不通则改用局域网 IP 让评委在同一网络下访问。
"""

from __future__ import annotations

import sys
import json
import traceback
from pathlib import Path

# 把 02_code 加入 sys.path，才能 import 项目模块
DEPLOY_DIR = Path(__file__).resolve().parent
PROJ_ROOT = DEPLOY_DIR.parent
CODE_DIR = PROJ_ROOT / "02_code"
sys.path.insert(0, str(CODE_DIR))

import numpy as np                                    # noqa: E402

from common import (                                  # noqa: E402
    CLASS_CN,
    CLASSES,
    TRAIN_DIR,
    VIS_DIR,
    WEIGHTS_DIR,
    ensure_dirs,
    log,
)

# --------------------------------------------------------------------------
# 全局模型缓存（Gradio 每次请求都重新加载模型会慢到无法演示）
# --------------------------------------------------------------------------
_MODEL = None
_MODEL_PATH: str | None = None


def _find_default_weight() -> Path | None:
    """
    按优先级找一个可用权重：主力配置 > 任意已训练 > 兜底。

    2026-09-20 主力模型由 `v8s640` 切换为 `v11s640`（用户拍板）。
    切换依据（均为**独立测试集**同口径实测，见 04_results/eval/）：
      - mAP50：v11s640 **0.7230** vs v8s640 **0.7356**
        ⚠️ **V3 扩容后：v8s640 反而高 0.0126，且 < ≈0.03 判读阈值 ⇒ 判不出高下。**
        （此处曾写「v11s640 0.7628 vs 0.6766，+0.086 超阈值」—— 那是 **V2**
        测试集 67 图 / 259 实例的口径，**V3 扩容到 285 图 / 1646 实例后已不成立**，
        该说法已作废，2026-09-22 更正。）
      - mAP50-95：v11s640 **0.44105** vs v8s640 0.43244（略高，亦在噪声内）
      - 体积：v11s640 **18.33 MB** vs v8s640 21.51 MB（更小）
      - 假阳：v11s640 **109** vs v8s640 119（更少；**该计数为 V2 口径**，引用须注明）
      - 跨类误判：2 vs 1（两者都极少）
    ⇒ **选型依据 = 精度等效（判不出高下）+ 体积更小 + 误检更少**；
      **不得再表述为「v11s 精度更优」**（详见 07_report/技术报告.md §5.3.2 / §5.3.6）。

    ⚠️ 优先指向 `03_weights/<run>_best.pt`（**已剥离优化器状态的成品**），
    而不是 `04_results/train/<run>/weights/best.pt`：
    两者在本项目实测体积相同（如 v11s640 均为 18.33 MB），但
    `03_weights` 是交付语义明确的成品目录，路径更稳定；
    `04_results/train/` 属训练工作目录，重训时会被覆盖。
    """
    candidates = [
        # 主力：v11s640（成品权重优先）
        WEIGHTS_DIR / "v11s640_best.pt",
        TRAIN_DIR / "v11s640" / "weights" / "best.pt",
        # 兜底：其余已训练配置
        WEIGHTS_DIR / "v8s640_best.pt",
        TRAIN_DIR / "v8s640" / "weights" / "best.pt",
        WEIGHTS_DIR / "v8n640_best.pt",
        TRAIN_DIR / "v8n640" / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    for c in sorted(WEIGHTS_DIR.glob("*_best.pt")):
        return c
    for c in sorted(TRAIN_DIR.glob("*/weights/best.pt")):
        return c
    return None


def get_model(weight_path: str | None = None):
    """惰性加载并缓存 YOLO 模型。"""
    global _MODEL, _MODEL_PATH
    from ultralytics import YOLO

    if weight_path is None:
        w = _find_default_weight()
        if w is None:
            raise FileNotFoundError(
                "未找到任何已训练权重。请先运行 train.py，"
                "或在界面中指定权重路径。"
            )
        weight_path = str(w)

    if _MODEL is not None and _MODEL_PATH == weight_path:
        return _MODEL

    log(f"加载模型: {weight_path}")
    _MODEL = YOLO(weight_path)
    _MODEL_PATH = weight_path
    return _MODEL


# --------------------------------------------------------------------------
# 核心推理：单张图 → 结构化结论
# --------------------------------------------------------------------------
def analyze(image_rgb: np.ndarray,
            calib_mode: str,
            calib_object_px: float,
            calib_object_mm: float,
            brick_pitch_mm: float,
            distance_m: float,
            focal_mm: float,
            env_class: str,
            conf_thr: float,
            weight_path: str,
            rectify_mode: str = "关闭（正对拍摄）",
            jgj125_parts: list | None = None):
    """
    接收 Gradio 传来的 RGB 数组，跑完整管线，返回：
      (标注图 RGB, 结论 Markdown, 原始 JSON 字符串)

    rectify_mode：斜拍正射校正开关（rectify.py / 报告 §6.5）
      "关闭（正对拍摄）"   —— 默认，行为与接线前**逐字节一致**
      "自动校正（斜拍照片）" —— 判定为斜拍时才施加（正对图不做无谓重采样）
      "强制校正（对照实验）" —— 即使判定已正对也施加

    jgj125_parts：JGJ 125-2016 危险点判据的构件部位勾选（可多选、可为空）。
      自检报告 P1-6 指出：grade_crack 的 is_main_rebar_zone / is_slab_tension
      两个开关**从未被端到端链路传入过**，于是 0.50mm / 1.00mm 两级危险判据
      在演示里永远不触发。现由本参数显式接入；默认空 = 只走 GB 50010，
      输出与接线前逐字节一致。
    """
    import cv2

    from gsd import (
        GB50010_LIMITS_MM,
        Calibration,
        # [2026-09-23 新增] 成像维度自检。⚠️ 漏了这行**编译能过、运行时才 NameError**
        # —— 所以改完必须实际跑一次 analyze()，不能只看 py_compile。
        assess_image_quality,
        assess_interpretability,
        calibrate_by_brick_period,
        calibrate_by_camera,
        calibrate_by_object,
        screening_capability,
    )
    from grade import building_risk_level, grade_all
    from measure import draw_measurement, measure_instance
    from rectify import rectify as rectify_fun

    if image_rgb is None:
        return None, "请先上传一张外墙照片。", "{}"

    # Gradio 给的是 RGB，管线内部统一 BGR
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    # ---- 0) 可选：斜拍正射校正 ----
    # 必须放在检测**之前**：校正改变几何（因而也改变像素尺度），
    # 先检测再校正会让 bbox 整体错位。这也是 rectify.py 长期游离在
    # 主链路之外时最容易被忽略的一点。
    rect = None
    if rectify_mode and not rectify_mode.startswith("关闭"):
        _force = rectify_mode.startswith("强制")
        _rres = rectify_fun(img, force=_force)
        rect = _rres.to_dict()
        rect["image_used"] = "rectified" if _rres.applied else "original"
        if _rres.applied:
            img = _rres.image
            h, w = img.shape[:2]

    try:
        model = get_model(weight_path or None)
    except Exception as e:
        return None, f"### 模型加载失败\n\n```\n{e}\n```", "{}"

    # ---- 1) 检测 ----
    results = model.predict(img, conf=float(conf_thr), imgsz=640, verbose=False)
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
    calib = None
    if calib_mode.startswith("标定物"):
        if calib_object_px and calib_object_px > 0:
            calib = calibrate_by_object(
                pixel_width=float(calib_object_px),
                real_width_mm=float(calib_object_mm),
                object_name=f"标定物({calib_object_mm:g}mm)",
            )
    elif calib_mode.startswith("砖缝"):
        calib = calibrate_by_brick_period(img, brick_pitch_mm=float(brick_pitch_mm))

    # 正射校正给出的「**已验证**的全画面统一尺度」。
    # 排序理由：排在「标定物」之后 —— 标定物是画面里的直接物理参照物，
    # 证据最强；但排在裸「砖缝周期」之前 —— 后者默认全画面尺度一致、
    # 并未验证，而 rectify 的尺度是校正后**两轴自检**通过的（残差 <= 10%）。
    if (not calib_mode.startswith("标定物")) \
            and rect and rect.get("scale_mm_per_px"):
        _rc = rect.get("confidence")
        if _rc not in ("high", "medium", "low"):
            _rc = "low"
        calib = Calibration(
            mm_per_px=float(rect["scale_mm_per_px"]),
            method="rectify_metric",
            confidence=_rc,
            detail={
                "source": "rectify.py 正射校正（校正后两轴自检通过）",
                "rectify_method": rect.get("method"),
                "uniform_scale": bool(rect.get("uniform_scale")),
                "axis_residual": (rect.get("detail") or {}).get("axis_residual"),
                "note": ("全画面同一尺度，故整张图可用一个 GSD；"
                         "未校正的斜拍图 GSD 随位置变化，不能这样用。"),
            },
        )

    if calib is None:
        calib = calibrate_by_camera(
            distance_m=float(distance_m),
            image_width_px=w,
            focal_35mm=float(focal_mm),
        )

    # ---- 3) 可判读性（核心创新）----
    interp_crack = assess_interpretability(calib, 0.30)
    interp_blob = assess_interpretability(calib, 10.0)
    capability = screening_capability(calib)

    # ---- 4) 量化 ----
    measurements = []
    for d in dets:
        m = measure_instance(img, d["bbox"], d["cls"], calib)
        if m is not None:
            # 置信度写进 dataclass 的「额外字段」。注意 Measurement.to_dict()
            # 只序列化 dataclass 已声明字段，所以下面组装 payload 时必须
            # 显式把它取出来，否则界面上置信度恒显示 0.00（已实测踩到）。
            m.__dict__["conf"] = d["conf"]
            measurements.append(m)

    # ---- 5) 分级 ----
    # JGJ 125 两级危险点判据的开关（自检报告 P1-6：此前从未被传入过）。
    # 只影响裂缝的「判据优先级」，不影响任何测量数字。
    #
    # ⚠️ 不要以为「勾选后等级一定会升高」—— 结果取决于环境类别（已实测核对）：
    #   二类环境 GB 限值 0.20mm，其 danger 门槛是 2×限值 = **0.40mm**，
    #   比 JGJ 的 0.50mm 更严 ⇒ 勾选「主筋处」只把判据出处由 GB 换成 JGJ，
    #   **不改变等级**（例：0.60mm 裂缝在二类环境本来就是 danger）。
    #   真正能翻转等级的是一类环境（限值 0.30mm，danger 门槛 0.60mm）
    #   且宽度落在 (0.50, 0.60] mm 区间 —— 例：一类环境 0.55mm，
    #   默认 attention → 勾选后 danger，limit_mm 由 0.30 换为 0.50。
    _parts = set(jgj125_parts or [])
    is_main_rebar_zone = "梁板受力主筋处（0.50mm 危险点）" in _parts
    is_slab_tension = "板受拉区（1.00mm 危险点）" in _parts
    grades = grade_all(measurements, env_class=env_class,
                       is_main_rebar_zone=is_main_rebar_zone,
                       is_slab_tension=is_slab_tension)
    risk = building_risk_level(grades)

    # ---- 5b) 拒答的「传染性」处理（本项目最容易被忽略、但最要命的一环）----
    #
    # 场景：相机参数兜底给出 GSD=67mm/px 时，一条裂缝会被量成 230mm 宽，
    # 于是 grade_crack 判为 danger（超限 1000 倍），明细表里赫然写着「严重」。
    # 而横幅同时说「不可判读」。**自相矛盾的输出比不输出更危险** ——
    # 使用者会只看到「严重」两个字，据此做处置决策。
    #
    # 正确做法：一旦某类目标不可判读，就要把该类的**定量结论全部降级为
    # 「无法判定」**，只保留「此处疑似存在 X」的存在性提示。
    # 这就是 selective prediction 必须贯穿到输出层，而不能只停在横幅里。
    crack_interp = assess_interpretability(calib, 0.30)
    blob_interp = assess_interpretability(calib, 10.0)
    # [2026-09-23 新增·成像维度] 图像质量自检：**几何合格 != 图像可用**。
    # 实测（logs/_IMAGE_QUALITY_GATE.md）：低照度 L1.0 时几何判据仍说「可判读」，
    # 而 mAP50 已从 0.723 崩到 0.1586 —— 只靠几何会在一张近乎全黑的图上说「可判读」。
    _quality = assess_image_quality(img)
    _quality_blocked = not _quality.ok
    if _quality_blocked:
        crack_unjudgeable = True
        blob_unjudgeable = True
    else:
        crack_unjudgeable = crack_interp.level != "measurement"
        blob_unjudgeable = blob_interp.level != "measurement"

    def _is_line(cls: str) -> bool:
        return cls in ("crack", "exposed_rebar", "rust")

    for m, g in zip(measurements, grades):
        unjudgeable = (crack_unjudgeable if _is_line(m.cls_name)
                       else blob_unjudgeable)
        # [2026-09-23 新增·测量窗口上界] 几何上 px 够，但超出本 ROI 的可靠上界
        # (ks−1) 时同样会静默失明。实测见 logs/_RISK_COVERAGE.md：
        # 单侧判据 risk 60.7% -> 补上界后 29.2%。
        _win_bad = not bool(getattr(m, "window_ok", True))
        if _win_bad:
            unjudgeable = True
        if unjudgeable:
            # 该类的尺寸本来就不可信，其 severity 无意义 -> 强制降为
            # 「无法判定」，并把原因写清楚
            prev = g.severity
            g.severity = "unjudgeable"
            g.exceeded = False
            # ⚠️ reason 里原本写着「裂缝宽 230.19mm > 限值 0.20mm」——
            # 即 severity 作废了，那个荒唐的数字仍会从 reason 字段漏出去。
            # 使用者读到「230mm 超限」照样会做处置决策。所以 reason 必须一起替换。
            g.reason = (f"仅在图像中检出「{CLASS_CN.get(m.cls_name, m.cls_name)}」"
                        f"疑似存在（置信度 {float(m.__dict__.get('conf', 0.0)):.2f}）；"
                        f"尺寸与分级均无法判定")
            _why = ("图像质量不达标（" + _quality.note + "）" if _quality_blocked
                    else (m.window_note if _win_bad
                          else "该类目标达不到毫米级判读门槛"))
            g.reason = g.reason.rstrip("。") + f"（原因：{_why}）"
            g.confidence_note = (
                f"本条尺寸结论已作废：当前 GSD={calib.mm_per_px:.2f} mm/px 下，"
                f"该类目标达不到毫米级判读门槛（原判 {prev} 不可信）。"
                f"仅保留「疑似存在」提示，建议靠近复拍后再定级。"
                + (f" 【本次具体原因】{_why}。" if (_win_bad or _quality_blocked) else "")
            )
    # 拒答后重新汇总风险等级：不可判定的条目不应参与 C 级升级
    effective = [g for g in grades if g.severity != "unjudgeable"]
    n_void = len(grades) - len(effective)
    if effective:
        risk = building_risk_level(effective)
    else:
        # 全部条目都不可判定 —— 不能报 A（"安全"）。
        # 对筛查工具而言，「判不了」和「没问题」是两回事，
        # 把前者说成后者是最危险的误报。
        risk = building_risk_level([])
        risk["level"] = "U"
        risk["level_desc"] = ("当前图像质量/分辨率不足以支撑任何定量判定，"
                              "无法给出危险性等级。")
        risk["need_professional_inspection"] = True
    risk["n_unjudgeable"] = n_void
    risk["abstained"] = bool(n_void)

    # ---- 6) 画图 ----
    colors = {
        "crack": (0, 0, 255), "spalling": (0, 140, 255),
        "efflorescence": (180, 120, 0), "exposed_rebar": (255, 0, 180),
        "rust": (0, 100, 255), "delamination": (0, 200, 100),
        "moss": (60, 180, 75),
    }
    vis = img.copy()
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

    banner = [
        f"GSD = {calib.mm_per_px:.3f} mm/px ({calib.method})",
        *([f"rectified: {rect['method']} [{rect['confidence']}]"]
          if (rect or {}).get("applied") else []),
        f"0.3mm crack -> {interp_crack.pixels_on_target:.2f}px "
        f"[{interp_crack.level.upper()}]",
        f"10mm blob   -> {interp_blob.pixels_on_target:.2f}px "
        f"[{interp_blob.level.upper()}]",
        f"risk level: {risk['level']}",
    ]
    for i, t in enumerate(banner):
        y = 24 + i * 24
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 1, cv2.LINE_AA)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    payload = {
        "image_size": [w, h],
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
        },
        "capability": capability["capability"],
        # 正射校正记录：None 表示本次未启用（见报告 §6.5）
        "rectify": rect,
        "image_used": (rect or {}).get("image_used", "original"),
        # 显式合并 conf：Measurement.to_dict() 不含它（非 dataclass 字段）
        "measurements": [
            {**m.to_dict(), "conf": round(float(m.__dict__.get("conf", 0.0)), 4)}
            for m in measurements
        ],
        "grades": [g.to_dict() for g in grades],
        "risk": risk,
        # 分级判据口径（自检 P1-6）：必须进 payload，不能只留在 analyze() 的
        # 局部作用域里 —— render_markdown() 是独立函数，它只能看到 payload。
        # 这也是「判据开关一旦没接通，界面就会默认沉默」的根因所在。
        # [2026-09-23 新增] 成像维度自检结果（几何之外的第二重拒答依据）
        "image_quality": _quality.to_dict(),
        "grading_criteria": {
            "env_class": env_class,
            "gb50010_limit_mm": float(GB50010_LIMITS_MM.get(env_class, 0.30)),
            "is_main_rebar_zone": bool(is_main_rebar_zone),
            "is_slab_tension": bool(is_slab_tension),
        },
    }

    return vis_rgb, render_markdown(payload), _dumps(payload)


def _dumps(obj) -> str:
    """
    JSON 序列化，兼容 numpy 标量/数组。

    为什么需要：管线里的数值大量来自 numpy（conf、area_ratio、
    assess_interpretability 的 feasible 等），而 `json.dumps` 只认
    Python 原生类型。典型报错：
        TypeError: Object of type bool is not JSON serializable
    这个 bool 其实是 `numpy.bool_`，看起来像内置 bool 所以很难一眼看出。
    统一在这里做一次兜底转换，避免每个产出点都要记得转。
    """
    import numpy as np

    def _default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"不可序列化: {type(o).__name__}")

    return json.dumps(obj, ensure_ascii=False, indent=2, default=_default)


# --------------------------------------------------------------------------
# 结论渲染
# --------------------------------------------------------------------------
def render_markdown(p: dict) -> str:
    """把结构化结果渲染成人能读的 Markdown 结论。"""
    interp = p["interpretability"]["crack_0.3mm"]
    risk = p["risk"]
    calib = p["calibration"]
    # 分级判据口径（见 analyze() 的 payload）。用 .get 兜底，保证旧 payload
    # 也能渲染 —— 缺字段时按「未勾选危险点判据」处理，即最保守的口径。
    _gc = p.get("grading_criteria") or {}
    _gc = {"env_class": _gc.get("env_class", "二类环境（露天/潮湿）"),
           "gb50010_limit_mm": float(_gc.get("gb50010_limit_mm", 0.30)),
           "is_main_rebar_zone": bool(_gc.get("is_main_rebar_zone", False)),
           "is_slab_tension": bool(_gc.get("is_slab_tension", False))}

    # 顶部能力判定：这是本系统的灵魂，必须放最前面。
    #
    # ⚠️ 设计要点：**必须按类别分别陈述**，不能用一个总判定盖全。
    # 例：GSD=0.5mm/px 时，裂缝（0.3mm 目标）不可判读，
    # 但剥落/泛碱（10mm 目标，占 20px）是完全可测的。
    # 若只报「不可判读」，使用者会误以为整张图都没用，把已经算准的
    # 剥落面积也丢掉 —— 这是「过度拒答」，和「不拒答」一样是失败。
    ri = p["interpretability"]
    ck, bl = ri["crack_0.3mm"], ri["blob_10mm"]
    _lvl_cn = {"measurement": "可精测", "screening": "仅可筛查",
               "insufficient": "不可判读"}

    # 顶部状态徽标：让「系统对自己能力的判断」在视觉上先于任何数字出现。
    _calib_c = calib["confidence"]
    _calib_badge = {"high": "高", "medium": "中", "low": "低"}.get(_calib_c, _calib_c)
    badge = (f"`GSD {calib['mm_per_px']:.3f} mm/px` "
             f"· 标定置信度 **{_calib_badge}**（{calib['method']}） "
             f"· 检出 **{p['n_detections']}** 处")

    if ck["level"] == "measurement" and bl["level"] == "measurement":
        head = ("## 本图可给出毫米级结论\n\n"
                "> 裂缝与块状缺陷均达到量化门槛，可输出毫米/平方米结果并定级。\n\n"
                f"{badge}")
    elif ck["level"] == "insufficient" and bl["level"] == "insufficient":
        head = ("## 不可判读 —— 系统拒绝给定量结论\n\n"
                f"> {ck['message']}\n\n"
                f"{badge}\n\n"
                "**这不是失败，而是设计。** 系统不会为了「有输出」而编造一个数字 —— "
                "在房屋安全场景里，把「判不了」说成「没问题」比漏检更危险。")
    else:
        head = (
            "## 部分可判读 —— 按类别区别对待\n\n"
            f"> **裂缝类**：{_lvl_cn.get(ck['level'], ck['level'])}"
            f"（0.3mm 在图上仅 {ck['pixels_on_target']:.2f}px，"
            f"门槛 3px）\n>\n"
            f"> **块状类**（剥落/泛碱/露筋等）："
            f"{_lvl_cn.get(bl['level'], bl['level'])}"
            f"（10mm 在图上 {bl['pixels_on_target']:.2f}px）\n\n"
            f"{badge}\n\n"
            "系统只对达到门槛的类别出定量结论，未达门槛的类别降级为"
            "「疑似存在」提示。"
        )

    level_icon = {"A": "[A] 安全", "B": "[B] 个别危险点",
                  "C": "[C] 局部危房", "D": "[D] 整体危房",
                  "U": "[U] 无法判定"}
    n_unjudgeable = risk.get("n_unjudgeable", 0)
    lines = [
        head,
        "",
        "---",
        "",
        f"### 房屋危险性等级（筛查粗判）：**{level_icon.get(risk['level'], risk['level'])}**",
        "",
        f"{risk['level_desc']}",
        "",
        f"- 缺陷总数 **{risk['n_defects']}**（其中结构性 **{risk['structural_defects']}**）",
        f"- danger **{risk['n_danger']}** / attention **{risk['n_attention']}**"
        + (f" / 潮湿线索 **{risk['n_moisture_hints']}**"
           if risk.get("n_moisture_hints") else ""),
        f"- 是否需要专业机构进场：**{'是' if risk['need_professional_inspection'] else '否'}**",
    ]
    if n_unjudgeable:
        lines.append(
            f"- **尺寸不可判定、已作废的条目：{n_unjudgeable} 条**"
            f"（当前图像分辨率不足以支撑该类的毫米级结论，"
            f"这些条目**未计入**上面的风险等级）"
        )
    lines += [
        "",
        "### 尺度标定",
        "",
        f"- GSD = **{calib['mm_per_px']:.4f} mm/px**（来源：`{calib['method']}`，"
        f"置信度 **{calib['confidence']}**）",
        f"- 裂缝类可可靠测量的最小宽度：**{ck['min_readable_mm']:.2f} mm**",
        f"- 块状类可可靠测量的最小尺寸：**{bl['min_readable_mm']:.2f} mm**",
        "",
        "### 分级判据口径",
        "",
        f"- 环境类别：**{_gc['env_class']}** → GB 50010 裂缝宽度限值"
        f" **{_gc['gb50010_limit_mm']:.2f} mm**",
    ]
    # 把「这次到底走没走 JGJ 125」显式写出来。
    # 自检 P1-6 的教训：判据开关一旦没接通，报告和界面都会**默认沉默** ——
    # 使用者以为系统按危险房屋标准判过，其实只走了一般环境限值。
    if _gc["is_main_rebar_zone"] or _gc["is_slab_tension"]:
        _hit = []
        if _gc["is_slab_tension"]:
            _hit.append("板受拉区（1.00mm 危险点）")
        if _gc["is_main_rebar_zone"]:
            _hit.append("梁板受力主筋处（0.50mm 危险点）")
        lines.append(
            f"- **JGJ 125-2016 危险点判据已启用：{'、'.join(_hit)}**"
            f"（判据优先级：JGJ 125 危险点 > GB 50010 环境限值）"
        )
    else:
        lines.append(
            "- 构件部位未勾选 → **本次只走 GB 50010 环境限值**，"
            "JGJ 125-2016 危险点判据（0.50 / 1.00 mm）**未参与判定**"
        )
    lines.append("")

    # 斜拍校正结论：只有启用了才渲染。
    # 必须把「没校正成功」也讲清楚 —— 否则使用者会以为图已经被摆正了，
    # 而后面所有毫米数都建立在「画面同一尺度」这个前提上。
    _rt = p.get("rectify")
    if _rt:
        _used = "校正后图像" if _rt.get("applied") else "**原图（未校正）**"
        lines += [
            "### 斜拍校正（正射校正）",
            "",
            f"- 是否施加校正：**{'是' if _rt.get('applied') else '否'}**"
            f"（方法 `{_rt.get('method')}`，置信度 `{_rt.get('confidence')}`）",
            f"- 本次检测与量化使用的是：{_used}",
            "- 全画面统一尺度：" + (
                f"**{_rt['scale_mm_per_px']:.4f} mm/px**"
                if _rt.get("scale_mm_per_px")
                else "未取得（两轴自检未通过）—— 故本次改用其它标定来源"),
            "",
            f"> {_rt.get('message', '')}",
            "",
            f"> 建议：{_rt.get('advice', '')}",
            "",
        ]

    if not p["measurements"]:
        lines.append("### 未检出缺陷\n\n当前置信度阈值下没有检出任何缺陷。")
    else:
        lines += ["### 检出明细", "",
                  "| # | 类别 | 置信度 | 关键量 | 分级 | 依据 |",
                  "|---|------|--------|--------|------|------|"]
        for i, (m, g) in enumerate(zip(p["measurements"], p["grades"]), 1):
            cn = CLASS_CN.get(m["cls_name"], m["cls_name"])
            if m["cls_name"] in ("crack", "exposed_rebar", "rust") and m["length_mm"] > 0:
                metric = f"L={m['length_mm']:.0f}mm, W={m['width_max_mm']:.2f}mm"
            else:
                # 面积用 cm² 表达：外墙缺陷常在 1~100 cm² 量级，
                # 用 m² 会写成一长串 0.000x，用 mm² 又动辄五位数。
                # 高分辨率下 1cm×1cm 的剥落只有 0.0001 m²，
                # 按 m² 保留 4 位会直接舍入成 0.0000 —— 已实测踩到。
                cm2 = m["area_m2"] * 1e4
                metric = (f"面积={cm2:.2f}cm²" if cm2 >= 0.01
                          else f"面积={m['area_mm2']:.1f}mm²")
            # 「不可判定」的条目：尺寸结论已作废，不该再把毫米数摆在台面上，
            # 否则使用者只会记住那个数字。
            if g["severity"] == "unjudgeable":
                metric = "尺寸不可判定"
                sev = "**无法判定**"
            else:
                sev = {"danger": "**严重**", "attention": "关注",
                       "ok": "正常"}.get(g["severity"], g["severity"])
            conf = m.get("conf", 0.0)
            lines.append(
                f"| {i} | {cn} | {conf:.2f} | {metric} | {sev} "
                f"| {g['reason']} |"
            )
        lines.append("")
        # 置信度提醒
        notes = [g["confidence_note"] for g in p["grades"] if g.get("confidence_note")]
        if notes:
            lines += ["### 口径提醒", ""]
            for n in dict.fromkeys(notes):
                lines.append(f"- {n}")
            lines.append("")

    # 能力清单：直接展示「这套图像配置下能做什么、不能做什么」
    lines += ["### 本图的能力边界（诚实清单）", "",
              "| 目标 | 实际尺寸 | 图上像素 | 是否可用 |",
              "|------|---------|---------|---------|"]
    for name, v in p["capability"].items():
        mark = "可用" if v["usable"] else "不可用"
        lines.append(f"| {name} | {v['target_mm']} mm | {v['pixels']} px | {mark} |")
    lines.append("")

    # ---- 判据推导（可复算）------------------------------------------------
    # 赛道五「可解释性」的实质：不只给结论，还要让人**自己能复算**为什么。
    # 这里把「目标实际尺寸 ÷ GSD = 图上像素数，再与门槛比较」这条链子完整写出来，
    # 任何人拿计算器都能验证 —— 不依赖任何模型内部量（不是置信度、不是热力图）。
    # ---- 第二重判据：成像质量（2026-09-23 新增）----
    # 与几何判据并列展示：几何回答「分辨率够不够」，成像回答「拍清楚了没有」。
    _q = p.get("image_quality") or {}
    if _q:
        _qok = "达标" if _q.get("ok") else "**未达标（已触发拒答）**"
        lines += ["### 第二重判据：成像质量（几何之外）", "",
                  "几何合格只说明「分辨率够」，**不代表「拍清楚了」**——"
                  "实测低照度下几何判据仍说可判读，而 mAP50 已从 0.723 崩到 0.1586。",
                  "",
                  "| 指标 | 本图实测 | 门槛 |",
                  "|---|---|---|",
                  "| 亮度均值 | {} | 需在 57 ~ 230 之间 |".format(_q.get("brightness")),
                  "| 噪声 MAD | {} | ≤ 4.22 |".format(_q.get("noise_mad")),
                  "| 对比度 std | {} | 仅记录（本次未设门槛） |".format(_q.get("contrast_std")),
                  "| 清晰度（拉普拉斯方差） | {} | 仅记录（本次未设门槛） |".format(
                      _q.get("sharpness_lapvar")),
                  "",
                  "> 判定：**{}**{}".format(
                      _qok, ("　——　" + str(_q.get("note"))) if _q.get("note") else ""),
                  ""]

    _gsd = float(p["calibration"]["mm_per_px"])
    lines += ["### 为什么这样判定？（判据推导，可复算）", "",
              "本图的尺度基准：**GSD = {:.4f} mm/px**（来源 `{}`，置信度 {}）。".format(
                  _gsd, p["calibration"]["method"], p["calibration"]["confidence"]),
              "",
              "判据只有一条除法：**`图上像素数 = 目标实际尺寸 ÷ GSD`**，再与两个门槛比较 ——",
              "",
              "| 门槛 | 取值 | 含义 |",
              "|---|---|---|",
              "| `MIN_RELIABLE_PX` | **3.0 px** | 低于此值，测量误差会超过 1 个国标分级步长 |",
              "| `COMFORTABLE_PX` | **8.0 px** | 至少 8px 才能稳定做骨架化 / 距离变换 |",
              "",
              "逐项代入（`{}`）：".format(p["calibration"]["detail"]),
              "",
              "| 目标 | 实际尺寸 | ÷ GSD | 图上像素 | 与门槛比较 | 结论 |",
              "|---|---|---|---|---|---|"]
    for _nm, _v in p["capability"].items():
        _px = float(_v["pixels"])
        if _px >= 8.0:
            _cmp = "{:.2f} px ≥ 8 px（舒适）".format(_px)
        elif _px >= 3.0:
            _cmp = "3 ≤ {:.2f} px < 8（勉强）".format(_px)
        else:
            _cmp = "{:.2f} px < 3 px（低于最小可判读）".format(_px)
        lines.append("| {} | {} mm | ÷ {:.4f} | **{:.2f} px** | {} | {} |".format(
            _nm, _v["target_mm"], _gsd, _px, _cmp,
            "可判读" if _v["usable"] else "不可判读"))
    lines += ["",
              "> **这条判据不依赖任何模型内部量** —— 不是置信度、不是注意力热力图，",
              "> 而是一条**可复算的几何判据**：任何人对同一张图都能算出同一个结论。",
              "> 我们回答的不是「模型看了哪里」，而是"
              "「**这张图够不够格让我们下结论**」。",
              ""]

    lines += ["---", "", f"> {risk['disclaimer']}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 视觉主题
# --------------------------------------------------------------------------
# 与照片收集站（08_photo_collector）保持同一套视觉语言：
#   白底 + Apple 式中性灰阶 + 多色渐变点缀（蓝→青→紫）。
#
# 为什么**强制浅色**（而不是跟随系统）：
#   1. 演示场景是教室/会议室投屏。深色界面经投影仪输出后对比度会被压扁、
#      暗部糊成一片；浅色白底投出来反而更干净、字更清楚。
#   2. 本系统大量返回「拒绝下结论」的结论，这种提示**必须**在任何环境都醒目。
#      因此浅色下不用淡灰，而是给它**饱和的彩色边条 + 同色浅底**
#      （见下方 .prose blockquote 与状态横幅），靠"彩色"而不是"暗底"抓注意力。
#   3. 与收集站统一，两处不用再维护两套配色。
#
# ⚠️ Gradio 的深浅色由 body 上的 .dark 类驱动，而该类取决于浏览器偏好，
# 不能假定评委的浏览器是浅色。所以这里不依赖 .dark，而是直接用
# !important 覆盖 .gradio-container 及其子元素，
# 保证**任何浏览器下都是同一套浅色外观**（演示一致性 > 尊重系统主题）。
APP_CSS = """
:root{
  /* 中性层：白底 */
  --wb-bg:#ffffff; --wb-bg2:#fbfbfd; --wb-panel:#ffffff; --wb-panel2:#f5f5f7;
  --wb-line:#e8e8ed; --wb-line2:#d2d2d7;
  --wb-ink:#1d1d1f; --wb-ink2:#424245; --wb-sub:#6e6e73; --wb-dim:#86868b;
  /* 彩色：多色渐变的三个色源 */
  --wb-blue:#0b63e5; --wb-teal:#0d8f80; --wb-violet:#6a3ce0; --wb-pink:#c2409b;
  /* 渐变：文字用（压深，保证白底可读） */
  --wb-grad:linear-gradient(96deg,#0b63e5 0%,#0d8f80 46%,#6a3ce0 100%);
  /* 渐变：面/描边用（更亮） */
  --wb-grad-fill:linear-gradient(96deg,#0a84ff 0%,#28c8b8 48%,#8b5cf6 100%);
  /* 状态色（浅底专用，比深色版更深） */
  --wb-ok:#1a8f4a; --wb-err:#d7263d; --wb-warn:#b26a00;
  --wb-ease:cubic-bezier(.16,1,.3,1);
  --wb-spring:cubic-bezier(.34,1.56,.64,1);
  --wb-sh:0 1px 2px rgba(0,0,0,.04), 0 4px 14px -8px rgba(0,0,0,.07);
}
body, .gradio-container{
  background:var(--wb-bg) !important;
  color:var(--wb-ink) !important;
  color-scheme:light !important;
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI","PingFang SC",
              "Hiragino Sans GB","Microsoft YaHei",sans-serif !important;
  -webkit-font-smoothing:antialiased !important;
  -moz-osx-font-smoothing:grayscale !important;
}
.gradio-container{
  max-width:1280px !important;
  /* 彩色只大面积出现在这里：白底上三团柔光（蓝 / 青 / 紫） */
  background-image:
    radial-gradient(900px 520px at 50% -12%, rgba(10,132,255,.13), transparent 66%),
    radial-gradient(760px 460px at 8% 4%,    rgba(40,200,184,.13), transparent 64%),
    radial-gradient(780px 480px at 94% 10%,  rgba(139,92,246,.12), transparent 64%) !important;
  animation:wb-in .8s var(--wb-ease) both !important;
}
@keyframes wb-in{
  from{ opacity:0; transform:translateY(14px); }
  to  { opacity:1; transform:none; }
}
/* 卡片 / 面板 */
.block, .form, .panel, .gr-box, .gr-panel,
.gradio-container .gr-group, .gradio-container .gr-accordion{
  background:var(--wb-panel) !important;
  border:1px solid var(--wb-line) !important;
  border-radius:18px !important;
  box-shadow:var(--wb-sh) !important;
  transition:border-color .5s var(--wb-ease), box-shadow .5s var(--wb-ease) !important;
}
.block:hover, .gr-group:hover{
  border-color:#c9dcfb !important;
  box-shadow:0 2px 6px rgba(0,0,0,.04), 0 16px 38px -20px rgba(0,0,0,.14) !important;
}
/* 文字 */
h1,h2,h3,h4,label,span,p,.prose, .prose *{
  color:var(--wb-ink) !important;
}
h1{
  font-weight:700 !important;
  letter-spacing:-.045em !important;
  line-height:1.06 !important;
}
h2{ font-weight:600 !important; letter-spacing:-.03em !important; }
h3{ font-weight:600 !important; letter-spacing:-.02em !important; }
/* 引用块：彩色边条 + 同色浅底。
   这是「不可判读」类提示的主要视觉载体，浅色下靠饱和彩色抓注意力。 */
.prose blockquote{
  border-left:4px solid transparent !important;
  border-image:var(--wb-grad-fill) 1 !important;
  background:linear-gradient(96deg,rgba(10,132,255,.07),rgba(139,92,246,.07)) !important;
  color:var(--wb-ink2) !important;
  border-radius:0 10px 10px 0 !important;
  padding:12px 18px !important;
}
.prose table{
  border-collapse:collapse !important; font-size:13px !important;
  background:#ffffff !important;
}
.prose th{
  background:var(--wb-panel2) !important; color:var(--wb-ink2) !important;
  border:1px solid var(--wb-line) !important; padding:8px 10px !important;
}
.prose td{
  border:1px solid var(--wb-line) !important; padding:7px 10px !important;
  background:transparent !important; color:var(--wb-ink2) !important;
}
.prose code{
  background:var(--wb-panel2) !important; color:var(--wb-blue) !important;
  border:1px solid var(--wb-line) !important; border-radius:6px !important;
  padding:1px 6px !important;
  font-weight:600 !important;
}
/* 输入控件 */
input, textarea, select, .gr-input, .gr-text-input{
  background:#ffffff !important; color:var(--wb-ink) !important;
  border:1px solid var(--wb-line2) !important; border-radius:12px !important;
}
input:focus, textarea:focus, select:focus{
  border-color:var(--wb-blue) !important;
  box-shadow:0 0 0 4px rgba(10,132,255,.14) !important;
  outline:none !important;
}
/* 主按钮：整块多色渐变 + 白字，界面最主要的彩色落点 */
button{
  transition:background .28s var(--wb-ease), border-color .28s var(--wb-ease),
             color .28s var(--wb-ease), transform .34s var(--wb-spring),
             box-shadow .34s var(--wb-ease), opacity .2s linear !important;
}
button:hover:not(:disabled){ transform:translateY(-1px) !important; }
button:active:not(:disabled){
  transform:scale(.975) !important;
  transition-duration:.09s !important;
}
button.primary, .gr-button-primary{
  position:relative !important; overflow:hidden !important;
  background:var(--wb-grad) !important;
  border:1px solid transparent !important; color:#ffffff !important;
  font-weight:700 !important;
  box-shadow:0 10px 30px -14px rgba(11,99,229,.55) !important;
}
button.primary:hover, .gr-button-primary:hover{
  background:var(--wb-grad) !important; filter:brightness(1.08) !important;
  box-shadow:0 14px 34px -14px rgba(11,99,229,.62) !important;
}
/* 掠过式高光，让主按钮「动」起来 */
button.primary::after, .gr-button-primary::after{
  content:"" !important; position:absolute !important; inset:0 !important;
  pointer-events:none !important;
  background:linear-gradient(105deg,transparent 34%,rgba(255,255,255,.45) 50%,transparent 66%) !important;
  transform:translateX(-130%) !important;
  transition:transform .9s var(--wb-ease) !important;
}
button.primary:hover::after, .gr-button-primary:hover::after{
  transform:translateX(130%) !important;
}
/* 次级按钮 */
button:not(.primary){
  background:#ffffff !important; color:var(--wb-ink) !important;
  border:1px solid var(--wb-line2) !important; border-radius:12px !important;
}
button:not(.primary):hover{
  background:var(--wb-panel2) !important; border-color:#b8b8bf !important;
}
button:focus-visible{
  outline:none !important;
  box-shadow:0 0 0 4px rgba(10,132,255,.22) !important;
}
/* 折叠面板标题 */
.gradio-container .label-wrap span, .gradio-container summary{
  color:var(--wb-ink2) !important; font-weight:600 !important;
}
/* 图像 / 代码区 */
.image-container, .gr-image, .code-wrap, .gr-code{
  background:var(--wb-bg2) !important; border:1px solid var(--wb-line) !important;
  border-radius:14px !important;
  transition:border-color .5s var(--wb-ease) !important;
}
.image-container:hover{ border-color:#b8b8bf !important; }
/* 折叠面板：展开更顺滑 */
.gradio-container .gr-accordion > .label-wrap{ transition:background .3s var(--wb-ease) !important; }
.gradio-container .label-wrap:hover{ background:rgba(0,0,0,.018) !important; }
/* 尊重系统「减弱动态效果」 */
@media (prefers-reduced-motion: reduce){
  .gradio-container, button, .block{
    animation:none !important; transition:none !important;
  }
  button:hover:not(:disabled){ transform:none !important; }
  button.primary::after{ display:none !important; }
}
footer{ display:none !important; }
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-track{background:#ffffff}
::-webkit-scrollbar-thumb{background:#d2d2d7;border-radius:10px;border:2px solid #ffffff}
::-webkit-scrollbar-thumb:hover{background:#b8b8bf}
"""

HEAD_HTML = """
<style>
  .gradio-container { min-height: 100vh; }
</style>
"""


def _theme_and_style(gr):
    """
    构造主题，并按 Gradio 版本决定把它们传给谁。

    ⚠️ 踩过的坑：**Gradio 6.0 起 `theme` / `css` / `head` 从 `Blocks(...)`
    构造器移到了 `launch(...)`。** 旧写法在 6.x 上不会报错，只会抛一条
    UserWarning 然后**静默忽略掉整套主题与 CSS** —— 界面上看起来「能跑」，
    但所有美化全部失效，且没有任何明显报错。这比直接崩溃更难发现。

    因此这里显式按主版本号分流，两个版本都能正确生效。

    返回 (传给 Blocks 的 kwargs, 传给 launch 的 kwargs)。
    """
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
    ).set(
        body_background_fill="#ffffff",
        body_text_color="#1d1d1f",
        block_background_fill="#ffffff",
        block_border_color="#e8e8ed",
        block_label_text_color="#424245",
        block_title_text_color="#424245",
        block_radius="18px",
        input_background_fill="#ffffff",
        input_border_color="#d2d2d7",
        input_radius="12px",
        button_primary_background_fill="#0b63e5",
        button_primary_background_fill_hover="#0a84ff",
        button_primary_text_color="#ffffff",
        button_large_radius="12px",
        button_small_radius="12px",
    )
    style = {"theme": theme, "css": APP_CSS, "head": HEAD_HTML}

    try:
        major = int(str(gr.__version__).split(".")[0])
    except Exception:
        major = 5
    return ({}, style) if major >= 6 else (style, {})


# --------------------------------------------------------------------------
# Gradio 界面
# --------------------------------------------------------------------------
def build_ui():
    import gradio as gr

    default_w = _find_default_weight()
    default_w_str = str(default_w) if default_w else ""

    blocks_kwargs, _launch_kwargs = _theme_and_style(gr)

    with gr.Blocks(title="外墙缺陷智能筛查系统", **blocks_kwargs) as demo:
        gr.Markdown(
            "# 外墙缺陷智能筛查系统\n"
            "**上传一张外墙照片，系统给出「哪里有缺陷、有多大、要不要请专业机构」的筛查结论。**\n\n"
            "> 本系统是**筛查工具**，不是法定鉴定工具。"
            "它会主动告诉你「这张图够不够格下结论」。\n\n"
            "`判读 0.3mm 裂缝需 GSD ≤ 0.1 mm/px` "
            "· `主摄须贴近到 27cm` · `像素门槛 3px / 8px` "
            "· `主力模型 v11s640 (yolo11s) · 测试集 mAP50 0.723"
            "（独立测试集 285 图 / 1646 实例 / 7 类）`"
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(label="上传外墙照片", type="numpy")
                with gr.Accordion("尺度标定（决定能不能测毫米）", open=True):
                    gr.Markdown(
                        "**下面三种标定方式只用一种。** 只有你选中的那种，才用得上它对应的"
                        "输入框；**没选中的框不参与计算**，保持默认值即可。\n"
                        "\n"
                        "| 标定方式 | 需要你填的框 | 什么时候用 |\n"
                        "|---|---|---|\n"
                        "| 相机参数估算（最粗） | 拍摄距离 + 等效焦距 | 画面里没有参照物，"
                        "只想知道大致量级 |\n"
                        "| **标定物（最准，推荐）** | 标定物像素宽度 + 实际长度 | 画面里有"
                        "尺子 / A4 纸 / 硬币，能量出它在图上占多少像素 |\n"
                        "| 砖缝周期（无标定物时） | 砖模数 | 画面是规则砖墙，用砖缝间距推算 |\n"
                        "\n"
                        "> **为什么「标定物」最准**：它是画面里的**直接物理参照物**，"
                        "不需要你报拍摄距离和镜头参数。\n"
                        "> 而「相机参数估算」全靠你手填的距离与焦距，**这两个数填错，"
                        "算出来的毫米数就跟着错**（误差约成正比）。"
                    )
                    calib_mode = gr.Radio(
                        ["相机参数估算（最粗）", "标定物（最准，推荐）", "砖缝周期（无标定物时）"],
                        value="相机参数估算（最粗）",
                        label="标定方式",
                        info="三选一。切换后请对照上表填对应的框 —— 其余框留默认值即可。",
                    )
                    with gr.Row():
                        calib_object_px = gr.Number(
                            value=0,
                            label="标定物像素宽度（标定物在图上占多少 px）",
                            info="默认为 0 表示「未标定」，此时本项不生效。"
                                 "要启用，先用看图工具在照片上量出标定物的像素宽度再填。"
                                 "例：A4 纸短边在图上占 420 像素，就填 420。")
                        calib_object_mm = gr.Number(
                            value=210,
                            label="标定物实际长度(mm)",
                            info="标定物的真实长度。常用取值：A4 短边 210 / A4 长边 297 / "
                                 "一元硬币直径 25。")
                    with gr.Row():
                        brick_pitch_mm = gr.Number(
                            value=250,
                            label="砖模数(mm，标准砖250)",
                            info="一块砖加一条灰缝的重复间距。标准砖约 250mm；"
                                 "只在选「砖缝周期」时参与计算。")
                        distance_m = gr.Number(
                            value=20,
                            label="拍摄距离(m)",
                            info="相机到墙面的直线距离。只在选「相机参数估算」时参与计算 ——"
                                 "填得越准，毫米数越可信。")
                    focal_mm = gr.Number(
                        value=24,
                        label="等效焦距(mm)",
                        info="镜头的等效 35mm 焦距。手机主摄约 24，2 倍变焦约 50，"
                             "5 倍约 120。只在选「相机参数估算」时参与计算。")
                    rectify_mode = gr.Radio(
                        ["关闭（正对拍摄）", "自动校正（斜拍照片）", "强制校正（对照实验）"],
                        value="关闭（正对拍摄）",
                        label="斜拍正射校正",
                        info="斜拍会让 GSD 随位置变化，毫米数不再可信。开启后先按砖缝"
                             "把立面校正到正对；若两轴自检通过，会改用「全画面统一尺度」"
                             "标定（证据强度仅次于标定物）。",
                    )
                with gr.Accordion("其它参数", open=False):
                    env_class = gr.Dropdown(
                        ["一类环境（室内干燥）", "二类环境（露天/潮湿）", "三类环境（干湿交替/海风）"],
                        value="二类环境（露天/潮湿）", label="环境类别（GB 50010）")
                    jgj125_parts = gr.CheckboxGroup(
                        ["梁板受力主筋处（0.50mm 危险点）", "板受拉区（1.00mm 危险点）"],
                        value=[],
                        label="构件部位（JGJ 125-2016 危险点判据）",
                        info="勾选后判据优先级变为「JGJ 125 危险点 > GB 50010 环境限值」。"
                             "不勾选时只走 GB 50010（0.30/0.20mm）—— 这是此前的默认行为。"
                             "危险点判据只改判定等级，不改任何毫米测量值。")
                    conf_thr = gr.Slider(0.05, 0.9, value=0.25, step=0.05,
                                         label="检测置信度阈值")
                    weight_path = gr.Textbox(value=default_w_str, label="模型权重路径")
                btn = gr.Button("开始分析", variant="primary", size="lg")

            with gr.Column(scale=1):
                img_out = gr.Image(label="标注结果")
                md_out = gr.Markdown(label="结论")
                with gr.Accordion("原始 JSON（可对接台账系统）", open=False):
                    json_out = gr.Code(label="", language="json")

        # ⚠️ inputs 的顺序**必须**与 analyze() 的形参顺序逐项对应（位置绑定，不按名字匹配）：
        #      ... env_class, conf_thr, weight_path, rectify_mode, jgj125_parts
        #    历史上这里曾把末尾四项写成 jgj125_parts/conf_thr/weight_path/rectify_mode，
        #    导致 weight_path 收到 0.25、rectify_mode 收到权重路径字符串（每张图被误触发正射校正）。
        #    改动本行前请先跑 logs/_verify_app_inputs.py 的签名对齐断言。
        btn.click(
            fn=analyze,
            inputs=[img_in, calib_mode, calib_object_px, calib_object_mm,
                    brick_pitch_mm, distance_m, focal_mm, env_class,
                    conf_thr, weight_path, rectify_mode, jgj125_parts],
            outputs=[img_out, md_out, json_out],
        )

        gr.Markdown(
            "---\n"
            "### 怎么读这个结果\n\n"
            "1. **先看最上面那一行状态** —— 它说的是「这张图够不够格下结论」，"
            "而不是「有没有裂缝」。\n"
            "2. **再看明细表** —— 凡是标注「尺寸不可判定」的行，"
            "它的毫米数已经被系统主动作废，**不要采信**。\n"
            "3. **最后看「能力边界诚实清单」** —— "
            "逐项目标列出「实际尺寸 / 图上像素 / 是否可用」。\n\n"
            "> **给评委的一句话**：本项目的核心不是「又训了一个裂缝检测器」，"
            "而是让系统**知道自己什么时候不可靠** —— "
            "在房屋安全领域，误判为安全比漏检更危险。"
        )
    return demo


def main() -> None:
    def _arg(name, default):
        prefix = f"--{name}="
        for a in sys.argv[1:]:
            if a.startswith(prefix):
                return a[len(prefix):]
        return default

    port = int(_arg("port", "7860"))
    share = "--share" in sys.argv[1:]

    ensure_dirs(VIS_DIR)
    w = _find_default_weight()
    log(f"可用权重: {w if w else '（未找到，请先训练）'}")

    try:
        import gradio as gr  # noqa: F401
    except ImportError:
        log("!! 未安装 gradio。请执行：")
        log(r'   D:\下载\python.exe -m pip install gradio '
            r'-i https://pypi.tuna.tsinghua.edu.cn/simple')
        return

    demo = build_ui()
    _blocks_kwargs, launch_style = _theme_and_style(gr)
    log(f"启动 Web 界面: http://127.0.0.1:{port}")
    log(f"视觉主题传参位置: {'launch()（Gradio 6+）' if launch_style else 'Blocks()（Gradio 5-）'}")
    try:
        demo.launch(server_name="127.0.0.1", server_port=port, share=share,
                    show_error=True, **launch_style)
    except TypeError:
        # 万一某个版本的 launch() 不接受这些参数，退回到无主题启动，
        # 而不是让整个演示起不来。宁可朴素，不可打不开。
        log("!! launch() 不接受主题参数，改用默认外观启动")
        demo.launch(server_name="127.0.0.1", server_port=port, share=share,
                    show_error=True)
    except Exception:
        log("启动失败:")
        log(traceback.format_exc())


if __name__ == "__main__":
    main()
