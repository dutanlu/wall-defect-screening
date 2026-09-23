# -*- coding: utf-8 -*-
"""
gsd.py —— GSD 感知与「可判读性」判定（本项目的核心创新模块）

===== 这是什么，为什么它是创新点 =====

现有外墙缺陷检测研究几乎都隐含一个假设：**输入图像是合格的**。
没有人问「站在楼下 30 米外仰拍的照片，能不能用来判结构安全」。

但物理上这是硬约束：
    GSD（地面采样距离）= 画面水平覆盖宽度 / 图像水平像素数
    可判读性 = 目标实际尺寸 / GSD        （经验门槛：>= 3 像素才可靠）

    主摄 24mm（水平 FOV≈74°）→ 覆盖宽度 ≈ 1.5d
    1200 万像素宽 4000px → GSD = 1.5d / 4000

    | 拍摄距离 | 主摄 GSD   | 5x长焦 GSD |
    |---------|-----------|-----------|
    |   5 m   | 1.9 mm/px | 0.38 mm/px|
    |  10 m   | 3.8 mm/px | 0.75 mm/px|
    |  20 m   | 7.5 mm/px | 1.5  mm/px|
    |  50 m   | 18.8 mm/px| 3.8  mm/px|

    国标阈值（GB 50010-2010）：一类环境裂缝限值 0.3mm、二类环境 0.2mm
    → 要判读 0.3mm 需要 GSD <= 0.1mm/px
    → 主摄须贴近到 27cm，5x 长焦须 1.3m。**地面拍高层根本做不到。**

结论：外墙工作的正确定位是**筛查**而非**鉴定**：
  - 筛查层（地面拍摄可行）：找「哪栋楼、哪个立面有问题」，
    目标缺陷是剥落/脱落/空鼓/渗水/露筋/锈迹，尺寸在厘米~米量级。
    1cm 剥落在 5x 长焦 20m 外 ≈ 6.7 px，绰绰有余。
  - 鉴定层（须无人机贴近或搭架）：测裂缝宽度是否超 0.2/0.3/0.5mm。

所以系统的第一等公民能力是：**自己判断图像够不够格，不够格就拒绝给结论**。
这就是 selective prediction / abstention。
答辩核心话术：
  「我们不是又一个裂缝检测器，而是第一个**知道自己什么时候不可靠**的外墙筛查系统
   —— 因为在房屋安全这件事上，误判为安全比漏检更危险。」

===== 三个输入来源（按可信度降序）=====
1. 实测标定（最可信）：用双面胶贴一张 A4 纸（或已知尺寸标定板）在墙面同距离拍一张，
   由标定物像素宽度直接反算 mm/px。推荐现场就用这一招。
2. 砖缝周期反推：标准砖（240x115x53）+ 灰缝 10mm，竖缝水平周期约 250mm，
   对墙面做自相关/FFT，从周期像素数反推 mm/px。需拍摄正对墙面且砖缝清晰。
3. 相机参数推算（最粗）：由焦距、传感器宽度、拍摄距离算，误差最大，仅作兜底。

本模块不依赖任何深度学习，纯几何，因此结果可解释、可被评委复算。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal

import cv2
import numpy as np

# --------------------------------------------------------------------------
# 国标阈值（写进代码便于报告引用）
# --------------------------------------------------------------------------
# GB 50010-2010《混凝土结构设计规范》表 3.4.4 最大裂缝宽度限值
GB50010_LIMITS_MM: dict[str, float] = {
    "一类环境（室内干燥）": 0.30,
    "二类环境（露天/潮湿）": 0.20,
    "三类环境（干湿交替/海风）": 0.20,
}
# JGJ 125-2016《危险房屋鉴定标准》危险点判据（构件受力主筋处）
JGJ125_DANGER_MM: dict[str, float] = {
    "梁板受力主筋处横向/斜裂缝": 0.50,
    "板受拉裂缝": 1.00,
}

# 最小可判读像素数（经验值）。低于此值，测量误差 > 1 个国标分级步长。
MIN_RELIABLE_PX: float = 3.0
# 毫米级测量的舒适门槛：至少 8 px 才能稳定做骨架化/距离变换
COMFORTABLE_PX: float = 8.0


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclass
class Calibration:
    """尺度标定结果：mm/px 及其来源与可信度。"""
    mm_per_px: float                 # 每像素代表的实际毫米数
    method: str                      # 来源：calib_object / brick_period / camera_params
    confidence: Literal["high", "medium", "low"]
    detail: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mm_per_px <= 0:
            raise ValueError("mm_per_px 必须为正数")

    @property
    def px_per_mm(self) -> float:
        return 1.0 / self.mm_per_px


@dataclass
class Interpretability:
    """可判读性判定结果。"""
    feasible: bool                   # 是否达到可靠判读门槛
    min_readable_mm: float           # 当前图像能可靠测量的最小尺寸
    comfort_mm: float                # 舒适测量门槛
    gsd_mm_per_px: float
    target_mm: float                 # 被判定的目标尺寸（如 0.3mm 裂缝）
    pixels_on_target: float          # 目标在当前图像上占多少像素
    level: str                       # screening / measurement / insufficient
    message: str                     # 给用户的人话结论
    advice: str                      # 如何改进

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# 标定方式 1：已知尺寸标定物（推荐，最可信）
# --------------------------------------------------------------------------
def calibrate_by_object(pixel_width: float, real_width_mm: float,
                        object_name: str = "A4 纸短边(210mm)") -> Calibration:
    """
    由标定物的像素宽度反算 mm/px。

    现场做法：拿一张 A4 纸（210x297mm）贴在待测墙面，在与目标同距离处拍一张，
    量出纸边的像素长度，代入即可。这是最可靠的现场标定法。
    """
    if pixel_width <= 0:
        raise ValueError("pixel_width 必须为正")
    return Calibration(
        mm_per_px=real_width_mm / pixel_width,
        method="calib_object",
        confidence="high",
        detail={"object": object_name, "pixel_width": pixel_width,
                "real_width_mm": real_width_mm},
    )


# --------------------------------------------------------------------------
# 标定方式 2：砖缝周期反推（无标定物时的首选）
# --------------------------------------------------------------------------
def calibrate_by_brick_period(image: np.ndarray,
                              brick_pitch_mm: float = 250.0,
                              axis: str = "x",
                              min_period_px: int = 8,
                              max_period_px: int = 400) -> Calibration | None:
    """
    利用标准砖的周期性反推尺度。

    原理：标准砖 240mm + 灰缝 10mm = 250mm 水平模数。墙面砖缝是准周期结构，
    对灰缝增强后的投影曲线做自相关，主峰位置即一个砖距的像素数。

    适用条件（不满足则返回 None，交给调用方回退到下一种方法）：
      - 拍摄方向大致正对墙面（严重透视会破坏周期间隔）
      - 砖缝在图像里清晰可辨
      - 画面内至少能看到 3~4 个完整砖距

    参数：
      brick_pitch_mm: 一个砖模数的实际长度。标准砖 240 + 灰缝 10 = 250mm。
                     不同砌块需相应调整。
      axis: 'x' 取水平砖缝周期，'y' 取垂直。
    """
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # 1) 提取竖直灰缝的响应。
    #
    # 踩过的两个坑（都很隐蔽，记录在此避免重犯）：
    #  坑1：直接对灰阶做投影——砖面本身的亮度起伏（砌块色差）会盖过灰缝。
    #  坑2：用「细长」黑帽核(1x24)——竖直灰缝在每一整列上都有响应，
    #       投影后 x 方向几乎变成常数，自相关在真周期处是负值
    #       （实测 ac[80]=-0.043），周期完全消失。
    #
    # 正确做法：竖直灰缝的本质是「水平方向的强梯度」。
    # 用水平 Sobel 求 x 方向梯度，再沿 y 取「高分位」（如 90%），
    # 既保留灰缝的列位置信息，又不会被砖面噪声主导。
    if axis == "x":
        grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    else:
        grad = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(grad)

    # 2) 沿正交方向取 90 分位投影（比 mean 抗噪，比 max 稳定）
    proj = (np.percentile(mag, 90, axis=0) if axis == "x"
            else np.percentile(mag, 90, axis=1)).astype(np.float64)
    if proj.size < 4 * min_period_px:
        return None

    # 调制深度检查：信号太弱说明墙面没有清晰周期结构
    # （太糊 / 透视畸变过强 / 不是砖墙）。强行给结果比拒绝更危险。
    mod = proj.std() / (proj.mean() + 1e-9)
    if mod < 0.05:
        return None

    proj = proj - proj.mean()

    # 3) 自相关
    ac = np.correlate(proj, proj, mode="full")[proj.size - 1:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]

    # 4) 在主峰候选范围里找峰值（跳过 0 延迟）
    hi = min(max_period_px, ac.size - 1)
    if hi <= min_period_px:
        return None
    seg = ac[min_period_px:hi]
    if seg.size == 0:
        return None

    # 关键：自相关会同时出现基频与倍频峰。直接取全局最大容易抓到倍频
    # （实测把 80px 的真周期误判成 143px，误差 44%）。
    # 正确做法：**基频一定是最小的那个显著峰**。所以先找所有局部极大，
    # 再从「最小的显著峰」开始，验证它是否真是周期（检查其倍频处是否也有峰）。
    peaks: list[tuple[int, float]] = []
    for i in range(1, seg.size - 1):
        if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.10:
            peaks.append((i + min_period_px, float(seg[i])))
    if not peaks:
        return None

    global_max = max(v for _, v in peaks)
    if global_max < 0.15:
        return None

    # 按周期从小到大试；第一个满足「2 倍周期处也有显著峰」的就是基频。
    # 若某候选的倍频处没有峰，说明它是噪声，跳过。
    def peak_at(lag):
        """取 lag 附近的局部最大自相关值（容忍 ±2px 误差）。"""
        idx = int(round(lag)) - min_period_px
        lo, hi = max(0, idx - 2), min(seg.size, idx + 3)
        return float(seg[lo:hi].max()) if hi > lo else 0.0

    chosen = None
    for lag, val in sorted(peaks, key=lambda p: p[0]):
        if val < 0.55 * global_max:
            continue
        # 验证倍频：2x 与 3x 中至少一个显著
        if peak_at(2 * lag) >= 0.4 * val or peak_at(3 * lag) >= 0.3 * val:
            chosen = (lag, val)
            break
    if chosen is None:
        # 没有可验证的基频，退回全局最大（至少结果可复现）
        chosen = max(peaks, key=lambda p: p[1])

    lag, peak = chosen

    # 5) 亚像素精修：在主峰附近做抛物线拟合
    if 0 < lag < ac.size - 1:
        y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-9:
            lag = lag + 0.5 * (y0 - y2) / denom
    lag = float(lag)
    if lag <= 0:
        return None

    mm_per_px = brick_pitch_mm / lag
    conf = "high" if peak > 0.35 else ("medium" if peak > 0.22 else "low")
    return Calibration(
        mm_per_px=mm_per_px,
        method="brick_period",
        confidence=conf,
        detail={"brick_pitch_mm": brick_pitch_mm, "period_px": round(lag, 2),
                "autocorr_peak": round(peak, 3), "axis": axis,
                "modulation_depth": round(float(mod), 4)},
    )


# --------------------------------------------------------------------------
# 标定方式 3：相机参数推算（兜底）
# --------------------------------------------------------------------------
def calibrate_by_camera(distance_m: float,
                        image_width_px: int,
                        focal_35mm: float = 24.0,
                        sensor_width_mm: float = 36.0) -> Calibration:
    """
    由拍摄距离和相机参数推算 GSD。误差最大（不知道真实焦距/裁切），仅兜底用。

    水平 FOV = 2*atan(sensor_width / (2*focal))
    覆盖宽度 = 2 * distance * tan(FOV/2)
    GSD = 覆盖宽度 / image_width_px
    """
    if distance_m <= 0 or image_width_px <= 0:
        raise ValueError("distance_m 与 image_width_px 必须为正")
    fov = 2.0 * np.arctan(sensor_width_mm / (2.0 * focal_35mm))
    cover_mm = 2.0 * (distance_m * 1000.0) * np.tan(fov / 2.0)
    return Calibration(
        mm_per_px=cover_mm / image_width_px,
        method="camera_params",
        confidence="low",
        detail={"distance_m": distance_m, "focal_35mm": focal_35mm,
                "sensor_width_mm": sensor_width_mm,
                "image_width_px": image_width_px,
                "fov_deg": round(float(np.degrees(fov)), 2),
                "cover_mm": round(float(cover_mm), 1)},
    )


# --------------------------------------------------------------------------
# 可判读性判定 —— 本模块的核心函数
# --------------------------------------------------------------------------
def assess_interpretability(calib: Calibration,
                            target_mm: float = 0.30,
                            min_px: float = MIN_RELIABLE_PX,
                            comfortable_px: float = COMFORTABLE_PX) -> Interpretability:
    """
    判定「在当前标定下，能不能可靠判断 target_mm 这个尺寸」。

    target_mm 的典型取值：
      0.2  —— 二类环境裂缝限值（GB 50010）
      0.3  —— 一类环境裂缝限值
      0.5  —— JGJ 125 梁板受力主筋处危险判据
      10   —— 1cm 剥落（筛查层目标，厘米量级）
      100  —— 10cm 空鼓/脱落（筛查层，最宽松）
    """
    px_on_target = target_mm / calib.mm_per_px
    min_readable = min_px * calib.mm_per_px
    comfort = comfortable_px * calib.mm_per_px

    if px_on_target >= comfortable_px:
        level, feasible = "measurement", True
        msg = (f"可精测：目标 {target_mm:g}mm 在当前图像上约 {px_on_target:.1f}px，"
               f"满足量化门槛（>={comfortable_px:g}px），可给出毫米级结论。")
        advice = "无需调整，可直接量化并出分级结论。"
    elif px_on_target >= min_px:
        level, feasible = "screening", True
        msg = (f"仅可筛查：目标 {target_mm:g}mm 占 {px_on_target:.1f}px，"
               f"达到判读下限（>={min_px:g}px）但不满足量化门槛"
               f"（>={comfortable_px:g}px），只能定性提示存在性，不宜报具体宽度。")
        advice = (f"如需毫米级结论，请靠近拍摄："
                  f"目标距离需缩短至当前的 {px_on_target / comfortable_px:.2f} 倍以内，"
                  f"或换用长焦/更高像素相机。")
    else:
        level, feasible = "insufficient", False
        need_shrink = comfortable_px / max(px_on_target, 1e-9)
        msg = (f"不可判读：目标 {target_mm:g}mm 仅占 {px_on_target:.2f}px，"
               f"低于判读下限（{min_px:g}px）。系统拒绝就此给出结论。")
        advice = (f"当前仅能可靠测量 >= {min_readable:.1f}mm 的缺陷。"
                  f"若必须判读 {target_mm:g}mm，拍摄距离需缩短至约 1/{need_shrink:.1f}，"
                  f"或改用无人机贴近/搭架人工检查——这属于「鉴定层」工作，"
                  f"不在本筛查系统的能力范围内。")

    return Interpretability(
        feasible=feasible,
        min_readable_mm=round(min_readable, 3),
        comfort_mm=round(comfort, 3),
        gsd_mm_per_px=round(calib.mm_per_px, 4),
        target_mm=target_mm,
        pixels_on_target=round(px_on_target, 2),
        level=level,
        message=msg,
        advice=advice,
    )


def screening_capability(calib: Calibration, min_px: float = MIN_RELIABLE_PX) -> dict:
    """
    回答「这套图像配置，筛查层能做哪些事」——直接对应政策分工。
    这是答辩时最有说服力的一张表：诚实说明能力边界。
    """
    items = {
        "大面积剥落(>100mm)": 100.0,
        "中等剥落(>50mm)": 50.0,
        "小剥落(>10mm)": 10.0,
        "露筋(钢筋直径10-25mm)": 12.0,
        "明显裂缝(>1.0mm)": 1.0,
        "裂缝定级(0.5mm)": 0.5,
        "裂缝定级(0.3mm)": 0.30,
        "裂缝定级(0.2mm)": 0.20,
    }
    out = {}
    for name, mm in items.items():
        px = mm / calib.mm_per_px
        out[name] = {
            "target_mm": mm,
            "pixels": round(px, 2),
            "usable": px >= min_px,
            "level": ("measurement" if px >= COMFORTABLE_PX
                      else ("screening" if px >= min_px else "insufficient")),
        }
    return {
        "calibration": asdict(calib),
        "min_px_threshold": min_px,
        "comfortable_px_threshold": COMFORTABLE_PX,
        "capability": out,
    }


# --------------------------------------------------------------------------
# 图像质量自检（成像维度）—— 2026-09-23 新增
# --------------------------------------------------------------------------
# ★ 为什么需要「第二个维度」：
#   `assess_interpretability()` 只看「目标尺寸 ÷ GSD」（**几何**），
#   对图像质量**完全不敏感** —— 同一张图无论多暗多糊，它给出的结论一模一样。
#   而实测（logs/_IMAGE_QUALITY_GATE.md，4 类退化 × 3 档 = 12 配置）显示：
#     · 几何判据对**全部 12 档**都说「可判读」；
#     · 但**低照度 L1.0 的 mAP50 从 0.723 崩到 0.1586**、过曝 L1.0 掉到 0.3044
#       —— 即「几何合格」的图可以**完全不可用**。
#   ⇒ 只靠几何判据会在一张近乎全黑的照片上说「可判读」。必须补上成像维度。
#
# 门槛怎么来的（可复现）：
#   逐退化族找「可用(mAP50>=0.5) → 不可用」的那一档，取两侧指标中点作候选；
#   再过一道**「不得误杀任何可用档」的自动校验** —— 该校验当场剔除了 3 个站不住
#   的门槛（含一个自相矛盾的 min>max），最终只采纳下面三个。
#
# ⚠️ 诚实边界：门槛由**仿真退化**导出，须在真实照片上复核后才可上线（同 §7.3）；
#    且是**工程经验门槛，不是国标**。
QUALITY_BRIGHTNESS_MIN: float = 57.0     # 低于此判「过暗」
QUALITY_BRIGHTNESS_MAX: float = 230.0    # 高于此判「过曝」
QUALITY_NOISE_MAD_MAX: float = 4.22      # 高于此判「高 ISO 噪点」


@dataclass
class ImageQuality:
    """输入图的客观质量自检结果。"""
    brightness: float = 0.0
    contrast_std: float = 0.0
    sharpness_lapvar: float = 0.0
    noise_mad: float = 0.0
    ok: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def assess_image_quality(image: "np.ndarray") -> ImageQuality:
    """对输入图做客观质量自检 ——「这张图的成像够不够格让我们下结论」。

    与 `assess_interpretability()` 的分工：
      · `assess_interpretability` 回答「**这张图配不配测毫米**」（几何 / 分辨率）
      · `assess_image_quality`   回答「**这张图拍清楚了没有**」（成像 / 光照噪声）
    两者都通过，才允许给定量结论。
    """
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gf = g.astype(np.float32)
    lap = cv2.Laplacian(gf, cv2.CV_32F)
    med = cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
    resid = gf - med
    mad = float(np.median(np.abs(resid - np.median(resid))))
    brightness = float(gf.mean())
    contrast = float(gf.std())
    sharp = float(lap.var())

    notes = []
    if brightness < QUALITY_BRIGHTNESS_MIN:
        notes.append("过暗（亮度均值 %.0f < %.0f）"
                     % (brightness, QUALITY_BRIGHTNESS_MIN))
    if brightness > QUALITY_BRIGHTNESS_MAX:
        notes.append("过曝（亮度均值 %.0f > %.0f）"
                     % (brightness, QUALITY_BRIGHTNESS_MAX))
    if mad > QUALITY_NOISE_MAD_MAX:
        notes.append("高 ISO 噪点（噪声 MAD %.2f > %.2f）"
                     % (mad, QUALITY_NOISE_MAD_MAX))

    return ImageQuality(brightness=round(brightness, 2),
                        contrast_std=round(contrast, 2),
                        sharpness_lapvar=round(sharp, 2),
                        noise_mad=round(mad, 2),
                        ok=(len(notes) == 0),
                        note="；".join(notes))
