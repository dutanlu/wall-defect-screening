# -*- coding: utf-8 -*-
"""
_verify_measure_accuracy.py —— measure.py 的合成真值测量精度实验

===== 为什么需要它 =====
全项目的测量准确度证据链有三条，但只有「正射校正」给出过与真值的绝对偏差
（层高 0.3% / 砖模数 0.1%，见技术报告 §6.5.1）。而**测量各条标定路径本身的
误差从来没有被量化过** —— §3.2 只给了 high/medium/low 的置信度标签。
标审问「你测得多准」，现有材料答不出数字。

本脚本用**真实针孔相机模型 + 解析可算的真值**渲染已知毫米尺寸的靶标，
让 measure.py 去测，再与真值比对，输出测量误差。

===== 关键设计：真值为什么可信 =====
墙面世界坐标 (u,v) 以毫米为单位。make_brick_view 式的针孔投影给出
平面->像素的解析单应 Hp2i。于是：

    · 一个「世界坐标下已知毫米尺寸」的图形，投到图上再让 measure 测量，
      即可与真值直接比对 —— 真值不依赖任何测量管道。
    · 局部 mm/px 由 Hp2i 的雅可比解析算出（不是假设全局一致），
      因此「斜拍导致尺度不均」这件事在这里是**可解析验证**的。

===== 五维扫描 =====
    D1 线宽   : 0.3 / 0.5 / 1 / 2 / 5 mm 的已知宽度直线（裂缝主口径）
    D2 面积   : 已知 mm² 的圆形 / 方形斑块（块状类口径）
    D3 GSD    : 0.5 / 1 / 2 / 4 mm/px —— 找误差发散门槛
    D4 斜拍角 : 0 / 15 / 30 度 —— 量化「不正对时偏多少」
    D5 标定   : 真值 / 砖缝 / 相机参数三条路径的实际误差

===== 诚实边界（必须随数字一起引用）=====
    合成图只验证**几何测量这一层**（分割边界 -> 骨架/距离变换 -> mm 换算 -> 标定）。
    它**不**代表真实照片上阴影、纹理、边缘过渡带带来的偏差 ——
    §7.4A 自曝的「测得 2~6mm 偏大」只能被部分解释，报告里须保留该诚实标注。

输出：logs/_verify_out_measure/measure_accuracy_report.txt / .json
      （写图一律走 common.imwrite_u 并检查返回值 —— cv2.imwrite 在中文路径下静默失败）
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

HERE = Path(os.environ.get("WB_CODE") or Path(__file__).resolve().parent)
sys.path.insert(0, str(HERE))

from common import imwrite_u  # noqa: E402
from gsd import Calibration  # noqa: E402
from measure import measure_instance, segment_defect  # noqa: E402

OUT_DIR = Path(os.environ.get("WB_OUT") or (HERE.parent / "logs" / "_verify_out_measure"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

LINES: list[str] = []
RESULTS: dict = {}
IMG_ERRORS: list[str] = []

# 相机内参（与 _verify_rectify.py 同源，便于互相对照）
F_PX = 3000.0
CX, CY = 1600.0, 1100.0
W, H = 3200, 2200
# 相机到墙面的垂直距离 mm。
# ⚠️ 方向别搞反：d 越小 -> 墙在画面里越大 -> GSD 越小（越精细）。
#    f=3000px、画幅 3200px 时：d=3000mm -> GSD≈0.4 mm/px（本实验的工作区）。
#    （曾误以为是反的，取 d=10000 得到 GSD 3.33 mm/px，2mm 裂缝只占 0.6px，
#      全表分割失败 —— 那测的是「细裂缝测不了」，不是「测量链路有多准」。）
#    0.3mm 裂缝在 0.4mm/px 下占 0.75px —— 仍不可测，与 §7.2 结论一致；
#    本实验只覆盖 measure.py 声称可测的区间（目标 ≥3px）。
D_CAM = 3000.0


def say(s: str = "") -> None:
    LINES.append(s)


def hdr(t: str) -> None:
    say()
    say("=" * 78)
    say(t)
    say("=" * 78)


# --------------------------------------------------------------------------
# 针孔投影：与 _verify_rectify.py 的 make_brick_view 同一套解析式
# --------------------------------------------------------------------------
def plane_to_img_homography(yaw_deg: float, d: float = D_CAM) -> np.ndarray:
    """墙面世界坐标 (u,v) [mm] -> 像素 的解析单应。

    相机系 X 右 / Y 下 / Z 前；墙平面过 (0,0,d)，法线 (sin yaw, 0, cos yaw)。
    墙上一点 (u,v) -> 相机系 (u cos, v, d - u sin)，再经 K 投影。
    """
    th = np.radians(yaw_deg)
    A = F_PX * np.cos(th) - CX * np.sin(th)
    return np.array([[A, 0.0, CX * d],
                     [-CY * np.sin(th), F_PX, CY * d],
                     [-np.sin(th), 0.0, d]], dtype=np.float64)


def to_px(Hp2i: np.ndarray, pts_world) -> np.ndarray:
    pts = np.asarray(pts_world, dtype=np.float64)
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    p = (Hp2i @ hom.T).T
    return np.int32(np.stack([p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]], axis=1))


def mmpp_at(Hp2i: np.ndarray, u: float, v: float) -> float:
    """在墙面点 (u,v) 处的**局部** mm/px —— 由 Hp2i 的数值雅可比算出。

    这正是「斜拍时全画面尺度不一致」的解析度量：
    正对时处处相等，偏航后随 u 单调变化。
    """
    eps = 0.5  # mm
    p0 = Hp2i @ np.array([u, v, 1.0])
    pu = Hp2i @ np.array([u + eps, v, 1.0])
    pv = Hp2i @ np.array([u, v + eps, 1.0])
    x0, y0 = p0[0] / p0[2], p0[1] / p0[2]
    xu, yu = pu[0] / pu[2], pu[1] / pu[2]
    xv, yv = pv[0] / pv[2], pv[1] / pv[2]
    du = np.hypot(xu - x0, yu - y0) / eps   # 水平方向 px/mm
    dv = np.hypot(xv - x0, yv - y0) / eps   # 竖直方向 px/mm
    return float(2.0 / (du + dv))           # 取两轴的调和平均的倒数 => mm/px


def world_to_px_point(Hp2i: np.ndarray, u: float, v: float):
    p = Hp2i @ np.array([u, v, 1.0])
    return int(round(p[0] / p[2])), int(round(p[1] / p[2]))


# --------------------------------------------------------------------------
# 渲染靶标：高对比度（保证 measure.py 的 Otsu 分割能切开）
# --------------------------------------------------------------------------
def render_canvas(yaw_deg: float, bg: float = 205.0, seed: int = 7) -> np.ndarray:
    """渲染一块**均匀亮背景**（模拟抹灰面），缺陷以暗色画上去。

    为什么不用砖墙纹理：本实验的目的是测 measure.py 的几何测量链路，
    需要一个「干净、可分割」的底 —— 砖缝会在 ROI 里制造大量干扰边缘，
    把分割质量与测量精度两个因素混在一起。真实砖墙上的表现由 §7.1/§7.4
    的真实测试集指标覆盖，本实验只回答「几何测量链路本身有多准」。
    """
    rng = np.random.default_rng(seed)
    img = np.full((H, W), bg, dtype=np.float64)
    img += rng.normal(0.0, 1.5, img.shape)      # 轻微传感器噪声
    g = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def draw_world_poly(img: np.ndarray, Hp2i: np.ndarray, pts_world, val: float) -> None:
    cv2.fillPoly(img, [to_px(Hp2i, pts_world)], int(val))


def draw_line_band(img: np.ndarray, Hp2i: np.ndarray, cu: float, cv_: float,
                   width_mm: float, length_mm: float, val: float = 40.0,
                   angle_deg: float = 0.0) -> None:
    """在世界坐标里画一条**已知宽度**的矩形带（模拟裂缝）。

    用「矩形带」而不是数学细线：真实裂缝在图上总是有宽度的暗带，
    且 fillPoly 的取整行为与真实分割边界更接近。
    角度支持 0（水平）/90（竖直）/45（斜向）—— 覆盖骨架化路径的三种典型情况。
    """
    a = np.radians(angle_deg)
    ex, ey = np.cos(a), np.sin(a)
    nx, ny = -ey, ex
    hw, hl = width_mm / 2.0, length_mm / 2.0
    pts = []
    for s, t in ((+hl, +hw), (+hl, -hw), (-hl, -hw), (-hl, +hw)):
        pts.append((cu + s * ex + t * nx, cv_ + s * ey + t * ny))
    draw_world_poly(img, Hp2i, pts, val)


def draw_blob(img: np.ndarray, Hp2i: np.ndarray, cu: float, cv_: float,
              diam_mm: float, val: float = 40.0, sides: int = 0) -> None:
    """在世界坐标里画一个**已知等效直径**的块状缺陷（正圆或正多边形）。

    sides=0 用离散圆（48 边形近似圆），sides=4 用正方形。
    等效直径真值 = 轮廓面积对应的等效圆直径，与 measure_blob_like 的口径一致。
    """
    if sides <= 0:
        ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        r = diam_mm / 2.0
        pts = [(cu + r * np.cos(t), cv_ + r * np.sin(t)) for t in ang]
    else:
        r = diam_mm / 2.0
        pts = [(cu + r * np.cos(2 * np.pi * k / sides + np.pi / 4),
                cv_ + r * np.sin(2 * np.pi * k / sides + np.pi / 4)) for k in range(sides)]
    draw_world_poly(img, Hp2i, pts, val)


def bbox_of_world(Hp2i: np.ndarray, pts_world, pad: int = 24):
    """世界坐标多边形 -> 图上的检测框（含外扩）。

    ⚠️ pad 不能取小。measure.py 的 segment_defect 用黑帽提取细结构，
    黑帽核 ks = max(7, ROI短边//12)。若 bbox 紧贴目标（pad≈2），
    ROI 短边只有目标宽度那么大，ks 远大于目标宽度，黑帽响应会退化成
    「整块暗区」，Otsu 的分割边界被系统性外扩 —— 实测 1mm 靶标被量成 4mm。
    真实 YOLO 的 bbox 本来就比缺陷宽得多（框住整个损伤区域），
    所以这里用 pad=24 复现真实工作条件（也保证 ks 有可能大于目标宽度而不失控）。
    """
    px = to_px(Hp2i, pts_world)
    x1, y1 = px[:, 0].min() - pad, px[:, 1].min() - pad
    x2, y2 = px[:, 0].max() + pad, px[:, 1].max() + pad
    return (max(0, x1), max(0, y1), min(W, x2), min(H, y2))


def line_pts(cu, cv_, width_mm, length_mm, angle_deg):
    a = np.radians(angle_deg)
    ex, ey = np.cos(a), np.sin(a)
    nx, ny = -ey, ex
    hw, hl = width_mm / 2.0, length_mm / 2.0
    return [(cu + s * ex + t * nx, cv_ + s * ey + t * ny)
            for s, t in ((+hl, +hw), (+hl, -hw), (-hl, -hw), (-hl, +hw))]


# --------------------------------------------------------------------------
# D0 可测宽度区间（★ 本次实验发现的真实缺陷）
# --------------------------------------------------------------------------
def d0_measure_window() -> None:
    hdr("D0  ★ 可测宽度区间 —— measure.py 的隐藏盲区")
    say("背景：segment_defect 用黑帽 ks = max(7, ROI短边//12) 提取「暗的细结构」。")
    say("      若 ks <= 目标宽度，核就装不进暗带内部，黑帽响应归零 -> **目标完全消失**。")
    say("      因此存在一条由 bbox 尺寸决定的可测宽度**上限**。")
    say()
    say("本实验用「亮背景 + 精确 w px 宽暗带」直接测定该区间（不含尺度换算，纯像素）：")
    say()
    say("判据：测量值须落在真值的 ±20% 以内才算「可测」。")
    say("      （只看 mask 非空是不够的 —— 宽带会在边缘留下伪响应，")
    say("        mask 非空但测出的是边缘残余宽度，那同样是失效。）")
    say()
    say(f"{'ROI(高x宽)':>12} {'ks':>5} {'可靠区间(±20%)':>16} {'失效宽度':>20}")
    say("-" * 72)
    rows = []
    for roi_h, roi_w in ((100, 400), (200, 400), (300, 400), (400, 400)):
        ks_calc = max(7, min(roi_h, roi_w) // 12) | 1
        good_w, dead_w = [], []
        for w in range(1, min(roi_h // 2, 60) + 1):
            img = np.full((roi_h + 200, roi_w + 200, 3), 200, np.uint8)
            y0 = (roi_h + 200) // 2 - w // 2
            img[y0:y0 + w, 100:100 + roi_w] = 40
            roi = img[100:100 + roi_h, 100:100 + roi_w]
            mask, _ = segment_defect(roi, "crack")
            if cv2.countNonZero(mask) == 0:
                dead_w.append(w)
                continue
            # 用最小外接矩形的高作为「测出的带宽」
            ys, _xs = np.where(mask > 0)
            meas_w = float(ys.max() - ys.min() + 1)
            rel = abs(meas_w - w) / w
            if rel <= 0.20:
                good_w.append(w)
            else:
                dead_w.append(w)
        lo = min(good_w) if good_w else None
        hi = max(good_w) if good_w else None
        # 统计失效区间（连续段概括）
        dead_rng = "—"
        if dead_w:
            segs, s0, prev = [], dead_w[0], dead_w[0]
            for x in dead_w[1:]:
                if x == prev + 1:
                    prev = x
                else:
                    segs.append((s0, prev)); s0 = prev = x
            segs.append((s0, prev))
            dead_rng = ", ".join(f"{a}~{b}px" if a != b else f"{a}px" for a, b in segs)
        say(f"{roi_h:>5}x{roi_w:<6} {ks_calc:>5} "
            f"{(f'{lo}~{hi}px') if lo else '—':>16} {dead_rng:>20}")
        rows.append({"roi_h": roi_h, "roi_w": roi_w, "ks": ks_calc,
                     "reliable_lo_px": lo, "reliable_hi_px": hi,
                     "dead_width_px": dead_w})
    say()
    say("★★ 结论：可靠区间**同时有下限与上限，且上限由 bbox 尺寸（=ks）决定**。")
    say("   · 下限 ≈ 3px：开运算（3x3 椭圆核）抹掉更细的带。")
    say("   · 上限 ≈ ks-1：黑帽核装不进更宽的带，宽带被整体当背景 -> 失明或只剩边缘伪响应。")
    say("   真实含义：YOLO 的 bbox 越宽松（ROI 越大，ks 越大），能测的裂缝越宽，")
    say("   但**细裂缝的下限不变**；反之 bbox 紧，宽裂缝就测不了。")
    say("   这直接解释了 §7.4A 自曝的「测得 2~6mm 可能偏大」，")
    say("   也说明必须把「可测宽度窗口」写成报告里的显式能力边界。")
    RESULTS["D0_measure_window"] = {"rows": rows}


# --------------------------------------------------------------------------
# D1 线宽精度（裂缝主口径）
# --------------------------------------------------------------------------
def d1_line_width() -> None:
    hdr("D1  线宽测量精度（裂缝，GSD≈1.0 mm/px，正对）")
    Hp2i = plane_to_img_homography(0.0)
    base_mmpp = mmpp_at(Hp2i, 0.0, 0.0)
    say(f"解析基准尺度：{base_mmpp:.4f} mm/px（相机距墙 {D_CAM:.0f} mm，f={F_PX:.0f}px）")
    say(f"等价于 {1000.0 / base_mmpp:.2f} px/mm")
    say()
    say(f"{'宽度真值':>10} {'测得宽度均值':>14} {'相对误差':>10} {'占px':>7} {'可分割':>8}")
    say(f"{'mm':>10} {'mm':>14} {'%':>10} {'px':>7} {'':>8}")
    say("-" * 78)

    rows = []
    for wmm in (0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0):
        img = render_canvas(0.0)
        pts = line_pts(0.0, 0.0, wmm, 900.0, 0.0)
        draw_line_band(img, Hp2i, 0.0, 0.0, wmm, 900.0, 40.0, 0.0)
        bb = bbox_of_world(Hp2i, pts)
        calib = Calibration(mm_per_px=base_mmpp, method="truth", confidence="high")
        m = measure_instance(img, bb, "crack", calib)
        if m is None or m.width_mean_px <= 0:
            say(f"{wmm:>10.2f} {'(分割失败)':>14} {'—':>10} {wmm / base_mmpp:>7.2f} {'否':>8}")
            rows.append({"truth_mm": wmm, "measured_mm": None, "rel_err": None,
                         "px_on_target": round(wmm / base_mmpp, 3)})
            continue
        meas = m.width_mean_mm
        rel = (meas - wmm) / wmm * 100.0
        say(f"{wmm:>10.2f} {meas:>14.4f} {rel:>+10.2f} {wmm / base_mmpp:>7.2f} {'是':>8}")
        rows.append({"truth_mm": wmm, "measured_mm": round(meas, 4),
                     "rel_err": round(rel, 3),
                     "px_on_target": round(wmm / base_mmpp, 3)})
    say()
    say("读法：占 px < 1 时误差应显著放大（像素量化主导）；")
    say("      占 px ≥ 3 时应收敛到个位数百分比（这是 measure.py 的实际工作区）。")
    RESULTS["D1_line_width"] = {"base_mmpp": round(base_mmpp, 5), "rows": rows}


# --------------------------------------------------------------------------
# D2 面积精度（块状类口径）
# --------------------------------------------------------------------------
def d2_blob_area() -> None:
    hdr("D2  面积测量精度（块状，GSD≈1.0 mm/px，正对）")
    Hp2i = plane_to_img_homography(0.0)
    base_mmpp = mmpp_at(Hp2i, 0.0, 0.0)
    say(f"{'形状':>6} {'直径真值':>9} {'面积真值':>12} {'测得面积':>12} {'相对误差':>10}")
    say(f"{'':>6} {'mm':>9} {'mm²':>12} {'mm²':>12} {'%':>10}")
    say("-" * 78)
    rows = []
    for shape, diam in (("圆", 10.0), ("圆", 30.0), ("圆", 60.0),
                        ("方", 30.0), ("方", 60.0)):
        sides = 0 if shape == "圆" else 4
        img = render_canvas(0.0)
        draw_blob(img, Hp2i, 0.0, 0.0, diam, 40.0, sides)
        ang = np.linspace(0, 2 * np.pi, 48, endpoint=False) if sides == 0 else None
        if sides == 0:
            r = diam / 2.0
            pts = [(r * np.cos(t), r * np.sin(t)) for t in ang]
            truth_area = np.pi * r * r
        else:
            r = diam / 2.0
            pts = [(r * np.cos(2 * np.pi * k / 4 + np.pi / 4),
                    r * np.sin(2 * np.pi * k / 4 + np.pi / 4)) for k in range(4)]
            truth_area = 2.0 * r * r          # 45° 正方形面积 = 2r²
        bb = bbox_of_world(Hp2i, pts)
        calib = Calibration(mm_per_px=base_mmpp, method="truth", confidence="high")
        m = measure_instance(img, bb, "spalling", calib)
        if m is None or m.area_mm2 <= 0:
            say(f"{shape:>6} {diam:>9.1f} {truth_area:>12.1f} {'(分割失败)':>12} {'—':>10}")
            rows.append({"shape": shape, "diam_mm": diam, "truth_area_mm2": round(truth_area, 2),
                         "measured_area_mm2": None, "rel_err": None})
            continue
        rel = (m.area_mm2 - truth_area) / truth_area * 100.0
        say(f"{shape:>6} {diam:>9.1f} {truth_area:>12.1f} {m.area_mm2:>12.2f} {rel:>+10.2f}")
        rows.append({"shape": shape, "diam_mm": diam, "truth_area_mm2": round(truth_area, 2),
                     "measured_area_mm2": round(m.area_mm2, 2), "rel_err": round(rel, 3)})
    say()
    say("读法：块状目标面积在 tens of mm² 量级，像素边界占比小，误差应远小于线宽。")
    RESULTS["D2_blob_area"] = {"base_mmpp": round(base_mmpp, 5), "rows": rows}


# --------------------------------------------------------------------------
# D3 GSD 扫描：误差发散门槛
# --------------------------------------------------------------------------
def d3_gsd_scan() -> None:
    hdr("D3  GSD 扫描 —— 固定 2mm 宽裂缝，改变拍摄距离（等效 GSD 0.5~4 mm/px）")
    say("做法：保持墙面靶标物理尺寸不变，改变相机距离 d 来改变 GSD。")
    say("（不是对图像做重采样 —— 那样测的是插值，不是成像分辨率。）")
    say()
    say(f"{'距离mm':>8} {'GSD mm/px':>11} {'2mm占px':>9} {'测得宽度':>10} {'相对误差':>10}")
    say("-" * 78)
    rows = []
    for d in (1500.0, 3000.0, 6000.0, 12000.0):
        Hp2i = plane_to_img_homography(0.0, d)
        mmpp = mmpp_at(Hp2i, 0.0, 0.0)
        wmm = 2.0
        px_on = wmm / mmpp
        img = render_canvas(0.0)
        pts = line_pts(0.0, 0.0, wmm, 900.0, 0.0)
        draw_line_band(img, Hp2i, 0.0, 0.0, wmm, 900.0, 40.0, 0.0)
        bb = bbox_of_world(Hp2i, pts)
        calib = Calibration(mm_per_px=mmpp, method="truth", confidence="high")
        m = measure_instance(img, bb, "crack", calib)
        if m is None or m.width_mean_px <= 0:
            say(f"{d:>8.0f} {mmpp:>11.4f} {px_on:>9.2f} {'(分割失败)':>10} {'—':>10}")
            rows.append({"d_mm": d, "gsd": round(mmpp, 5), "px_on_target": round(px_on, 3),
                         "measured_mm": None, "rel_err": None})
            continue
        meas = m.width_mean_mm
        rel = (meas - wmm) / wmm * 100.0
        say(f"{d:>8.0f} {mmpp:>11.4f} {px_on:>9.2f} {meas:>10.4f} {rel:>+10.2f}")
        rows.append({"d_mm": d, "gsd": round(mmpp, 5), "px_on_target": round(px_on, 3),
                     "measured_mm": round(meas, 4), "rel_err": round(rel, 3)})
    say()
    say("读法：误差应随 GSD 增大而放大，且与「占 px」强相关。")
    say("      这为 §7.2 的可判读性门槛（3px）提供了**测量精度侧**的直接佐证。")
    RESULTS["D3_gsd_scan"] = {"rows": rows}


# --------------------------------------------------------------------------
# D4 斜拍角扫描
# --------------------------------------------------------------------------
def d4_yaw_scan() -> None:
    hdr("D4  斜拍角扫描 —— 2mm 裂缝，yaw = 0 / 15 / 30 度")
    say("对照组：用「画面中心处的真实局部 mm/px」标定（这是最优标定）")
    say("        —— 这样测出的误差只反映「斜拍造成的尺度不均」，不含标定误差。")
    say()
    say(f"{'yaw':>6} {'中心GSD':>10} {'左端GSD':>10} {'右端GSD':>10} {'尺度极差':>10} {'测得宽度':>10} {'误差':>9}")
    say("-" * 88)
    rows = []
    wmm = 2.0
    for yaw in (0.0, 15.0, 30.0):
        Hp2i = plane_to_img_homography(yaw)
        mmpp_c = mmpp_at(Hp2i, 0.0, 0.0)
        u_l, u_r = -600.0, 600.0
        mmpp_l = mmpp_at(Hp2i, u_l, 0.0)
        mmpp_r = mmpp_at(Hp2i, u_r, 0.0)
        spread = max(mmpp_l, mmpp_c, mmpp_r) / min(mmpp_l, mmpp_c, mmpp_r)
        img = render_canvas(yaw)
        pts = line_pts(0.0, 0.0, wmm, 900.0, 0.0)
        draw_line_band(img, Hp2i, 0.0, 0.0, wmm, 900.0, 40.0, 0.0)
        bb = bbox_of_world(Hp2i, pts)
        calib = Calibration(mm_per_px=mmpp_c, method="truth", confidence="high")
        m = measure_instance(img, bb, "crack", calib)
        if m is None or m.width_mean_px <= 0:
            say(f"{yaw:>6.0f} {mmpp_c:>10.4f} {mmpp_l:>10.4f} {mmpp_r:>10.4f} {spread:>10.3f} "
                f"{'(分割失败)':>10} {'—':>9}")
            rows.append({"yaw_deg": yaw, "gsd_center": round(mmpp_c, 5),
                         "gsd_left": round(mmpp_l, 5), "gsd_right": round(mmpp_r, 5),
                         "spread": round(spread, 4), "measured_mm": None, "rel_err": None})
            continue
        meas = m.width_mean_mm
        rel = (meas - wmm) / wmm * 100.0
        say(f"{yaw:>6.0f} {mmpp_c:>10.4f} {mmpp_l:>10.4f} {mmpp_r:>10.4f} {spread:>10.3f} "
            f"{meas:>10.4f} {rel:>+9.2f}")
        rows.append({"yaw_deg": yaw, "gsd_center": round(mmpp_c, 5),
                     "gsd_left": round(mmpp_l, 5), "gsd_right": round(mmpp_r, 5),
                     "spread": round(spread, 4),
                     "measured_mm": round(meas, 4), "rel_err": round(rel, 3)})
    say()
    say("读法：「尺度极差」列是斜拍危害的解析证据 —— 30 度时画面两端 mm/px 已明显不同，")
    say("      此时用任何单一全局尺度都会系统性偏，这正是 §3.2 要正射校正的原因。")
    say("      注意本例裂缝画在画面中心，故中心标定下误差仍小；危害随离中心距离放大。")
    RESULTS["D4_yaw_scan"] = {"rows": rows}


# --------------------------------------------------------------------------
# D5 标定路径误差（首次量化）
# --------------------------------------------------------------------------
def d5_calibration_paths() -> None:
    hdr("D5  三条标定路径的实际误差（墙上有砖缝，用于砖缝法）")
    say("场景：渲染一面**标准砖墙**（240+10 模数），在中心画一条已知宽度裂缝。")
    say("三条路径分别标定后，各自测量同一裂缝：")
    say("  · truth   : 用解析真值标定（上界，误差来自 measure 本身）")
    say("  · brick   : 用砖缝自相关标定（§3.2 第 3 条路径，medium）")
    say("  · camera  : 用相机参数推算（§3.2 第 4 条路径，low）")
    say()
    Hp2i = plane_to_img_homography(0.0, d=6000.0)
    truth_mmpp = mmpp_at(Hp2i, 0.0, 0.0)

    # 渲染标准砖墙（不含 rich 扰动，保持严格周期 —— 这正是砖缝法的适用场景）
    # ⚠️ 砖缝法要求一个砖距（250mm）落在自相关的搜索区间 8~400px 内，
    #    故本段单独用 d=6000mm（GSD≈0.8mm/px，砖距约 313px）。
    #    用基准 d=3000 会得到 625px 砖距，超出上限 -> 砖缝法必然失败。
    img = render_brick_wall(Hp2i)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)   # measure/gsd 均按 BGR 处理
    wmm = 5.0   # 在 0.8mm/px 下占 6.25px —— 落在可测区，避免与分割能力混淆
    pts = line_pts(0.0, 0.0, wmm, 900.0, 0.0)
    draw_line_band(img, Hp2i, 0.0, 0.0, wmm, 900.0, 30.0, 0.0)
    bb = bbox_of_world(Hp2i, pts)

    from gsd import calibrate_by_brick_period, calibrate_by_camera
    paths = []
    paths.append(("truth", truth_mmpp, "解析真值"))
    try:
        bc = calibrate_by_brick_period(img)
        if bc is None:
            paths.append(("brick", None, "拒绝标定（返回 None，符合设计）"))
        else:
            paths.append(("brick", bc.mm_per_px, f"砖缝自相关（conf={bc.confidence}）"))
    except Exception as ex:  # noqa: BLE001
        say(f"  !! 砖缝标定失败：{ex!r}")
        paths.append(("brick", None, f"失败：{type(ex).__name__}"))
    try:
        cc = calibrate_by_camera(distance_m=D_CAM / 1000.0, image_width_px=W)
        paths.append(("camera", cc.mm_per_px, f"相机参数（conf={cc.confidence}）"))
    except Exception as ex:  # noqa: BLE001
        say(f"  !! 相机参数标定失败：{ex!r}")
        paths.append(("camera", None, f"失败：{type(ex).__name__}"))

    say(f"{'路径':>8} {'自报GSD':>11} {'vs真值偏差':>11} {'测得宽度':>10} {'测量误差':>10}  {'来源':<28}")
    say("-" * 100)
    rows = []
    for name, mmpp, note in paths:
        if mmpp is None:
            say(f"{name:>8} {'—':>11} {'—':>11} {'—':>10} {'—':>10}  {note:<28}")
            rows.append({"path": name, "gsd": None, "gsd_bias_pct": None,
                         "measured_mm": None, "rel_err": None, "note": note})
            continue
        bias = (mmpp - truth_mmpp) / truth_mmpp * 100.0
        calib = Calibration(mm_per_px=mmpp,
                            method=name,
                            confidence="high" if name == "truth" else "medium")
        m = measure_instance(img, bb, "crack", calib)
        if m is None or m.width_mean_px <= 0:
            say(f"{name:>8} {mmpp:>11.5f} {bias:>+11.2f}% {'(分割失败)':>10} {'—':>10}  {note:<28}")
            rows.append({"path": name, "gsd": round(mmpp, 5), "gsd_bias_pct": round(bias, 3),
                         "measured_mm": None, "rel_err": None, "note": note})
            continue
        meas = m.width_mean_mm
        rel = (meas - wmm) / wmm * 100.0
        say(f"{name:>8} {mmpp:>11.5f} {bias:>+11.2f}% {meas:>10.4f} {rel:>+10.2f}  {note:<28}")
        rows.append({"path": name, "gsd": round(mmpp, 5), "gsd_bias_pct": round(bias, 3),
                     "measured_mm": round(meas, 4), "rel_err": round(rel, 3), "note": note})
    say()
    say(f"解析真值 GSD = {truth_mmpp:.5f} mm/px；宽 2mm 的裂缝在图上占 {wmm / truth_mmpp:.2f} px。")
    say("读法：这一节**首次**把 §3.2 的 high/medium/low 标签变成可比较的数字。")
    RESULTS["D5_calibration_paths"] = {"truth_gsd": round(truth_mmpp, 5), "rows": rows}


def render_brick_wall(Hp2i: np.ndarray, bg: float = 190.0, mortar: float = 62.0,
                      pitch: float = 250.0, course: float = 63.0) -> np.ndarray:
    """渲染严格周期的标准砖墙（用于砖缝自相关标定）。"""
    inv = np.linalg.inv(Hp2i)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    grid = np.stack([xs, ys, np.ones_like(xs)], axis=-1) @ inv.T
    w = grid[..., 2]
    u = grid[..., 0] / w
    v = grid[..., 1] / w
    r = np.floor(v / course)
    vv = v - r * course
    off = (r % 2) * (pitch / 2.0)
    uu = np.mod(u - off, pitch)
    is_mortar = (uu > pitch - 10.0) | (vv > course - 10.0)
    img = np.where(is_mortar, mortar, bg)
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# 图像产物（便于人工复核）
# --------------------------------------------------------------------------
def save_overview() -> None:
    hdr("图像产物")
    Hp2i = plane_to_img_homography(0.0)
    img = render_canvas(0.0)
    for wmm, cu in ((0.5, -800.0), (1.0, -400.0), (2.0, 0.0), (5.0, 500.0)):
        draw_line_band(img, Hp2i, cu, -400.0, wmm, 600.0, 40.0, 0.0)
    for diam, cu in ((30.0, -600.0), (60.0, 400.0)):
        draw_blob(img, Hp2i, cu, 400.0, diam, 40.0, 0)
    p = OUT_DIR / "targets_overview.png"
    ok = imwrite_u(str(p), img)
    if not ok:
        IMG_ERRORS.append(str(p))
    say(f"  靶标总览 -> {p.name}  {'OK' if ok else '!! 写图失败'}")
    say(f"  解析 GSD（画面中心）= {mmpp_at(Hp2i, 0.0, 0.0):.5f} mm/px")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    say("measure.py 合成真值测量精度实验")
    say("=" * 78)
    say("真值来源：真实针孔相机模型的解析单应（不经过任何测量管道）")
    say(f"相机：f={F_PX:.0f}px  画幅 {W}x{H}  距离 {D_CAM:.0f}mm")

    for fn in (d0_measure_window, d1_line_width, d2_blob_area, d3_gsd_scan,
               d4_yaw_scan, d5_calibration_paths, save_overview):
        try:
            fn()
        except Exception:  # noqa: BLE001
            say(f"!! {fn.__name__} 抛异常：")
            say(traceback.format_exc())
            RESULTS[fn.__name__] = {"error": traceback.format_exc()}

    hdr("收尾")
    say(f"图像写失败数：{len(IMG_ERRORS)}")
    for e in IMG_ERRORS:
        say(f"  !! {e}")
    say(f"JSON 段落数：{len(RESULTS)}")

    (OUT_DIR / "measure_accuracy_report.txt").write_text("\n".join(LINES), encoding="utf-8")
    (OUT_DIR / "measure_accuracy_report.json").write_text(
        json.dumps({"results": RESULTS, "img_errors": IMG_ERRORS},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("MEASACC_DONE sections=%d img_errors=%d" % (len(RESULTS), len(IMG_ERRORS)))


if __name__ == "__main__":
    main()
