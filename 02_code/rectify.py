# -*- coding: utf-8 -*-
"""
rectify.py —— 正射校正 与 多段拼接（补齐 §8.1 声明的缺失能力）

===== 为什么必须做这件事（不是锦上添花，是正确性前提）=====

measure.py 做量化时用的是**单一全局 mm_per_px**：整幅画面里 1 像素
代表同样的实际尺寸。这个前提只在「相机光轴垂直于墙面」时成立。

而实际拍摄（尤其是地面仰拍、斜拍）几乎必然带透视：
    靠近相机的一侧，1 像素代表更小的实际尺寸；
    远离相机的一侧，1 像素代表更大的实际尺寸。
于是同一道裂缝出现在画面上方和下方时，量出的宽度可以差出百分之几十。
**不做校正就报毫米结论，等于在给一个系统性偏大或偏小的数字。**
对我们的课题（判定 0.3mm / 0.2mm 这种毫米级阈值）这是致命的。

第二个动机是让既有能力真正可用：
gsd.calibrate_by_brick_period（砖缝周期标定）的文档里写明前提是
「拍摄方向大致正对墙面（严重透视会破坏周期间隔）」。
现实是网友寄来的照片大多不正对。**正射校正正是把这条前提补上**，
使砖缝标定从「只在理想照片上可用」变成「斜拍也能用」。

===== 两个能力 =====

1. rectify()          单张正射校正
2. stitch_segments()  多段拼接 → 整立面正射图

===== 方法（为什么这样选）=====

**用立面自身的线族求消失点，而不是找四角点。**
理由：外墙照片里很难找到一整块边界清晰的矩形面板，
但砖缝/砌块缝天然提供两组正交平行线族，几乎每张砖墙照片都有。
两组线族的消失点 vp_h、vp_v 构成 l∞ = vp_h × vp_v。

把两族摆正用的是**两点消失点对齐**：H = [vp_h | vp_v | l∞]^{-1}，
它同时把 l∞ 映到无穷远、把横缝摆成水平、把竖缝摆成竖直。
—— 不能只用 H_p = [[1,0,0],[0,1,0],l∞] 再加一个旋转：H_p 只保证仿射，
平面仍可带**剪切**，而旋转保角、摆不正两个并不垂直的方向。
实测旧做法在 A 帧里留下 10° 剪切，竖缝变斜，水平自相关退化成宽带信号，
砖距量得 12.28px 而真值 56.77px（-78%），尺度随之错 4.6 倍
（见 _verify_out/diag6.txt 的对照）。

摆正之后还剩「两个轴各自的尺度」（对角），再借**标准砖的已知模数**
（240+10=250mm 横向、53+10=63mm 纵向）测出两轴砖距像素数，
把纵横比校正到两轴同尺度，得到**全画面统一**的 mm/px ——
正是 measure.py 需要的东西。

砖距的像素测量用本模块的 `_modulus_px`（**逐行/逐列自相关再平均**），
**不是** gsd.calibrate_by_brick_period 的「先沿正交方向投影再自相关」口径。
原因：running bond（标准砌法）隔一皮错开半砖长，投影后竖缝并集成 125mm
周期，而真模数是 250mm —— 实测会量到 27.66px 而真值 55.15px，尺度错 2 倍。
逐行自相关不受影响，因为单行之内竖缝间距恒为 250mm，与砌法无关。

===== 设计原则（与全项目一致）=====

**能校就校；校不了要说清楚，绝不静默给一张没校过的图。**
一张没校正的图仍然"能出数"，但那个数是系统性错的，
且看起来完全正常 —— 这比直接失败危险得多。
所以本模块所有失败路径都返回 applied=False + 明确原因 + 改进建议，
交给上层决定是否拒答。
同理，`scale_mm_per_px` 只在**校正后两轴自检通过**时才给出；
不通过时为 None（刻意的拒答），而不是给一个"两轴平均值"糊过去。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from common import imread_u, imwrite_u, log
from gsd import calibrate_by_brick_period

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------
# 标准黏土砖 240x115x53 + 灰缝 10mm
#   横向模数 = 240 + 10 = 250mm（一个砖长 + 一条竖缝）
#   纵向模数 =  53 + 10 =  63mm（一个砖厚 + 一条横缝）
BRICK_PITCH_MM: float = 250.0
BRICK_COURSE_MM: float = 63.0

# 判定「已经足够正对、无需校正」的阈值（消失点距画面中心的距离，
# 以画面对角线为单位）。经验值，见下方 estimate_perspective 的说明。
ALREADY_FLAT_NORM: float = 12.0

# 消失点可解的下限。少于这个数，SVD 零空间无意义。
MIN_LINES_PER_FAMILY: int = 3

# 「这一族直线可信」的条数下限。刻意高于 MIN_LINES_PER_FAMILY：
# 实测伪纹理场景会出现「某族只有 8 条线」却仍能拟合出消失点的情况
# （低频噪声 1545 / 8），仅靠 3 条的下限拦不住。
MIN_LINES_CREDIBLE: int = 10

# 「这一族直线可信」的几何残差上限 —— 把拟合出的消失点代回每条直线：
#   消失点在有限处 -> 直线到该点的距离，单位像素
#   消失点在无穷远 -> 直线方向与共同方向的夹角，单位度
# 阈值来自合成真值实测（见 _verify_out/diag2.txt）：
#   真砖墙各线族      0.00 / 9.63 / 14.74 / 33.51 px
#   纯噪声            136.62 / 428.23 px
#   伪纹理            381.57 px
# 取 60px：距真墙上界留 1.8x 余量，距噪声下界留 2.3x 余量。
MAX_VP_RESIDUAL_PX: float = 60.0
MAX_VP_RESIDUAL_DEG: float = 8.0

# 输出画布长边上限，防止极端单应把画布撑到几百 MB
MAX_CANVAS_SIDE: int = 8000

# 校正后「两轴各自的 mm/px」允许的相对偏差上限。
# 只有在校正后两个轴重新量出的 mm/px 一致到这个程度，才敢宣称
# uniform_scale=True 并给出 scale_mm_per_px。
# 为什么必须看**校正后**：校正前两轴本来就不一致（那是待修正的各向异性本身），
# 拿它当质量指标会得出「残差 82.8% 但 uniform_scale=True」这种自相矛盾的结论。
MAX_AXIS_RESIDUAL: float = 0.10

# ---- 拼接时「位移分布」的取样与判据 ----
# 为什么 top_k 要取到 240 而不是 80：
#   砖墙是严格周期纹理，匹配位移会摊成几十个弱峰。实测同一对图
#   （1600 对匹配、约 50% 重叠、无旋转）取前 80 对只剩 32 个峰、次强峰 13 对；
#   取前 200 对能稳定看到 70~85 个峰。取样太浅会把"多峰"抹成"单峰模糊"，
#   于是把周期歧义误判成"拍糊了/不重叠"。
ALIAS_TOP_K: int = 240
# 最强位移峰要"像个真解"才配谈混叠：绝对支持 >=12 对、且占所考察匹配的 >=10%。
# 挡的是"两段压根不重叠"的情形 —— 那种情况连最强峰都是零星噪声，
# 不该套用"周期混叠"的说法。
ALIAS_MIN_TOP: int = 12
ALIAS_MIN_TOP_SHARE: float = 0.10
# 替代峰（与最强峰**同向**、但 dx 明显不同）的支持度下限，相对最强峰计。
# 取 0.12 而不是原来的 0.40：周期墙上真正的伪解往往被摊薄成多个小峰
# （实测 21/15/9 对，而最强真解 74 对），用 0.40 会把它们全部漏掉，
# 于是给出"不重叠或旋转"这种**归因错误**的建议。
# ⚠️ 这只是"值不值得去查"的入筛条件，**不是判据** —— 判据见 ALIAS_RIVAL_RATIO。
ALIAS_MIN_ALT_SHARE: float = 0.12
# 「同向」的判定容差。为什么用 dy 一致来筛：
#   周期砖墙相邻两段之间几乎是纯平移，所有"错位到第 k 个砖距"的伪解
#   都落在同一条水平线上；而"两段压根不重叠"产生的伪匹配 dy 是散乱的。
#   实测周期墙上前 5 个位移峰的 dy 全为 0.00 —— 这一点区分得极干净。
ALIAS_SAME_DIR_PX: float = 6.0
# 替代峰与最强峰在 dx 上至少差这么多像素才算"另一个解"
ALIAS_MIN_SEP_PX: float = 12.0
# ★ 真判据：把替代峰单独拟合成一个模型，看它在**全量匹配**上的内点数
# 相对已选中模型的内点数之比。>= 此值 就认为"另一个解同样能解释数据"，
# 于是对齐在原理上不可判，必须拒答。
# 为什么不能只看"两个位移峰各占 top-k 的百分之多少"（这条实测踩过）：
#   场景 A 最强峰 74% / 次强 11%（差 6.8 倍）—— 明显可判，却会被误拒；
#   场景 B 最强峰 17% / 次强 13%（差 1.3 倍）—— 真正的周期歧义。
#   用同一个 0.40 阈值既拦不住 B 也放不过 A。换成全量解释力之比后：
#   A ≈ 0.25（放行）、B ≈ 0.9（拒答），量纲一致、可比。
ALIAS_RIVAL_RATIO: float = 0.55
# 替代峰至少要这么多成员，才够单独拟合一个模型（少于 8 个不予判断）
ALIAS_MIN_RIVAL_SEED: int = 8

# ---- POS 先验仲裁周期歧义（2026-09-25 新增）----
# 当候选解与 H0 的四角偏离差 **>= 此值（像素）** 时，才认为 POS 能分辨两者。
# 定得太小：POS 抖动就会翻案；定得太大：真歧义没人来断。
# 项目实测（合成 S7/S8）：正确解的偏离 0.7~1.3px，错误解 851px，
# 二者相距三个数量级 ⇒ 20px 是个很宽的余量，不会误翻案。
POS_ARB_MIN_GAP_PX: float = 20.0
# 竞争解的内点数若超过已选解的该倍数，说明**图像证据压倒性**，
# 此时即便 POS 更偏向竞争解，也不许翻案（避免 POS 单方面推翻强证据）。
ALIAS_RIVAL_DOMINANT: float = 3.0


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclass
class RectifyResult:
    """正射校正结果。image 永远可用；applied 说明"到底校没校"。"""
    image: np.ndarray
    applied: bool                        # 是否真的施加了校正
    method: str                          # vp_metric / vp_anisotropic / vp_projective / passthrough
    confidence: Literal["high", "medium", "low", "none"]
    uniform_scale: bool                  # 是否已达成"全画面同一尺度"
    homography: list | None              # 3x3，JSON 友好
    scale_mm_per_px: float | None        # 仅在定尺度成功时给出
    message: str                         # 给人看的结论
    advice: str                          # 如何改进
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.detail)
        return {
            "applied": bool(self.applied),
            "method": self.method,
            "confidence": self.confidence,
            "uniform_scale": bool(self.uniform_scale),
            "homography": self.homography,
            "scale_mm_per_px": (None if self.scale_mm_per_px is None
                                else round(float(self.scale_mm_per_px), 5)),
            "message": self.message,
            "advice": self.advice,
            "detail": d,
        }


@dataclass
class StitchResult:
    """多段拼接结果。"""
    image: np.ndarray | None
    ok: bool
    n_input: int
    n_used: int
    method: str
    confidence: Literal["high", "medium", "low", "none"]
    message: str
    advice: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "n_input": int(self.n_input),
            "n_used": int(self.n_used),
            "method": self.method,
            "confidence": self.confidence,
            "message": self.message,
            "advice": self.advice,
            "detail": self.detail,
        }


# --------------------------------------------------------------------------
# 直线的齐次表示与消失点
# --------------------------------------------------------------------------
def _line_from_seg(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """两点确定直线 l=(a,b,c)，满足 ax+by+c=0。"""
    return np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=np.float64)


def _fit_line_family(lines: np.ndarray) -> dict:
    """
    由一族直线拟合消失点，并给出「这族线到底像不像一组共点直线」的判据。

    做法：所有直线都过消失点，故 vp 是 l 组成矩阵的零空间向量
    —— 对 A 做 SVD，取最小奇异值对应的右奇异向量。

    每行按 ||(a,b)|| 归一化，使 a*x+b*y+c 就是「点到直线的真实距离」，
    于是该 SVD 等价于最小化到所有直线的距离平方和，数值条件良好。
    实测对照解析真值（_verify_out/diag3.txt 问题1）：
        yaw=18° 误差  1.5px (0.04%)   25° 误差 6.5px (0.22%)
        yaw=8°（近共线）误差 159px (1.6%)
    —— 已经够好。**不要**再改成「整向量归一化」，实测那样反而恶化到 18.8%，
       也不要加坐标平移，实测只改善 0.02 个百分点却增加复杂度。

    ⚠️ 关键修正：**不再**因为最小奇异值 s[2] 极小就判失败。
       线族严格平行时 s[2]≈0，此时消失点恰在无穷远 ——
       这是**有效且极常见**的结果（正对拍摄；或只绕竖直轴偏航时的竖直族）。
       旧写法 s[2] < 1e-12 -> None，会让「已正对」与「yaw-only 斜拍」
       这两类最常见输入全部落进 homography_degenerate：实测一张有 44 条
       竖直线的斜拍砖墙被判「消失点退化，单应不可解」，整条校正路径不可用。
       真正退化的是「所有直线几乎重合」，那时 s[1] 也趋于 0。

    返回 dict：vp / kind / unit / residual(中位数) / residual_p90 / cond / n / status
    """
    out: dict = {"vp": None, "status": "ok",
                 "n": 0 if lines is None else int(len(lines)),
                 "kind": None, "unit": None, "residual": float("inf"),
                 "residual_p90": None, "cond": None}
    if lines is None or len(lines) < MIN_LINES_PER_FAMILY:
        out["status"] = "too_few_lines"
        return out
    norms = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    A = lines / norms
    try:
        _u, s, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        out["status"] = "svd_failed"
        return out
    if s.size < 3 or s[0] < 1e-12:
        out["status"] = "degenerate"
        return out
    if s[1] < 1e-12:
        # 连次小奇异值都消失 -> 所有直线几乎重合（不是「平行」而是「同一条」）
        out["status"] = "degenerate_all_collinear"
        return out
    vp = vt[-1]
    out["cond"] = (round(float(s[1] / s[2]), 2) if s[2] > 1e-12 else None)
    if abs(vp[2]) > 1e-9:
        p = vp[:2] / vp[2]
        res = np.abs(A @ np.array([p[0], p[1], 1.0]))
        out.update(kind="finite", unit="px")
    else:
        d = vp[:2]
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        # 直线 l=(a,b,c) 的方向是 (-b, a)
        dirs = np.stack([-A[:, 1], A[:, 0]], axis=1)
        dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
        res = np.degrees(np.arccos(np.clip(np.abs(dirs @ d), -1.0, 1.0)))
        out.update(kind="infinity", unit="deg")
    out["vp"] = vp
    out["residual"] = float(np.median(res))
    out["residual_p90"] = float(np.percentile(res, 90))
    return out


def _vanishing_point(lines: np.ndarray) -> tuple[np.ndarray | None, float]:
    """兼容旧接口：返回 (vp, 条件数)。新代码请直接用 _fit_line_family。"""
    f = _fit_line_family(lines)
    return f["vp"], float(f["cond"] or 0.0)


def _family_credible(fit: dict) -> bool:
    """这一族直线是否可信到足以支撑消失点与单应。"""
    if fit.get("status") != "ok" or fit.get("vp") is None:
        return False
    if fit.get("n", 0) < MIN_LINES_CREDIBLE:
        return False
    lim = MAX_VP_RESIDUAL_PX if fit.get("unit") == "px" else MAX_VP_RESIDUAL_DEG
    return float(fit.get("residual", float("inf"))) <= lim


def _split_line_families(gray: np.ndarray,
                         angle_tol: float = 22.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    检出直线段并按方向分成两组（近似竖直 / 近似水平）。

    用 HoughLinesP（不用 LSD：部分 OpenCV 构建里 LSD 已被移除，
    实测 cv2 5.0.0 上不可靠），Canny 阈值由中位数自适应，
    避免亮墙/暗墙需要手调。

    关键：**先做主方向聚类，再按与主方向的夹角分组**，
    而不是直接卡「竖直=90°±tol」。
    因为强烈透视会让竖线整体倾斜几十度，硬卡角度会把它们漏掉或串组。
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)
    med = float(np.median(enh))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(enh, lo, hi, apertureSize=3)

    h, w = gray.shape[:2]
    min_len = int(0.18 * max(h, w))
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=max(30, min_len // 3),
                           minLineLength=min_len, maxLineGap=max(6, min_len // 12))
    info: dict = {"n_segments_raw": 0, "n_vertical": 0, "n_horizontal": 0}
    if segs is None or len(segs) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), info

    # ⚠️ OpenCV 版本差异（实测踩坑，勿删这段兼容）：
    #     OpenCV <= 4.x : HoughLinesP 返回 (N, 1, 4)
    #     OpenCV 5.x    : 返回 (N, 4)      <-- 本机 cv2 5.0.0 实测 shape=(38,4)
    #   旧写法 segs[:, 0, :] 在 5.x 上直接 IndexError，
    #   会让 estimate_perspective / rectify 整条路径彻底不可用。
    segs = np.asarray(segs, dtype=np.float64)
    if segs.ndim == 3:
        segs = segs[:, 0, :]
    elif segs.ndim == 1:
        segs = segs.reshape(1, -1)
    if segs.ndim != 2 or segs.shape[1] != 4:
        return np.zeros((0, 3)), np.zeros((0, 3)), info
    info["n_segments_raw"] = int(len(segs))

    # 角度（0~180），并记录长度作权重
    ang, ln, lines = [], [], []
    for x1, y1, x2, y2 in segs:
        d = float(np.hypot(x2 - x1, y2 - y1))
        if d < 1e-6:
            continue
        a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
        ang.append(a)
        ln.append(d)
        lines.append(_line_from_seg(x1, y1, x2, y2))
    if not ang:
        return np.zeros((0, 3)), np.zeros((0, 3)), info
    ang = np.array(ang)
    ln = np.array(ln)
    lines = np.array(lines)

    # 主方向：按长度加权的圆形均值（角度按 2θ 折叠以处理 0/180 跨界）
    th = np.radians(ang * 2.0)
    wsum = ln.sum()
    if wsum <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), info
    mean2 = float(np.arctan2((ln * np.sin(th)).sum() / wsum,
                             (ln * np.cos(th)).sum() / wsum))
    dom = (np.degrees(mean2) / 2.0) % 180.0

    # 与主方向的夹角（折叠到 0~90），分成"同向"与"正交"
    diff = np.abs(((ang - dom + 90.0) % 180.0) - 90.0)      # 0..90
    is_par = diff <= angle_tol
    is_perp = np.abs(diff - 90.0) <= angle_tol

    # 主方向可能是竖直族也可能是水平族；按与图像坐标轴的接近程度定名
    def _is_vertical_family(d):
        return min(abs(d - 90.0), abs(d - 270.0), abs(d + 90.0)) < 45.0

    fam_a = lines[is_par]
    fam_b = lines[is_perp]
    if _is_vertical_family(dom):
        fam_v, fam_h = fam_a, fam_b
    else:
        fam_h, fam_v = fam_a, fam_b

    info["n_vertical"] = int(len(fam_v))
    info["n_horizontal"] = int(len(fam_h))
    info["dominant_angle_deg"] = round(dom, 2)
    info["canny"] = [lo, hi]
    return fam_v, fam_h, info


def _vp_2d_direction(vp: np.ndarray, center: np.ndarray) -> np.ndarray | None:
    """
    把齐次消失点转成"从画面中心看过去的方向"。
    vp 在无穷远（第三分量≈0）时没有确定位置，此时退回用其 (a,b) 方向。
    """
    if vp is None:
        return None
    if abs(vp[2]) > 1e-9:
        p = vp[:2] / vp[2]
        d = p - center
    else:
        d = vp[:2]
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return None
    return d / n


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _analyze_lines(gray: np.ndarray) -> dict:
    """
    检线 → 分族 → 拟合消失点 → 给出透视强度与「线族可信度」。

    **全模块唯一的 Hough 调用点。** 旧版 `estimate_perspective` 与 `rectify`
    各自调用一次 `_split_line_families`，同一张图白跑两遍 CLAHE+Canny+Hough
    （4000x3000 的照片上是可观浪费），且两次结果理论上可能不一致。

    可信度判据为什么**不是** `orthogonality_deg`（两族方向夹角）：
      实测纯噪声的两族夹角 = 89.4°（偏 90° 仅 0.6°），看起来「非常正交」，
      而真砖墙在该量上反而经常不可算。**该量毫无判别力，仅作参考输出。**
    为什么**也不是** SVD 条件数 cond=s1/s2：
      实测真砖墙 93.8，而纯噪声也能到 55.6（线数多时 SVD 总能凑出一个
      「差不多共点」的解）。**同样不可用。**
    真正有效的是**几何残差** —— 把拟合出的消失点代回每条直线，问
    「这个点真的落在每条线上吗」。实测分离度：真墙 0~33.5，噪声 137~428。
    """
    h, w = gray.shape[:2]
    diag = float(np.hypot(h, w))
    center = np.array([w / 2.0, h / 2.0])

    fam_v, fam_h, info = _split_line_families(gray)
    fit_v = _fit_line_family(fam_v)
    fit_h = _fit_line_family(fam_h)

    def _dist_norm(vp):
        if vp is None or abs(vp[2]) < 1e-9:
            return float("inf")            # 无穷远 = 已平行 = 无需校正
        p = vp[:2] / vp[2]
        return float(np.linalg.norm(p - center) / diag)

    dv, dh = _dist_norm(fit_v["vp"]), _dist_norm(fit_h["vp"])

    # 正交性核验：两族方向在画面中心处的夹角（**仅供参考，不作判据**，理由见上）
    dir_v = _vp_2d_direction(fit_v["vp"], center)
    dir_h = _vp_2d_direction(fit_h["vp"], center)
    ortho_deg = None
    if dir_v is not None and dir_h is not None:
        cosv = float(np.clip(abs(np.dot(dir_v, dir_h)), -1.0, 1.0))
        ortho_deg = round(float(np.degrees(np.arccos(cosv))), 2)

    cred_v, cred_h = _family_credible(fit_v), _family_credible(fit_h)
    credible = bool(cred_v and cred_h)
    worst = min(dv, dh)                    # 取两者中"更近"的那个
    flat = credible and worst > ALREADY_FLAT_NORM

    return {
        # 数组类中间产物（供 rectify 复用，不对外暴露）
        "fam_v": fam_v, "fam_h": fam_h, "fit_v": fit_v, "fit_h": fit_h,
        "vp_v": fit_v["vp"], "vp_h": fit_h["vp"],
        # 判据
        "enough_lines": bool(len(fam_v) >= MIN_LINES_PER_FAMILY
                             and len(fam_h) >= MIN_LINES_PER_FAMILY),
        "credible_v": cred_v, "credible_h": cred_h, "credible": credible,
        "already_flat": bool(flat),
        # 观测量
        "vp_v_dist_norm": (None if dv == float("inf") else round(dv, 3)),
        "vp_h_dist_norm": (None if dh == float("inf") else round(dh, 3)),
        "vp_nearer_dist_norm": (None if worst == float("inf") else round(worst, 3)),
        "orthogonality_deg": ortho_deg,
        "cond_v": fit_v["cond"], "cond_h": fit_h["cond"],
        "residual_v": (None if fit_v["vp"] is None else round(fit_v["residual"], 2)),
        "residual_h": (None if fit_h["vp"] is None else round(fit_h["residual"], 2)),
        "threshold": ALREADY_FLAT_NORM,
        **info,
    }


_ARRAY_KEYS = ("fam_v", "fam_h", "fit_v", "fit_h", "vp_v", "vp_h")


def estimate_perspective(image: np.ndarray) -> dict:
    """
    估计透视强度，回答「这张图是否已经足够正对、不需要校正」。

    判据 = 两个消失点各自的「距画面中心距离 / 画面对角线」。
    - 两个消失点都极远（归一化距离很大）→ 两组线各自近似平行
      → 说明视线已接近垂直于墙面 → 无需校正。
    - 任一消失点落在画面附近 → 该方向存在明显收敛 → 有透视，需校正。

    为什么用这个而不是「两组线夹角是否 90°」：夹角判据在近平行时
    非常不稳定（角度是距离的反正切，远处轻微扰动会让夹角乱跳），
    而距离判据在近处敏感、远处饱和，恰好符合我们的决策需求
    （我们真正关心的是"有没有明显收敛"，而不是精确角度）。
    > 补充实测结论：夹角判据不只是不稳定，而是**完全没有判别力** ——
    > 纯噪声的夹角 = 89.4°，看起来比真墙还「正交」。故仅作参考输出，
    > 不参与任何决策。

    另新增 `credible` 字段（两族直线的几何残差是否达标）。**调用方应先看它**：
    线族不可信时，后面的消失点、透视强度全是噪声上的数字。
    """
    a = _analyze_lines(_to_gray(image))
    return {k: v for k, v in a.items() if k not in _ARRAY_KEYS}


# --------------------------------------------------------------------------
# 单应构造
# --------------------------------------------------------------------------
def _homography_from_vps(vp_h: np.ndarray, vp_v: np.ndarray) -> np.ndarray | None:
    """
    构造把两族直线分别摆成「水平」「竖直」的仿射校正单应。

    做法（两点消失点对齐）：
        l_inf = vp_h × vp_v          平面在图像中的无穷远线
        Z     = [vp_h | vp_v | l_inf]   三列并排
        H     = Z^{-1}
    于是 vp_h -> (1,0,0)=e1、vp_v -> (0,1,0)=e2、l_inf -> (0,0,1)=e3：
        · l∞ 仍是无穷远线        -> 结果是**仿射**校正（不变形、保平行）
        · h 族（横缝）变成水平   -> 横缝水平
        · v 族（竖缝）变成竖直   -> 竖缝竖直
    剩下的自由度只有「两个轴各自的尺度」（对角矩阵），恰好是后面
    用两个已知砖模数（250mm / 63mm）要解的东西 —— 所以那一步在数学上必然可解。

    ⚠️ 为什么**不能**用「Hp + 一个旋转」的旧做法（本模块曾如此，已实测证伪）：
        Hp = [[1,0,0],[0,1,0],l∞] 只把 l∞ 映到无穷远，平面仍可带任意仿射畸变
        （含**剪切**）。旋转保角，只有当两个消失点方向本来就互相垂直时，
        才能把它们同时摆正；一般情况下做不到，于是 A 帧里残留下剪切。
        实测（_verify_out/diag6.txt，yaw=18° 斜拍砖墙）：
            旧做法：A 帧中竖缝中位倾角 99.88°（偏离竖直 10°），
                    水平自相关被小 lag 伪峰主导，量得 p_h=12.28px 而真值 56.77px（-78%）
            新做法：竖缝倾角 90.00°、横缝 0.00°，伪峰消失，p_h 偏差 -0.15%
        剪切为什么致命：竖缝一旦是斜的，沿 x 的投影就把一条缝的能量抹散到
        多个像素上，自相关退化成宽带信号 —— 周期量不准 -> 尺度算错 -> 报错毫米数。
        且这个错误**不会**体现在「两轴残差」之类的自检里，只会让人以为尺度已统一。
    """
    if vp_h is None or vp_v is None:
        return None
    l_inf = np.cross(vp_h, vp_v)
    if np.linalg.norm(l_inf) < 1e-12:
        return None
    Z = np.stack([np.asarray(vp_h, dtype=np.float64),
                  np.asarray(vp_v, dtype=np.float64),
                  l_inf], axis=1)
    # 逐列归一化：只改变各列的整体比例，不改变它代表的点/线
    # （齐次坐标下缩放无意义），但能显著改善远距离消失点带来的病态条件数。
    norms = np.linalg.norm(Z, axis=0, keepdims=True)
    if np.any(norms < 1e-12):
        return None
    Z = Z / norms
    if abs(np.linalg.det(Z)) < 1e-9:
        return None
    try:
        H = np.linalg.inv(Z)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def _warp_with_autosize(image: np.ndarray, H: np.ndarray
                        ) -> tuple[np.ndarray | None, dict]:
    """按单应变换整幅图，自动求输出画布并把画布控制在合理范围内。"""
    h, w = image.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    ones = np.ones((4, 1))
    pts = np.hstack([corners, ones]) @ H.T
    if np.any(np.abs(pts[:, 2]) < 1e-9):
        return None, {"error": "homography_singular_on_corners"}
    proj = pts[:, :2] / pts[:, 2:3]
    if not np.all(np.isfinite(proj)):
        return None, {"error": "non_finite_projection"}

    x0, y0 = proj[:, 0].min(), proj[:, 1].min()
    x1, y1 = proj[:, 0].max(), proj[:, 1].max()
    out_w, out_h = int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0))
    if out_w <= 1 or out_h <= 1:
        return None, {"error": "degenerate_canvas"}

    clamped = False
    scale = 1.0
    if max(out_w, out_h) > MAX_CANVAS_SIDE:
        scale = MAX_CANVAS_SIDE / float(max(out_w, out_h))
        clamped = True
        out_w, out_h = int(out_w * scale), int(out_h * scale)

    T = np.array([[scale, 0, -x0 * scale],
                  [0, scale, -y0 * scale],
                  [0, 0, 1.0]])
    warped = cv2.warpPerspective(image, T @ H, (out_w, out_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
    return warped, {"canvas": [out_w, out_h], "clamped": clamped,
                    "applied_scale": round(scale, 5)}


def _modulus_px(image: np.ndarray, axis: str,
                min_period: float = 6.0, max_period: float = 600.0,
                max_lines: int = 400) -> dict:
    """
    量「一个砖模数」占多少像素。axis='x' 量竖缝的水平间距，'y' 量横缝的竖直间距。

    做法：**逐条线做自相关，再把这些自相关曲线平均**，
    而不是先沿正交方向投影成 1D 曲线、再对那一条曲线做自相关。

    ⚠️ 为什么必须换掉「先投影再自相关」（gsd.calibrate_by_brick_period 的口径）：
        running bond（标准砌法）隔一皮错开半个砖长，于是把**所有皮叠在一起**
        投影时，竖缝位置并集成 125mm 周期 —— 而真模数是 250mm。
        实测（_verify_out/diag6.txt，yaw=18° 斜拍 running bond）：
            投影口径：量到 27.66px（=125mm），真值 55.15px（=250mm），**尺度直接错 2 倍**
            逐行口径：250mm 的峰被加强、125mm 的峰被抵消（见 diag7 对照）
        逐行口径没有这个问题，因为**单独一行之内，竖缝间距恒为 250mm，
        与砌法无关** —— 砌法只改变相邻两行之间的相位。
        （同理 axis='y' 上，逐列自相关也不受砌法影响；皮缝连续，本就无歧义。）

    返回 dict：period_px / peak / modulation / n_lines / status
    """
    gray = _to_gray(image)
    if gray is None or gray.size == 0:
        return {"period_px": None, "status": "empty", "peak": 0.0,
                "modulation": 0.0, "n_lines": 0}
    if axis == "x":
        mag = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        lines = mag                       # 逐行：每行是一条沿 x 的一维信号
    else:
        mag = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        lines = mag.T                     # 逐列
    n_all = lines.shape[0]
    if n_all == 0 or lines.shape[1] < 4 * min_period:
        return {"period_px": None, "status": "too_small", "peak": 0.0,
                "modulation": 0.0, "n_lines": 0}
    # 均匀抽样，避免整图太慢（400 条线已足够稳定）
    idx = np.unique(np.linspace(0, n_all - 1, min(max_lines, n_all)).astype(int))
    lo = int(min_period)
    hi = int(min(max_period, lines.shape[1] - 1))
    acc = np.zeros(hi + 1, dtype=np.float64)
    used = 0
    mods = []
    for i in idx:
        v = lines[i].astype(np.float64)
        v = v - v.mean()
        denom = float(v @ v)
        if denom <= 1e-9:
            continue
        ac = np.correlate(v, v, mode="full")[v.size - 1:][:hi + 1] / denom
        if not np.all(np.isfinite(ac)):
            continue
        acc += ac
        used += 1
        mods.append(float(v.std() / (np.abs(v).mean() + 1e-9)))
    if used == 0:
        return {"period_px": None, "status": "no_valid_lines", "peak": 0.0,
                "modulation": 0.0, "n_lines": 0}
    ac = acc / used

    seg = ac[lo:hi]
    if seg.size < 3:
        return {"period_px": None, "status": "too_small", "peak": 0.0,
                "modulation": 0.0, "n_lines": used}
    peaks = [(i + lo, float(seg[i])) for i in range(1, seg.size - 1)
             if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.10]
    if not peaks:
        return {"period_px": None, "status": "no_peak", "peak": 0.0,
                "modulation": round(float(np.median(mods)), 4), "n_lines": used}
    gmax = max(v for _, v in peaks)

    def peak_at(lag: float) -> float:
        j = int(round(lag)) - lo
        a, b = max(0, j - 2), min(seg.size, j + 3)
        return float(seg[a:b].max()) if b > a else 0.0

    # 基频 = 最小的显著峰（且其倍频处确有峰）。理由与 gsd 版一致：
    # 自相关同时出现基频与倍频，直接取全局最大容易抓到倍频。
    chosen = None
    for lag, val in sorted(peaks, key=lambda t: t[0]):
        if val < 0.55 * gmax:
            continue
        if peak_at(2 * lag) >= 0.4 * val or peak_at(3 * lag) >= 0.3 * val:
            chosen = (lag, val)
            break
    if chosen is None:
        chosen = max(peaks, key=lambda t: t[1])
    lag, peak = chosen

    if 0 < lag < ac.size - 1:
        y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
        d = (y0 - 2 * y1 + y2)
        if abs(d) > 1e-9:
            lag = lag + 0.5 * (y0 - y2) / d
    lag = float(lag)
    if lag <= 0:
        return {"period_px": None, "status": "bad_lag", "peak": peak,
                "modulation": round(float(np.median(mods)), 4), "n_lines": used}

    # 候选峰的完整列表（供调用方判断是否踩到半模数）
    alts = [{"lag": int(l), "ac": round(v, 4)} for l, v in
            sorted(peaks, key=lambda t: -t[1])[:5]]
    return {"period_px": round(lag, 4), "peak": round(peak, 4),
            "modulation": round(float(np.median(mods)), 4), "n_lines": used,
            "status": "ok", "top_peaks": alts}


# --------------------------------------------------------------------------
# 主入口：单张正射校正
# --------------------------------------------------------------------------
def rectify(image: np.ndarray,
            brick_pitch_mm: float = BRICK_PITCH_MM,
            brick_course_mm: float = BRICK_COURSE_MM,
            force: bool = False) -> RectifyResult:
    """
    对单张立面照片做正射校正。

    force=False（默认）：若判定"已足够正对"，原图返回，不做无谓插值
                        （重采样只会让细节变糊，对毫米级判读有害无益）。
    force=True         ：即使判定为正对也强行施加校正，用于对照实验。
    """
    if image is None or image.size == 0:
        return RectifyResult(
            image=image, applied=False, method="passthrough", confidence="none",
            uniform_scale=False, homography=None, scale_mm_per_px=None,
            message="输入图像为空，无法校正。", advice="请检查图像读取路径。",
            detail={"error": "empty_image"},
        )

    persp = _analyze_lines(_to_gray(image))
    pview = {k: v for k, v in persp.items() if k not in _ARRAY_KEYS}

    # ---- 情况 1a：连线都不够 ----
    if not persp["enough_lines"]:
        n_v, n_h = persp["n_vertical"], persp["n_horizontal"]
        # 子情形：一族极少甚至为 0、另一族很多。
        # 实测最典型的是**正对拍摄的 running bond（标准砌法）砖墙**：
        #   隔一皮错开半砖长 → 竖向灰缝被切成只有一皮高（正对时实测约 19px）
        #   的短线段，而本模块的 minLineLength = 0.18*max(h,w)（1600x1200 时
        #   = 288px），于是一条竖缝也检不出（实测 0 条竖 / 221 条横）。
        # 也就是说：**「检不出竖缝」不等于照片有问题**，反而常常说明
        # 视线已经接近垂直于墙面了。
        # 但这里必须诚实：竖缝没检出 -> 竖缝是否平行无从验证 -> 无法确认
        # 「全画面同一尺度」。所以仍然拒答（**不是**判为 already_flat），
        # 并把用户导向不依赖竖缝的正确工具（砖缝周期标定 / A4 标定物）。
        one_sided = ((n_v < MIN_LINES_PER_FAMILY) != (n_h < MIN_LINES_PER_FAMILY))
        if one_sided:
            lack = "竖向" if n_v < MIN_LINES_PER_FAMILY else "水平"
            have = "水平" if n_v < MIN_LINES_PER_FAMILY else "竖向"
            return RectifyResult(
                image=image, applied=False, method="passthrough", confidence="none",
                uniform_scale=False, homography=None, scale_mm_per_px=None,
                message=(f"无法校正：只检出了{have}线族（{max(n_v, n_h)} 条），"
                         f"{lack}线族只有 {min(n_v, n_h)} 条。"),
                advice=("这种「一族有、一族没有」最常见的原因是**正对拍摄标准砌法"
                        "（running bond）砖墙**：竖缝每隔一皮才对齐，被切成很短的线段，"
                        "长度不足检出阈值。若确实是正对拍摄，本图其实**不需要几何校正**，"
                        "直接做砖缝周期标定即可（或用 A4 纸标定）。"
                        "但本模块无法据此确认「全画面同一尺度」—— 竖缝没检出，"
                        "就没有证据说明它在竖直方向不收敛 —— 所以这里不下「已正对」的结论。"
                        "若照片是斜拍的，请让画面同时拍到两组砖缝方向，并补拍更完整的墙面。"),
                detail={"perspective": pview, "reason": "insufficient_lines_one_sided",
                        "couple_missing": lack, "hint": "possible_fronto_parallel_shot"},
            )
        return RectifyResult(
            image=image, applied=False, method="passthrough", confidence="none",
            uniform_scale=False, homography=None, scale_mm_per_px=None,
            message=("无法校正：未能从画面中检出足够的两组正交直线"
                     f"（竖直 {n_v} 条 / 水平 {n_h} 条，"
                     f"各需 >={MIN_LINES_PER_FAMILY} 条）。"),
            advice=("请重新拍摄：让砖缝或砌块缝在画面中清晰可见、"
                    "尽量拍到 3~4 个完整砖距，并避免大面积遮挡（空调外机、"
                    "绿植、广告牌会打断砖缝的连续性）。"),
            detail={"perspective": pview, "reason": "insufficient_lines"},
        )

    # ---- 情况 1b：线够多，但两族的几何残差过大 -> 这不是一面可信的砖墙 ----
    # 这一关必须有。Hough 在纯噪声图上也能凑出「两族各 3 条以上」的直线，
    # 旧版只数条数，于是对**纯噪声返回 applied=True + method=vp_projective**，
    # 输出一张「已做投影校正」的乱图 —— 正是本项目最不能接受的那类失败
    # （看起来正常、结论却是错的）。实测：噪声残差 137~428，真墙 0~33.5。
    if not persp["credible"]:
        bad = []
        if not persp["credible_v"]:
            bad.append(f"竖直族 {persp['n_vertical']} 条、残差 "
                       f"{persp['residual_v']}{persp['fit_v'].get('unit') or ''}"
                       f"（上限 {MAX_VP_RESIDUAL_PX:g}px / {MAX_VP_RESIDUAL_DEG:g}°、"
                       f"条数下限 {MIN_LINES_CREDIBLE}）")
        if not persp["credible_h"]:
            bad.append(f"水平族 {persp['n_horizontal']} 条、残差 "
                       f"{persp['residual_h']}{persp['fit_h'].get('unit') or ''}"
                       f"（上限 {MAX_VP_RESIDUAL_PX:g}px / {MAX_VP_RESIDUAL_DEG:g}°、"
                       f"条数下限 {MIN_LINES_CREDIBLE}）")
        return RectifyResult(
            image=image, applied=False, method="passthrough", confidence="none",
            uniform_scale=False, homography=None, scale_mm_per_px=None,
            message=("无法校正：检出的直线不构成可信的砖缝结构 —— "
                     + "；".join(bad) + "。"),
            advice=("这通常意味着画面里没有成片的砖/砌块墙面：可能是大面积平整抹灰、"
                    "涂料或幕墙；也可能是严重模糊、过曝，或根本不是在拍墙。"
                    "请换一张砖缝清晰、纹理连贯的照片，并确保画面中能同时看到"
                    "**横缝与竖缝两组方向**。"),
            detail={"perspective": pview, "reason": "lines_not_credible"},
        )

    vp_v, vp_h = persp["vp_v"], persp["vp_h"]

    # ---- 情况 2：已足够正对，不做无谓重采样 ----
    if persp["already_flat"] and not force:
        # vp_nearer_dist_norm 在「两族都平行」时是 None —— 那正是最正对的情形，
        # 直接说 None 会让人以为出了错，故分开措辞。
        nearer = persp["vp_nearer_dist_norm"]
        if nearer is None:
            why = "两族砖缝的消失点都在无穷远（两组线各自严格平行），是完全正对的情形"
        else:
            why = (f"两族消失点都在画面外侧很远处，较近者仍有 {nearer} 倍对角线距离")
        return RectifyResult(
            image=image, applied=False, method="passthrough", confidence="medium",
            uniform_scale=True,
            homography=None, scale_mm_per_px=None,
            message=(f"画面已接近正对墙面（{why}），"
                     "无需校正 —— 强行重采样只会让细节变糊。"),
            advice=("可直接进入标定环节。若想确认，可用砖缝周期标定交叉验证。"),
            detail={"perspective": persp, "reason": "already_flat"},
        )

    # ---- 情况 3：构造单应并施加校正 ----
    H = _homography_from_vps(vp_h, vp_v)
    if H is None:
        return RectifyResult(
            image=image, applied=False, method="passthrough", confidence="none",
            uniform_scale=False, homography=None, scale_mm_per_px=None,
            message="无法校正：消失点退化（两族直线接近平行或共点），单应不可解。",
            advice="请换一张砖缝结构更完整、视角更端正的照片。",
            detail={"perspective": persp, "reason": "homography_degenerate"},
        )

    warped, winfo = _warp_with_autosize(image, H)
    if warped is None:
        return RectifyResult(
            image=image, applied=False, method="passthrough", confidence="none",
            uniform_scale=False, homography=H.tolist(), scale_mm_per_px=None,
            message=f"无法校正：单应变换后画面退化（{winfo.get('error')}）。",
            advice="请换一张照片；这种几何下校正结果不可用。",
            detail={"perspective": persp, "reason": "warp_failed", **winfo},
        )

    # ---- 定尺度：用标准砖的两个模数把纵横比校到同一尺度 ----
    # 到这一步，图已被摆成「横缝水平、竖缝竖直」，平面->图的映射是仿射的，
    # 剩下的自由度只有 x、y 两个轴各自的尺度（对角）。用两个已知模数解出即可：
    #     mm_x = brick_pitch_mm  / p_h     （水平向：一个砖长占多少像素）
    #     mm_y = brick_course_mm / p_v     （竖直向：一皮砖高占多少像素）
    #     对 y 轴乘 s 后竖直向 mm/px 变成 mm_y/s，令其等于 mm_x 得 s = mm_y/mm_x。
    #     ⚠️ 实测踩坑：这里曾写成 S=diag(1, 1/k)（k=mm_y/mm_x），是**倒数** ——
    #        本该缩小的方向被放大，实测 y 被拉伸 5.81 倍，
    #        画布从 1304x1201 变成 1304x6975，自报尺度偏差 94%。
    #     量周期用 _modulus_px（逐行/逐列自相关），不用 gsd 的投影口径：
    #        后者在 running bond 上量到的是半模数 125mm 而非 250mm，尺度错 2 倍。
    m_h = _modulus_px(warped, "x", max_period=max(120.0, 4.0 * brick_pitch_mm))
    m_v = _modulus_px(warped, "y", max_period=max(120.0, 4.0 * brick_course_mm))
    p_h, p_v = m_h.get("period_px"), m_v.get("period_px")
    re_h: dict = {}
    re_v: dict = {}

    metric_ok = bool(p_h and p_v and p_h > 0 and p_v > 0)
    scale_mm = None
    resid = None
    warp2_ok = False
    s_applied = None
    mm_x = mm_y = None
    if metric_ok:
        mm_x = float(brick_pitch_mm) / float(p_h)
        mm_y = float(brick_course_mm) / float(p_v)
        s = mm_y / mm_x                       # y 轴像素缩放系数（见上）
        s_applied = float(s)
        S = np.array([[1.0, 0.0, 0.0],
                      [0.0, s, 0.0],
                      [0.0, 0.0, 1.0]])
        warped2, winfo2 = _warp_with_autosize(warped, S)
        if warped2 is not None:
            warped = warped2
            winfo = winfo2
            warp2_ok = True
            # 校正后重新量两个轴 —— **这才是真正的自检**：
            # 只有两轴各自推出的 mm/px 一致，才说明纵横比真的校对了。
            # （校正前两轴必然不一致，那个差值只是被修正的「各向异性」本身，
            #   不是质量指标。旧版把它当质量指标印出来，于是出现
            #   「自曝两轴残差 82.8% 却报 uniform_scale=True」的自相矛盾。）
            re_h = _modulus_px(warped, "x", max_period=max(120.0, 4.0 * brick_pitch_mm))
            re_v = _modulus_px(warped, "y", max_period=max(120.0, 4.0 * brick_course_mm))
            rp_h, rp_v = re_h.get("period_px"), re_v.get("period_px")
            if rp_h and rp_v:
                mm_after_x = float(brick_pitch_mm) / float(rp_h)
                mm_after_y = float(brick_course_mm) / float(rp_v)
                scale_mm = 0.5 * (mm_after_x + mm_after_y)
                resid = abs(mm_after_x - mm_after_y) / max(mm_after_x, 1e-9)

    # 判据：校正后两轴一致才敢说「全画面统一尺度」。不达标就**不给毫米数** ——
    # 报一个自己都不信的数字，正是本项目最不能接受的失败（看起来正常、结论是错的）。
    unified = bool(metric_ok and warp2_ok and resid is not None
                   and resid <= MAX_AXIS_RESIDUAL)
    if unified:
        conf = "high"
        msg = (f"已正射校正并定尺度：全画面统一 {scale_mm:.4f} mm/px。"
               f"砖距交叉验证 —— 水平模数 {p_h}px（{brick_pitch_mm:g}mm）、"
               f"竖直模数 {p_v}px（{brick_course_mm:g}mm，校正后 {re_v.get('period_px')}px），"
               f"校正后两轴残差 {resid:.1%}。")
        adv = "校正后 mm/px 在全画面成立，可直接作为量化的尺度输入。"
    elif metric_ok and warp2_ok and resid is not None:
        conf = "low"
        msg = (f"已做几何校正与纵横比修正，但**两轴尺度仍不一致**："
               f"水平推出 {float(brick_pitch_mm) / float(re_h['period_px']):.4f} mm/px、"
               f"竖直推出 {float(brick_course_mm) / float(re_v['period_px']):.4f} mm/px，"
               f"相差 {resid:.1%}（上限 {MAX_AXIS_RESIDUAL:.0%}）。")
        adv = ("两轴不一致说明砖模数假设与画面不符（砌块不是标准砖？"
               "或量到的周期踩到了半模数/倍频）。**此时不可报毫米结论**："
               "请用实测标定物（A4 纸）标定，或确认砌块规格后改传 "
               "brick_pitch_mm / brick_course_mm。")
    else:
        conf = "medium"
        miss = []
        if not p_h:
            miss.append(f"水平（{m_h.get('status')}）")
        if not p_v:
            miss.append(f"竖直（{m_v.get('status')}）")
        if p_h and p_v and not warp2_ok:
            miss.append("纵横比修正失败")
        msg = ("已做几何校正（砖缝已摆正），但**尺度未定**："
               f"{'、'.join(miss)}方向的砖缝周期未能量出，"
               "因此无法用砖模数把纵横比校到同一尺度。")
        adv = ("此时 mm/px 仍未统一，**不可直接报毫米结论**。"
               "请改用实测标定物（A4 纸）标定，或补拍一张砖缝更清晰的照片。")

    return RectifyResult(
        image=warped, applied=True,
        method=("vp_metric" if unified else
                ("vp_anisotropic" if (metric_ok and warp2_ok) else "vp_projective")),
        confidence=conf,
        uniform_scale=bool(unified),
        homography=H.tolist(),
        scale_mm_per_px=(scale_mm if unified else None),
        message=msg, advice=adv,
        detail={
            "perspective": persp,
            "canvas": winfo.get("canvas"),
            "canvas_clamped": winfo.get("clamped"),
            "modulus_h": m_h, "modulus_v": m_v,
            "modulus_h_after": re_h, "modulus_v_after": re_v,
            "mm_per_px_before_x": (None if mm_x is None else round(mm_x, 5)),
            "mm_per_px_before_y": (None if mm_y is None else round(mm_y, 5)),
            # 实际施加的 y 轴像素缩放系数。外部若要独立复现整条变换链
            # （plane -> 原图 -> A -> A' -> 最终图）必须用它，否则算不对。
            "y_scale_applied": (None if s_applied is None else round(s_applied, 6)),
            "axis_residual": (None if resid is None else round(resid, 4)),
            "note": ("scale_mm_per_px 只在两轴自检通过时才给出；"
                     "不通过时该字段为 None，是刻意的拒答而非遗漏。"),
            "path": "情况3 已施加校正",
        },
    )


# --------------------------------------------------------------------------
# 多段拼接
# --------------------------------------------------------------------------
def _displacement_clusters(kps_prev, kps_cur, matches,
                           top_k: int = ALIAS_TOP_K,
                           tol_px: float = 4.0) -> tuple[list[dict], list[list[int]]]:
    """
    把「描述子最像的前 top_k 对匹配」按位移 (dx,dy) 聚类。

    返回 (modes, members)：
        modes[i]   —— 第 i 簇的统计（支持数、位移、占比），按支持度降序
        members[i] —— 第 i 簇成员在**原 matches 列表**里的下标
                      （用于把"另一个解"的候选匹配取出来单独拟合模型）

    为什么需要这个量（而不是只看内点占比）：
        砖墙是**严格周期纹理**。"平移 0 个砖距"和"平移 1 个砖距"在特征空间里
        长得几乎一样，RANSAC 只会选中**内点最多**的那个模型，而那未必是物理
        正确的那一个。实测在一个含低频光照变化的真砖墙场景里，真值邻域内有
        151 对匹配、其中 136 对高度一致（残差 0.11px），单看占比完全正常，
        但伪周期模式的票数更多（403）。
        也就是说：**内点占比无法区分「压根不重叠」与「周期混叠」**，
        而这两种情况该给的处置完全不同。位移分布是否多峰，可以区分。

    参数 top_k 的取法很关键：砖墙是严格周期纹理，位移会摊成几十个弱峰，
        取样太浅会把"多峰"抹成"单峰模糊"。见 ALIAS_TOP_K 的注释。

    诚实边界：本函数只描述「匹配位移的分布形状」，不判断哪个模式是对的。
        它能说的最强结论是「存在两个互相一致、但彼此相差固定偏移的匹配簇」。
        真正判定"是否可判"要看 _model_inliers：**另一个位移能解释多少数据**。

    已知弱点（如实记录）：聚类是**贪心顺序**的 —— 依次把每对匹配并入第一个
        质心足够近的簇，因此一个簇的质心会随成员增加而漂移，理论上可能
        把两个相邻的峰并成一个。实测在 240 对、tolerance 4px 下未观察到
        这个问题（各峰间距远大于 4px），但换成周期更密的正交网格时需复核。
    """
    nfit = max(8, int(top_k))
    pairs: list[tuple[float, float]] = []
    for m in matches[:nfit]:
        x1, y1 = kps_prev[m.queryIdx].pt
        x2, y2 = kps_cur[m.trainIdx].pt
        pairs.append((float(x2 - x1), float(y2 - y1)))
    if not pairs:
        return [], []

    clusters: list[list[int]] = []
    cents: list[tuple[float, float]] = []
    for i, p in enumerate(pairs):
        for ci, cl in enumerate(clusters):
            cx, cy = cents[ci]
            if abs(p[0] - cx) <= tol_px and abs(p[1] - cy) <= tol_px:
                cl.append(i)
                k = len(cl)
                cents[ci] = (cx + (p[0] - cx) / k, cy + (p[1] - cy) / k)
                break
        else:
            clusters.append([i])
            cents.append(p)

    order = sorted(range(len(clusters)), key=lambda c: len(clusters[c]),
                   reverse=True)
    modes, members = [], []
    for ci in order:
        cl = clusters[ci]
        k = len(cl)
        modes.append({"n": k,
                      "dx": round(sum(pairs[j][0] for j in cl) / k, 2),
                      "dy": round(sum(pairs[j][1] for j in cl) / k, 2),
                      "share": round(k / len(pairs), 3)})
        members.append(cl)
    return modes, members


def _match_displacement_modes(kps_prev, kps_cur, matches,
                              top_k: int = 80, tol_px: float = 4.0) -> list[dict]:
    """_displacement_clusters 的只取统计量的薄封装（供外部诊断调用）。"""
    modes, _ = _displacement_clusters(kps_prev, kps_cur, matches,
                                      top_k=top_k, tol_px=tol_px)
    return modes


def _model_inliers(kps_prev, kps_cur, matches, seed_matches,
                   tol_px: float = 3.0) -> int | None:
    """
    拿 seed_matches 拟合一个单应，再数它在**全部** matches 上的内点数。

    这是回答"对齐是否唯一"的**唯一正确口径**：
        不能拿"另一个位移簇在 top-k 采样里占多少"当依据 —— 那既受采样深度影响，
        又和最强簇的绝对票数不可比。实测就栽在这里：某场景最强位移簇占 74%、
        次强只占 11%（差 6.8 倍），按"占比"判会被误拒；而把次强簇拿去和
        全量匹配比对，它的解释力只有最强解的 ~1/4，应当放行。
        反过来，严格周期墙上的伪解在全量口径下解释力与真解**几乎相等** ——
        那才是真正不可判的情形。

    返回 None 表示种子太少、无法给出判断（调用方应视为"不足以否定"）。
    """
    if seed_matches is None or len(seed_matches) < 8:
        return None
    src = np.float32([kps_prev[m.queryIdx].pt for m in seed_matches]).reshape(-1, 1, 2)
    dst = np.float32([kps_cur[m.trainIdx].pt for m in seed_matches]).reshape(-1, 1, 2)
    Hc, _ = cv2.findHomography(dst, src, cv2.RANSAC, tol_px,
                               maxIters=2000, confidence=0.995)
    if Hc is None:
        return None
    all_src = np.float32([kps_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    all_dst = np.float32([kps_cur[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(all_dst, Hc)
    d = np.linalg.norm(proj.reshape(-1, 2) - all_src.reshape(-1, 2), axis=1)
    return int((d <= tol_px).sum())


def _alias_gap(modes: list[dict],
               periods_px: list[float | None]) -> tuple[float | None, int | None, float | None]:
    """
    计算「最强模式」与「次强模式」之间的位移间距，并试着用砖距解释它。

    返回 (sep_px, k, period_px)：间距、间距≈k 个砖距、以及用的那个砖距。
    砖距量不出来时 k/period 为 None —— 此时只能说"位移有多峰"，不能说"是周期混叠"。
    """
    if len(modes) < 2:
        return None, None, None
    sep = float(np.hypot(modes[1]["dx"] - modes[0]["dx"],
                         modes[1]["dy"] - modes[0]["dy"]))
    for per in periods_px:
        if not per or per <= 4.0:
            continue
        r = sep / float(per)
        k = int(round(r))
        if 1 <= k <= 8 and abs(r - k) <= 0.18:
            return sep, k, float(per)
    return sep, None, None


def _feather_weight(shape: tuple[int, int]) -> np.ndarray:
    """
    生成羽化权重：用距离变换，越靠近画面中心权重越高。
    拼接时按权重加权平均，消除硬接缝。
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    # 到四条边的距离
    d = np.minimum.reduce([xx.astype(np.float32), yy.astype(np.float32),
                           (w - 1 - xx).astype(np.float32),
                           (h - 1 - yy).astype(np.float32)])
    d = np.maximum(d, 0.0)
    if d.max() > 0:
        d = d / d.max()
    # 抬底，避免边缘权重恰好为 0 导致接缝处变暗
    return (0.15 + 0.85 * d).astype(np.float32)


# --------------------------------------------------------------------------
# 无人机 POS 先验 → 单应初值（2026-09-25 新增）
# --------------------------------------------------------------------------
# 设计依据见 07_report/无人机POS先验接入设计论证_20260925.md。
#
# 核心思路：**先验给初值，特征做精修**。
#   POS（yaw/pitch/roll/d）→ 解析单应 H0（把墙面当已知平面）
#   → 用 H0 把相邻段放到大致正确的重叠位置 → 特征匹配只在正确区域内做精修。
#
# 为什么不是「直接用 POS」：民用 GNSS 米级、IMU 有累积漂移，
#   单独定标不够精；但它足够好到能**消除周期性砖墙的歧义**
#   （此时不再依赖「图案唯一性」消歧，而依赖「位姿把两段放到正确位置」）。
#
# ⚠️ 诚实边界：本组函数**未在真机航拍数据上验证过**（本项目无真机数据），
#   仅在 `_verify_rectify.py` 的合成场景 S7/S8 上验证（含注噪与注错两档）。
#   报告里**不得**宣称「已实现无人机测绘」。

POS_MIN_D_M = 0.5                 # 离墙距离下限（更近则透视过强、平面假设失效）
POS_MAX_DEVIATION_PX = 40.0       # 精修后四角平均位移 > 该值 ⇒ 判「POS 不可信」


def _rot_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """yaw-pitch-roll（度）→ 3x3 旋转矩阵 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    Rz = np.array([[np.cos(y), -np.sin(y), 0.0],
                   [np.sin(y), np.cos(y), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[np.cos(p), 0.0, np.sin(p)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(p), 0.0, np.cos(p)]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(r), -np.sin(r)],
                   [0.0, np.sin(r), np.cos(r)]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _homography_from_pose(pose_i: dict, pose_j: dict,
                          K: np.ndarray,
                          plane_normal: np.ndarray | None = None,
                          plane_d: float | None = None
                          ) -> np.ndarray | None:
    """
    由两段拍摄位姿解析地求「第 j 段 → 第 i 段」的单应初值 H0。

    姿态模型（简化但自洽）：
      · 相机固连在无人机上，光轴大致垂直指向墙面；
      · 位姿含 位置 (x,y) 平移（米）、离墙距离 d、以及姿态角 (yaw,pitch,roll)；
      · 墙面在世界系里是一个**已知平面**（取 Z=0 平面，法向 n=(0,0,1)、偏移 d）。

    单应推导（标准平面诱导单应）：
        H = K_j→i @ (R_ij + t_ij · nᵀ / d) @ K_j^{-1}
    其中 R_ij = R_iᵀ @ R_j、t_ij = R_iᵀ @ (t_j - t_i)（都换到相机 i 的坐标系）。

    ⚠️ 这是**初值**，不是终解。调用方必须用特征精修，并按
       `POS_MAX_DEVIATION_PX` 检查精修量与 H0 的偏离是否过大。

    参数
    ----
    pose_i, pose_j : dict
        各段位姿，至少含 `dx_m, dy_m, d_m, yaw, pitch, roll`（缺省按 0 处理）。
        这些量都是**世界系**（墙面系）下的值：dx/dy 为沿墙面的横向/纵向位移。
    K : np.ndarray
        3x3 相机内参。合成分段用同一个 K（同机同焦）。
    plane_normal, plane_d : 可选
        墙面法向（世界系）与平面偏移。默认 n=(0,0,1)、d 取两段 d_m 的均值。

    返回
    ----
    3x3 单应（已归一化到 H[2,2]=1），或 None（输入非法/退化）。
    """
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or abs(np.linalg.det(K)) < 1e-12:
        return None

    def _g(p, k, dv=0.0):
        try:
            v = float(p.get(k, dv))
            return v if np.isfinite(v) else dv
        except Exception:
            return dv

    di = _g(pose_i, "d_m")
    dj = _g(pose_j, "d_m")
    if min(di, dj) < POS_MIN_D_M:
        return None

    R_i = _rot_ypr(_g(pose_i, "yaw"), _g(pose_i, "pitch"), _g(pose_i, "roll"))
    R_j = _rot_ypr(_g(pose_j, "yaw"), _g(pose_j, "pitch"), _g(pose_j, "roll"))
    # 世界系位置：沿墙横/纵 (dx,dy)，离墙 (d)
    t_i = np.array([_g(pose_i, "dx_m"), _g(pose_i, "dy_m"), di], dtype=np.float64)
    t_j = np.array([_g(pose_j, "dx_m"), _g(pose_j, "dy_m"), dj], dtype=np.float64)

    R_ij = R_i.T @ R_j
    t_ij = R_i.T @ (t_j - t_i)

    if plane_normal is None:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = np.asarray(plane_normal, dtype=np.float64).reshape(3)
        n = n / max(np.linalg.norm(n), 1e-12)
    # 平面方程 nᵀX = plane_d；缺省时取两段距离均值（墙面在光轴前方 d 处）
    pd = (di + dj) * 0.5 if plane_d is None else float(plane_d)
    if abs(pd) < 1e-9:
        return None

    H = K @ (R_ij + np.outer(t_ij, n) / pd) @ np.linalg.inv(K)
    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def _corner_deviation_px(H_ref: np.ndarray, H_cmp: np.ndarray,
                         shape: tuple[int, int]) -> float:
    """两单应把同一幅图四角映到目标系后，四角平均位移（像素）。"""
    h, w = shape[:2]
    c = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64)

    def _proj(H):
        p = c @ H.T
        if np.any(np.abs(p[:, 2]) < 1e-9):
            return None
        return p[:, :2] / p[:, 2:3]

    a, b = _proj(H_ref), _proj(H_cmp)
    if a is None or b is None:
        return float("inf")
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def stitch_segments(images: list,
                    max_features: int = 4000,
                    min_inliers: int = 12,
                    min_inlier_ratio: float = 0.15,
                    poses: list | None = None,
                    K: np.ndarray | None = None) -> StitchResult:
    """
    把多段立面照片拼成一张整立面图。

    算法：相邻两两做 ORB 特征匹配 → RANSAC 求单应 → 沿链累积到首图坐标系
         → 全部 warp 到统一画布 → 按距离变换权重羽化融合。

    为什么两两串接而不是一次全匹配：
    立面分段拍摄通常是沿墙面依次平移，相邻段重叠最大、匹配最可靠；
    与首图直接匹配在跨度大时重叠不足，容易失配。串接还能逐段定位失败点。

    **任何一段匹配失败都会中止并明确报告是哪一段**，
    不静默跳过 —— 跳过后拼出来的图会有空洞，而用户不会知道哪里缺了。

    输入要求：
      - 各段之间有 **20%~50% 重叠**（重叠太少无法匹配，太多浪费分辨率）
      - 同一面墙、同一光照条件（跨时段拍摄会因白平衡差异导致匹配退化）

    ★ 新增可选参数（2026-09-25）：
      - `poses`：每段的位姿 `[{dx_m,dy_m,d_m,yaw,pitch,roll}, ...]`（无人机 POS）；
      - `K`    ：3x3 相机内参。
      两者**都为 None 时，本函数行为逐字节不变**（保持手机手持盲匹配路径）。
      给定时走「POS 先验 → 特征精修」路径：用解析单应 H0 限制匹配区域，
      并在精修量远离 H0 时**判 POS 不可信而拒答**（不照着错 POS 拼出错立面）。
    """
    n = len(images)
    if n == 0:
        return StitchResult(None, False, 0, 0, "none", "none",
                            "没有输入图像。", "请提供至少 2 张分段照片。")
    if n == 1:
        return StitchResult(images[0], True, 1, 1, "single", "medium",
                            "只有一张图，无需拼接，原图返回。",
                            "如需完整立面，请补充其它分段。")

    use_pose = poses is not None and K is not None and len(poses) == n
    if (poses is not None or K is not None) and not use_pose:
        # 有给但不完整：明确报错，**不静默退回盲匹配**（否则等于悄悄换了口径）
        return StitchResult(
            None, False, n, 0, "pose", "none",
            "给了 POS/内参但参数不完整（需 len(poses)==段数 且 K 为 3x3）。",
            "请同时提供每段位姿与相机内参；或两者都不给，退回手持盲匹配。",
            {"n_pose": 0 if poses is None else len(poses), "has_K": K is not None},
        )

    orb = cv2.ORB_create(nfeatures=max_features)
    grays = []
    for im in images:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
        grays.append(g)

    kps, dess = [], []
    for g in grays:
        k, d = orb.detectAndCompute(g, None)
        kps.append(k)
        dess.append(d)
    bad = [i for i, d in enumerate(dess) if d is None or len(d) < 12]
    if bad:
        return StitchResult(
            None, False, n, 0, "orb", "none",
            f"第 {[b + 1 for b in bad]} 张图特征点不足，无法参与拼接。",
            "这几张可能过曝、过暗或纹理缺失（大面积平整墙面）。请在画面里包含砖缝、"
            "窗框、管道等有结构的参照物。",
            {"n_keypoints": [0 if d is None else int(len(d)) for d in dess]},
        )

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    H_chain = [np.eye(3, dtype=np.float64)]
    pair_info = []
    n_pose_rejected = 0
    for i in range(1, n):
        prev, cur = i - 1, i
        matches = bf.match(dess[prev], dess[cur])
        matches = sorted(matches, key=lambda m: m.distance)
        info = {"pair": [prev + 1, cur + 1], "n_matches": int(len(matches))}

        # ---- 可选：无人机 POS 先验作为初值 ----
        H0 = None
        if use_pose:
            H0 = _homography_from_pose(poses[prev], poses[cur],
                                       np.asarray(K, dtype=np.float64))
            info["pose_prior"] = H0 is not None
            if H0 is None:
                info["pose_reason"] = "homography_from_pose_failed"

        if len(matches) < 8:
            pair_info.append({**info, "ok": False, "reason": "too_few_matches"})
            return StitchResult(
                None, False, n, 0, "orb+ransac", "none",
                f"第 {prev + 1} 段与第 {cur + 1} 段匹配失败：仅 {len(matches)} 对特征"
                "（至少需 8 对）。",
                "段间重叠可能不足。请保证相邻两张有 **20%~50% 画面重叠**，"
                "且拍摄时相机姿态变化不要过大（避免大幅旋转/变焦）。",
                {"pairs": pair_info},
            )
        src = np.float32([kps[prev][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kps[cur][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        Hc, mask = cv2.findHomography(dst, src, cv2.RANSAC, 3.0,
                                      maxIters=4000, confidence=0.995)
        if Hc is None or mask is None:
            pair_info.append({**info, "ok": False, "reason": "homography_none"})
            return StitchResult(
                None, False, n, 0, "orb+ransac", "none",
                f"第 {prev + 1} 段与第 {cur + 1} 段无法求出可靠单应。",
                "请补拍这两段之间的过渡照片（多拍一张重叠段最容易解决），"
                "或让各段的拍摄方向尽量一致。",
                {"pairs": pair_info},
            )

        # ---- POS 一致性守卫：精修量若远离 H0，说明 POS 不可信 ⇒ 拒答 ----
        if H0 is not None:
            dev = _corner_deviation_px(H0, Hc, grays[cur].shape)
            info["pose_dev_px"] = round(dev, 2) if np.isfinite(dev) else None
            # 只有在 H0 本应可用（未 None）且偏离超限时才拒
            if np.isfinite(dev) and dev > POS_MAX_DEVIATION_PX:
                n_pose_rejected += 1
                pair_info.append({**info, "ok": False,
                                  "reason": "pose_prior_inconsistent"})
                return StitchResult(
                    None, False, n, 0, "pose+orb+ransac", "low",
                    (f"第 {prev + 1} 段与第 {cur + 1} 段：POS 先验与图像证据**不一致**"
                     f"（四角平均偏离 {dev:.1f}px > 上限 {POS_MAX_DEVIATION_PX:.0f}px）。"),
                    ("这通常意味着 POS 本身不可信（GNSS 漂移过大、IMU 未收敛、"
                     "或离墙距离 d 给错），也可能是墙面强透视破坏了平面假设。"
                     "本系统选择**拒绝拼接**，而不是照着错 POS 拼出一个错误的立面后"
                     "再去做毫米级判读。请改用纯图像盲匹配（不给 poses），"
                     "或校准 POS / 提供更准的离墙距离。"),
                    {"pairs": pair_info, "pose_inconsistent": True,
                     "pose_dev_px": info["pose_dev_px"],
                     "pose_max_dev_px": POS_MAX_DEVIATION_PX},
                )

        inl = int(mask.sum())
        ratio = inl / float(len(matches))

        # 匹配位移的分布形状 —— 这是唯一能区分「不重叠」与「周期纹理歧义」的量。
        modes, members = _displacement_clusters(kps[prev], kps[cur], matches)
        n_cons = min(len(matches), ALIAS_TOP_K)
        top_ok = (len(modes) >= 1
                  and modes[0]["n"] >= max(ALIAS_MIN_TOP,
                                           ALIAS_MIN_TOP_SHARE * n_cons))
        alts: list[tuple[dict, list[int]]] = []
        if top_ok:
            dy0 = modes[0]["dy"]
            for mi in range(1, len(modes)):
                m = modes[mi]
                if abs(m["dy"] - dy0) > ALIAS_SAME_DIR_PX:
                    continue
                if abs(m["dx"] - modes[0]["dx"]) < ALIAS_MIN_SEP_PX:
                    continue
                if m["n"] >= max(3, int(ALIAS_MIN_ALT_SHARE * modes[0]["n"])):
                    alts.append((m, members[mi]))
        alias_suspected = bool(alts)

        # ★ 真判据：让"另一个位移"单独拟合一个模型，看它在全量匹配上解释多少。
        #   只有它能与已选中的解**分庭抗礼**，才说明对齐不可判。
        rival_inl, rival_ratio = None, None
        if alias_suspected and len(alts[0][1]) >= ALIAS_MIN_RIVAL_SEED:
            rival_inl = _model_inliers(kps[prev], kps[cur], matches,
                                       [matches[j] for j in alts[0][1]])
            if rival_inl is not None:
                rival_ratio = rival_inl / float(max(inl, 1))
        ambiguous = bool(rival_inl is not None
                         and rival_inl >= ALIAS_RIVAL_RATIO * max(inl, 1))

        sep, k_rep, per_used = None, None, None
        per_x = per_y = None
        if ambiguous:
            # 只有确认歧义时才去量砖距（自相关不便宜，不进主路径）。
            # 用本模块的 _modulus_px（逐行口径）而不是 gsd 的投影口径：
            # 后者在 running bond 上量到的是半模数，会把「相隔几个砖距」算成 2 倍。
            try:
                ph = _modulus_px(grays[prev], "x",
                                 max_period=max(120.0, 4.0 * BRICK_PITCH_MM))
                pv = _modulus_px(grays[prev], "y",
                                 max_period=max(120.0, 4.0 * BRICK_COURSE_MM))
                per_x, per_y = ph.get("period_px"), pv.get("period_px")
            except Exception:
                pass
            # 间距只在「最强峰 ↔ 最强的同向替代峰」之间量。
            # 旧写法直接用 modes[1]，那可能是**方向都不同**的另一个簇，
            # 拿它算出来的"整数个砖距"没有物理含义。
            sep, k_rep, per_used = _alias_gap([modes[0], alts[0][0]],
                                              [per_x, per_y])

        info.update({"inliers": inl, "inlier_ratio": round(ratio, 4),
                     "modes": modes[:3],
                     "alias_suspected": alias_suspected,
                     "rival_inliers": rival_inl,
                     "rival_ratio": (None if rival_ratio is None
                                     else round(rival_ratio, 3)),
                     "ambiguous": ambiguous,
                     "alt_modes": [a[0] for a in alts[:3]],
                     "ok": bool(inl >= min_inliers and ratio >= min_inlier_ratio)})
        pair_info.append(info)

        # ---- ★ POS 先验**仲裁**周期歧义（2026-09-25 新增）----
        #
        # 这是 POS 先验真正该起作用的地方。**不是**「用 H0 去筛匹配」——
        # 实测那样做很糟：ORB 匹配本身噪声大，用 H0 投影后按 3% 画面对角线筛，
        # 1700 对里只剩 1~4 对（几乎全是离群点），等于没用。
        # （反面教材见 logs/_verify_pose_prior.txt 与 S7 的 `先验筛后保留 1 对`。）
        #
        # 正确用法：**当盲匹配给出两个旗鼓相当的解时，让 POS 来投票**。
        # 砖墙的歧义是「相隔整数个砖距」——这两个解在图像证据上确实分不出高下，
        # 但它们在**世界几何**上差着一整个砖距的位移，POS 能分辨。
        #
        # 判据（三条都要满足才敢采信）：
        #   ① 某个候选解与 H0 的四角偏离 明显小于 另一个（差 ≥ POS_ARB_MIN_GAP_PX）；
        #   ② 被选中的那个偏离本身要小（≤ POS_MAX_DEVIATION_PX），否则说明 POS 本身离谱；
        #   ③ 竞争解的内点数不能压倒性更高（否则图像证据太强，不该让 POS 翻盘）。
        if ambiguous and H0 is not None:
            try:
                # 重建两个候选解各自的全量单应
                Hs = [Hc]                                     # 已选解
                if len(alts[0][1]) >= ALIAS_MIN_RIVAL_SEED:
                    Hr, mr = cv2.findHomography(
                        np.float32([kps[cur][m.trainIdx].pt
                                    for m in [matches[j] for j in alts[0][1]]]
                                   ).reshape(-1, 1, 2),
                        np.float32([kps[prev][m.queryIdx].pt
                                    for m in [matches[j] for j in alts[0][1]]]
                                   ).reshape(-1, 1, 2),
                        cv2.RANSAC, 3.0, maxIters=4000, confidence=0.995)
                    if Hr is not None:
                        Hs.append(Hr)
                devs = [_corner_deviation_px(H0, h, grays[cur].shape) for h in Hs]
                info["pose_candidates_dev_px"] = [None if not np.isfinite(d)
                                                  else round(d, 2) for d in devs]
                if len(Hs) == 2 and np.isfinite(devs[0]) and np.isfinite(devs[1]):
                    gap = abs(devs[0] - devs[1])
                    # ③ 竞争解内点不能压倒（图像证据太强则不许 POS 翻案）
                    rival_not_dominant = (rival_inl is None
                                          or rival_inl <= ALIAS_RIVAL_DOMINANT
                                          * max(inl, 1))
                    sel_dev, riv_dev = devs[0], devs[1]
                    if gap >= POS_ARB_MIN_GAP_PX and rival_not_dominant:
                        # 情形 A：**已选解贴合 H0、竞争解不贴合** ⇒ POS 确认了已选解，
                        #         歧义被打破，直接沿用 Hc（无需改选）。
                        if sel_dev <= POS_MAX_DEVIATION_PX and riv_dev > sel_dev:
                            ambiguous = False
                            alias_suspected = False
                            info["ambiguous"] = False
                            info["pose_arbitrated"] = "confirm_selected"
                            info["pose_dev_px"] = round(sel_dev, 2)
                            info["alias_arbitrated_dev_px"] = [
                                round(sel_dev, 2), round(riv_dev, 2)]
                            pair_info[-1].update(info)
                        # 情形 B：**竞争解贴合 H0、已选解不贴合** ⇒ POS 推翻盲匹配，
                        #         改选竞争解。
                        elif (riv_dev <= POS_MAX_DEVIATION_PX
                              and riv_dev < sel_dev):
                            ambiguous = False
                            alias_suspected = False
                            Hc = Hs[1]
                            info["ambiguous"] = False
                            info["pose_arbitrated"] = "switch_to_rival"
                            info["pose_dev_px"] = round(riv_dev, 2)
                            info["alias_arbitrated_dev_px"] = [
                                round(sel_dev, 2), round(riv_dev, 2)]
                            # 用仲裁后的 Hc 重算在"全量匹配"上的支持度，供下面门槛判据用
                            try:
                                proj = (np.hstack([src.reshape(-1, 2),
                                                   np.ones((len(matches), 1))])
                                        @ Hc.T)
                                if np.all(np.abs(proj[:, 2]) > 1e-9):
                                    pred = proj[:, :2] / proj[:, 2:3]
                                    dd = np.linalg.norm(
                                        pred - dst.reshape(-1, 2), axis=1)
                                    inl = int((dd <= 3.0).sum())
                                    ratio = inl / float(len(matches))
                            except Exception:
                                pass
                            info.update({"inliers": inl,
                                         "inlier_ratio": round(ratio, 4),
                                         "ok": bool(inl >= min_inliers
                                                    and ratio >= min_inlier_ratio)})
                            pair_info[-1].update(info)
            except Exception as _e:  # noqa: BLE001
                info["pose_arbitrate_error"] = repr(_e)

        # ---- 周期混叠：内点再高也不能用 ----
        # 砖墙没有特征能区分「下一皮」与「下两皮」。当"另一个位移"在全量匹配上
        # 能解释到与已选解同量级的内点数时，正确解**在原理上不可判**，
        # 此时返回任何单应都是在猜。
        # 这类失败最危险的地方恰恰是：内点占比可能**完全正常**（实测一对
        # 26% 的配对就是这么通过的），所以不能用 inlier_ratio 当门。
        if ambiguous:
            a0 = alts[0][0]
            over = ""
            if inl >= min_inliers and ratio >= min_inlier_ratio:
                over = (f"⚠️ 本对的内点占比 {ratio:.1%} **并不低**，单看它是会通过的 —— "
                        "这正是这类失败最危险的地方：拼出来的图看着毫无异常。")
            near = ""
            if k_rep is not None:
                near = (f"两峰相距 {sep:.1f}px ≈ {k_rep} 个砖距"
                        f"（砖距实测 {per_used:.1f}px）。")
            else:
                near = f"两峰相距 {sep:.1f}px（砖距未能量出，无法折算成砖距数）。"
            return StitchResult(
                None, False, n, 0, "orb+ransac", "low",
                (f"第 {prev + 1} 段与第 {cur + 1} 段无法确定唯一对齐：检测到**周期性混叠**。"
                 f"匹配位移呈 {len(modes)} 个峰，最强两个各有 {modes[0]['n']}、"
                 f"{a0['n']} 对支持（占所考察 {n_cons} 对匹配的 "
                 f"{modes[0]['share']:.0%}/{a0['share']:.0%}）；"
                 f"把第二个位移单独拟合后，它在**全量 {len(matches)} 对匹配**上"
                 f"解释出 {rival_inl} 个内点，而已选中的解是 {inl} 个"
                 f"（比值 {rival_ratio:.2f}）。" + near +
                 "换言之：把这一段的砖缝对齐到另一个位置，证据强度与正确对齐"
                 "基本相当，而砖墙本身没有任何特征能区分它们。" + over),
                ("这不是拍摄质量问题，而是**砖墙的周期性**造成的固有歧义 —— "
                 "提高重叠率无法解决。请改用以下任一办法："
                 "① 让相邻两段都包含同一个**非周期参照物**（窗框、落水管、空调支架、"
                 "层间腰线、墙上的门牌），用它做唯一锚点；"
                 "② 在段间贴一张 **A4 大小、画上棋盘格/二维码等高对比图案**的纸片，"
                 "并确保**它同时出现在相邻两段里**（实测：A4 棋盘格纸片能把配对内点"
                 "从 17% 提到 42~46%，余量从踩线变成 +27 个百分点以上）；"
                 "⚠️ 但**每个接缝的纸片必须能彼此区分**（编号或不同图案）——"
                 "各接缝用同一张脸，纸片自己就变成了一个新周期：实测两解在全量匹配上"
                 "解释出 626 : 625 个内点（比值 1.00，完全平票），照样只能拒答；"
                 "③ 保留一段完整的过渡照，改用它把两段桥接。"),
                {"pairs": pair_info, "modes": modes[:3], "alias": True,
                 "alias_sep_px": (None if sep is None else round(sep, 2)),
                 "alias_k_periods": k_rep,
                 "alias_period_px": (None if per_used is None
                                     else round(per_used, 2)),
                 "n_considered": n_cons,
                 "alias_suspected": True, "ambiguous": True,
                 "rival_inliers": rival_inl, "rival_ratio": rival_ratio},
            )

        if inl < min_inliers or ratio < min_inlier_ratio:
            if alias_suspected:
                # 位移多峰、且都与最强峰同向 —— 明显是平移性周期纹理的指纹，
                # 只是"另一个解"解释力不足（或种子太少）而没能确认歧义。
                # 措辞上必须区分「疑似周期歧义」与「不重叠/拍糊」，
                # 否则用户会去重拍（无效），而不是去找一个非周期参照物（有效）。
                g = float(np.hypot(alts[0][0]["dx"] - modes[0]["dx"],
                                   alts[0][0]["dy"] - modes[0]["dy"]))
                why = (f"内点 {inl}/{len(matches)}（占比 {ratio:.1%}）低于门槛"
                       f"（内点>={min_inliers} 且占比>={min_inlier_ratio:.0%}），"
                       f"且匹配位移不唯一 —— 除最强位移（{modes[0]['n']} 对支持）外，"
                       f"还有 {len(alts)} 个**同向但平移量不同**的位移簇，"
                       f"最近的一个有 {alts[0][0]['n']} 对支持、与最强位移相差 {g:.1f}px。"
                       "各簇方向一致而平移量不等，是**平移性周期纹理**的指纹，"
                       "而不是「拍糊了」或「不重叠」。")
                act = ("砖缝在画面里周而复始，RANSAC 挑出的平移量只在一个砖距内可信。"
                       "重拍同一面墙帮助有限；请让相邻两段都拍到同一个"
                       "**非周期参照物**（窗框、落水管、空调支架、腰线、门牌）作为唯一锚点，"
                       "或在段间贴一张 **A4 大小、带棋盘格/二维码图案**的纸片"
                       "（务必让它同时出现在相邻两段里，且各接缝的图案互不相同），"
                       "并补拍一段过渡照。")
            else:
                why = (f"内点 {inl}/{len(matches)}（占比 {ratio:.1%}），"
                       f"低于门槛（内点>={min_inliers} 且占比>={min_inlier_ratio:.0%}）。")
                act = ("内点比例低通常意味着两段其实不重叠、或拍摄时发生了旋转/缩放。"
                       "请重拍并使用相近的拍摄距离和角度。")
            return StitchResult(
                None, False, n, 0, "orb+ransac", "low",
                f"第 {prev + 1} 段与第 {cur + 1} 段匹配不可靠：{why}",
                act, {"pairs": pair_info, "modes": modes[:3],
                      "alias_suspected": alias_suspected, "ambiguous": False,
                      "rival_inliers": rival_inl, "rival_ratio": rival_ratio,
                      "alt_modes": [a[0] for a in alts[:3]]},
            )
        H_chain.append(H_chain[-1] @ np.linalg.inv(Hc))

    # 求统一画布
    all_corners = []
    for i, im in enumerate(images):
        h, w = im.shape[:2]
        c = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64)
        p = c @ H_chain[i].T
        if np.any(np.abs(p[:, 2]) < 1e-9):
            return StitchResult(None, False, n, 0, "orb+ransac", "none",
                                "拼接几何退化，画布无法确定。",
                                "请重拍。", {"pairs": pair_info})
        all_corners.append(p[:, :2] / p[:, 2:3])
    all_corners = np.vstack(all_corners)
    x0, y0 = all_corners[:, 0].min(), all_corners[:, 1].min()
    x1, y1 = all_corners[:, 0].max(), all_corners[:, 1].max()
    out_w, out_h = int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0))
    if out_w <= 1 or out_h <= 1:
        return StitchResult(None, False, n, 0, "orb+ransac", "none",
                            "拼接后画布尺寸退化。", "请重拍。", {"pairs": pair_info})

    clamped = False
    if max(out_w, out_h) > MAX_CANVAS_SIDE:
        return StitchResult(
            None, False, n, 0, "orb+ransac", "low",
            (f"拼接结果画布达 {out_w}x{out_h}，超出安全上限 {MAX_CANVAS_SIDE}。"
             "这通常说明段数过多或单应出现异常放大。"),
            "请减少分段数量，或分两次拼接后人工对照。"
            "（这里选择拒绝而不是降采样，因为降采样会破坏毫米级判读的像素预算。）",
            {"pairs": pair_info, "canvas": [out_w, out_h]},
        )

    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    wsum = np.zeros((out_h, out_w, 1), dtype=np.float32)
    T = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
    used = 0
    for i, im in enumerate(images):
        h, w = im.shape[:2]
        wgt = _feather_weight((h, w))[:, :, None]
        warped = cv2.warpPerspective(im.astype(np.float32), T @ H_chain[i],
                                     (out_w, out_h), flags=cv2.INTER_LINEAR)
        ww = cv2.warpPerspective(wgt, T @ H_chain[i], (out_w, out_h),
                                 flags=cv2.INTER_LINEAR)
        # ⚠️ OpenCV 5 会丢掉长度为 1 的通道维：(H,W,1) -> (H,W)。
        #    不补回来，下面的 canvas*ww 与 wsum+=ww 会因广播失败而崩。
        if ww.ndim == 2:
            ww = ww[:, :, None]
        canvas += warped * ww
        wsum += ww
        used += 1

    m = wsum[:, :, 0] > 1e-6
    out = np.zeros_like(canvas)
    # ⚠️ 必须用 wsum[m]（形状 (N,1)），不能用 wsum[m][:, None]（形状 (N,1,1)）：
    #    后者与 canvas[m]（形状 (N,3)）广播会得到 (N,N,3)，
    #    实测 N≈81 万时直接申请 7.15 TiB 内存并崩溃
    #    （numpy._core._exceptions._ArrayMemoryError, shape (809610, 809610, 3)）。
    if np.any(m):
        out[m] = canvas[m] / wsum[m]
    out = np.clip(out, 0, 255).astype(np.uint8)

    cover = float(m.mean())
    n_pair_ok = sum(1 for p in pair_info if p.get("ok"))
    conf = "high" if (n_pair_ok == n - 1 and cover > 0.35) else "medium"
    method = ("pose+orb+ransac+feather" if use_pose else "orb+ransac+feather")
    pose_note = ""
    if use_pose:
        n_prior = sum(1 for p in pair_info if p.get("pose_prior"))
        n_filt = sum(1 for p in pair_info if p.get("pose_filter"))
        pose_note = (f"（POS 先验生效：{n_prior}/{n - 1} 对给出解析初值，"
                     f"{n_filt} 对用先验筛过匹配）")
    return StitchResult(
        out, True, n, used, method, conf,
        (f"拼接成功：{used} 段 → {out_w}x{out_h}，"
         f"有效覆盖 {cover:.1%}，逐段内点 "
         + "、".join(f"{p.get('inliers', '?')}" for p in pair_info) + "。"
         + pose_note),
        ("拼接图为统一坐标系下的整体立面，可直接接正射校正与量化；"
         "但请留意覆盖度 —— 低于 100% 说明有些区域只有单张图贡献，"
         "该处分辨率不叠加。"),
        {"pairs": pair_info, "canvas": [out_w, out_h], "coverage": round(cover, 4),
         "clamped": clamped, "use_pose": bool(use_pose),
         "n_pose_rejected": int(n_pose_rejected)},
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _main() -> None:
    import sys
    from pathlib import Path

    from common import argv_flag, list_images

    single = argv_flag("image")
    folder = argv_flag("dir")
    out = argv_flag("out", "")
    do_stitch = argv_flag("stitch") is not None
    force = argv_flag("force") is not None

    if do_stitch:
        if not folder:
            log("拼接模式需 --dir=<目录>（目录内按文件名排序即为拍摄顺序）")
            return
        paths = list_images(folder)
        if len(paths) < 2:
            log(f"目录内只有 {len(paths)} 张图，无法拼接")
            return
        log(f"待拼接 {len(paths)} 段：{[p.name for p in paths]}")
        imgs = [imread_u(p) for p in paths]
        ok = [im for im in imgs if im is not None]
        if len(ok) != len(imgs):
            log(f"!! 有 {len(imgs) - len(ok)} 张读取失败，已中止")
            return
        res = stitch_segments(ok)
        log(res.message)
        if not res.ok:
            log(f"建议：{res.advice}")
            return
        dst = Path(out) if out else (Path(folder) / "_stitched.jpg")
        imwrite_u(dst, res.image)
        log(f"拼接结果已写入 {dst}")
        log(f"置信度={res.confidence}  建议：{res.advice}")
        return

    if not single:
        log("请指定 --image=<文件> 或 --dir=<目录> --stitch")
        return
    img = imread_u(single)
    if img is None:
        log(f"读取失败: {single}")
        return
    res = rectify(img, force=force)
    log(f"applied={res.applied}  method={res.method}  conf={res.confidence}")
    log(res.message)
    log(f"建议：{res.advice}")
    if res.scale_mm_per_px is not None:
        log(f"全画面统一尺度 = {res.scale_mm_per_px:.4f} mm/px")
    if out:
        imwrite_u(out, res.image)
        log(f"校正结果已写入 {out}")
    _ = sys


if __name__ == "__main__":
    _main()
