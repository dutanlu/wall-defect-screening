# -*- coding: utf-8 -*-
"""
measure.py —— 缺陷几何量化（YOLO 之后的那一半：OpenCV 负责「有多大」）

架构分工（务必记住）：
  YOLO    负责「在哪里、是什么」—— 出 bbox + 类别
  OpenCV  负责「有多大」          —— 像素级宽度/长度/面积 + 尺度标定
  规则代码 负责国标分级

为什么不能只用 YOLO：bbox 无法回答「这条裂缝是 0.35mm 还是 1.2mm」；
裂缝细长，框内宽度信息在 bbox 表征里是缺失的。必须回到像素级操作。

本模块提供：
  1. ROI 裁剪 + 前景分割（自适应阈值，适配裂缝/剥落/锈迹不同形态）
  2. 裂缝类：骨架化求长度、距离变换求平均/最大宽度（对细长目标正确）
  3. 块状类（剥落/空鼓）：轮廓面积、等效直径、面积占比
  4. 全部结果同时给「像素」和「毫米/平方米」两套单位
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from gsd import Calibration, MIN_RELIABLE_PX


@dataclass
class Measurement:
    """一个缺陷实例的量化结果。像素与物理单位并存，便于追溯。"""
    cls_name: str
    bbox_xyxy: tuple[int, int, int, int]
    # 几何（像素）
    length_px: float = 0.0
    width_mean_px: float = 0.0
    width_max_px: float = 0.0
    area_px: float = 0.0
    equiv_diameter_px: float = 0.0
    # 几何（物理）
    length_mm: float = 0.0
    width_mean_mm: float = 0.0
    width_max_mm: float = 0.0
    area_mm2: float = 0.0
    area_m2: float = 0.0
    # 形态学
    elongation: float = 0.0        # 细长比 = 长度/平均宽度，>8 认为是线状缺陷
    area_ratio: float = 0.0        # 缺陷面积占 ROI 面积比例
    method: str = ""               # 使用的分割方法，便于复核
    # ---- 测量窗口可用性（2026-09-23 新增，来自 risk–coverage 实验的发现）----
    # ★ 为什么必须记：几何判据只保证了**下界**（`px >= MIN_RELIABLE_PX`），
    #   但分割用的形态学核 `ks` 同时决定了**上界 `ks−1`** —— 像素**太多**时
    #   宽目标会静默失明（§9.1 记录的盲区；ks 依赖 ROI 尺寸，框太紧就会踩到）。
    #   实测（logs/_RISK_COVERAGE.md）：单侧判据下 risk 达 **60.7%**，
    #   补上上界后降到 **29.2%**。故把可用性判定**逐实例**记下来、交给上层拒答。
    ks: int = 0                    # 本实例实际使用的形态学核尺寸
    window_ok: bool = True         # 尺寸是否落在可靠区间 [MIN_RELIABLE_PX, ks−1]
    window_note: str = ""          # 越界原因（供界面/JSON 直接展示）

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox_xyxy"] = list(self.bbox_xyxy)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


# --------------------------------------------------------------------------
# 前景分割
# --------------------------------------------------------------------------
def segment_defect(roi: np.ndarray, cls_name: str) -> tuple[np.ndarray, str]:
    """
    在 ROI 内分割出缺陷前景。返回 (二值掩膜 uint8 0/255, 方法名)。

    设计原则：**宁缺毋滥**。分割过多会把背景当缺陷，导致长度/面积爆炸
    （实测教训：自适应阈值的 C 参数取错，300x300 的 ROI 里 14305 个前景像素，
     而真实裂缝只占约 1200 个 —— 误差 10 倍以上）。
    因此这里改用「Otsu + 显著性约束」：
      1. 用黑帽/顶帽提取局部异常响应，得到「疑似缺陷」的响应图
      2. 对响应图做 Otsu 全局阈值 —— Otsu 自动找双峰分割点，无需手工调 C
      3. 再用连通域面积过滤掉碎斑
      4. 最后做一次开运算是为了去掉孤立的细噪点

    不同缺陷形态差异大，分两套策略：
      - 线状类（crack / exposed_rebar / rust）：细长、局部偏暗
        -> CLAHE 增强 + 黑帽（提取比周围暗的细结构）
      - 块状类（spalling / efflorescence / delamination / moss）：区域色调/亮度异常
        -> CLAHE 增强 + 绝对值拉普拉斯/局部对比度

    ⚠️ 新增类别必须归入上述两类之一。默认落在 else（块状）分支，
       所以把「线状类」写成白名单而非黑名单是刻意的：
       漏登记的线状缺陷会被当块状处理，但对块状误判为线状后果更严重
       （黑帽会试图在苔藓/泛碱里找细线，量出无意义的长度）。
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    # CLAHE 提升局部对比度，抑制整体光照不均
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)

    line_like = cls_name in ("crack", "exposed_rebar", "rust")
    if line_like:
        # 黑帽：提取比周围暗的细结构（裂缝主体）。
        # 核大小取 ROI 短边的 ~1/12，至少 7。核太大反而把宽裂缝也抹平。
        ks = max(7, (min(roi.shape[:2]) // 12) | 1)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
        resp = cv2.morphologyEx(enh, cv2.MORPH_BLACKHAT, k)
        method = f"blackhat{ks}+otsu"
    else:
        # 块状：用「与局部中值之差的绝对值」作响应，
        # 中值滤波核取短边 1/8，能突出成片异常区域
        ks = max(9, (min(roi.shape[:2]) // 8) | 1)
        med = cv2.medianBlur(enh, min(ks, 31) if ks <= 31 else 31)
        resp = cv2.absdiff(enh, med)
        method = f"meddev{ks}+otsu"

    # Otsu 全局阈值：自动定位响应分布的分离点
    resp_n = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(resp_n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 去掉孤立噪点：开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    # 面积过滤：丢掉过小的连通域
    min_area = max(12, int(0.0008 * roi.shape[0] * roi.shape[1]))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)

    # ★ 2026-09-26 改为**查表**（单次 O(N) 向量化）。
    #   原写法是逐连通域做一次**全图** `labels == i` 比较 ⇒ O(连通域数 × ROI 像素数)，
    #   ROI 大且连通域多时会明显变慢（实测该函数在真实 ROI 上约 16.8 ms）。
    #   等价性要点：原循环是 `range(1, n)`，**标签 0（背景）从不参与**
    #   ⇒ 查表时也必须排除 0，`lut[0]` 保持 0。两者产出 mask 字节完全相同，
    #   由 `logs/_verify_lut_filter.py` 逐像素验证。
    area = stats[:, cv2.CC_STAT_AREA]
    keep_ids = np.nonzero(area >= min_area)[0]
    keep_ids = keep_ids[keep_ids != 0]          # ★ 显式排除背景标签 0
    lut = np.zeros(n, dtype=np.uint8)
    lut[keep_ids] = 255
    mask = lut[labels]
    return mask, method


# --------------------------------------------------------------------------
# 线状缺陷测量：骨架化求长度 + 距离变换求宽度
# --------------------------------------------------------------------------
def _thin_skeleton_zs_full(img: np.ndarray) -> np.ndarray:
    """原版：全图向量化 Zhang-Suen 细化。

    ⚠️ **本函数是回归对照的基准，不要删、不要改**。
    2026-09-26 优化前的实现即此；`_thin_skeleton_active` 必须与它**逐像素等价**，
    由 `logs/_verify_thin_active.py` 在 **28 个用例**上同时导入两者做对照。

    ★★★ **本函数会原地修改传入的数组**（内部有 `img[delete] = 0`）。
      上游 `_thin_skeleton` 因为先 `(mask > 0).astype(np.uint8)` 造了新数组，
      所以正式路径不受影响；但**直接拿它做基准测试会踩坑**：
      在同一份数组上连续调用，第 1 次就把输入"细化"了，之后都是在
      **已收敛的骨架**上跑 ⇒ 耗时被严重低估、等价性比对退化成平凡通过。
      （实测证据：同一份数组 `img.sum()` 报 327,526 前景，
        紧接着的 `np.nonzero` 只剩 30,059 点。）
      ⇒ **凡对它计时或比对，必须每轮传 `.copy()`。**
    """
    def neighbors(im):
        """返回 8 邻域矩阵 (P2..P9)，顺序按 Zhang-Suen 定义（顺时针）。"""
        p2 = np.roll(im, -1, 0)          # 上
        p3 = np.roll(np.roll(im, -1, 0), 1, 1)   # 右上
        p4 = np.roll(im, 1, 1)           # 右
        p5 = np.roll(np.roll(im, 1, 0), 1, 1)    # 右下
        p6 = np.roll(im, 1, 0)           # 下
        p7 = np.roll(np.roll(im, 1, 0), -1, 1)   # 左下
        p8 = np.roll(im, -1, 1)          # 左
        p9 = np.roll(np.roll(im, -1, 0), -1, 1)  # 左上
        return p2, p3, p4, p5, p6, p7, p8, p9

    changed = True
    guard = 0
    while changed and guard < 200:
        changed = False
        guard += 1
        for step in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = neighbors(img)
            # B(P1) = 前景邻居数
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            # A(P1) = 0->1 转换次数
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            A = np.zeros_like(B, dtype=np.int16)
            for i in range(8):
                A += ((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16)

            if step == 0:
                cond = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)

            delete = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if delete.any():
                img[delete] = 0
                changed = True
    return (img * 255).astype(np.uint8)


# Zhang-Suen 的 8 邻域偏移，**顺序必须与 np.roll 版逐个对应**：
#   p2=(+1,0) p3=(+1,-1) p4=(0,-1) p5=(-1,-1) p6=(-1,0) p7=(-1,+1) p8=(0,+1) p9=(+1,+1)
# 注意 p2 名义上是「上」，但 np.roll(im,-1,axis=0)[y,x] 取的是 im[y+1,x]。
# 这个顺序构成一个完整的环，A（0->1 跳变数）才成立 —— **不要"顺手修正"它**，
# 一改就会得到与原版不同的骨架。
_ZS_OFFSETS = ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))

# 掩膜像素数低于此值时走全图版：小掩膜上活跃集的固定开销（坐标拼装、unique）
# 反而比全图布尔运算更贵（实测 190x243 仅 1.58x，更小则可能 <1x）。
_ZS_ACTIVE_MIN_PX = 4096

# ★ 关于「宽正面兜底」——**实测证明它弊大于利，故不采用**（留档以免后人重做）
#   曾实现过一个兜底：检测首轮删除面宽度 ratio=(|D0|+|P1|)/|前景| ≥ 0.35 时，
#   放弃活跃集、交回全图版。动机是修「密集细条纹」形态（合成实测 0.37~0.51x）。
#   但真实负载实测结果**变差**：
#     · 无兜底：变慢 0/80，中位 4.17x，整体 4.08x
#     · 有兜底：变慢 7/80，中位 3.62x，整体 3.27x
#   根因：兜底为了做判断必须先跑一次 `_zs_eval_full`（约 10~25 ms），
#   而退回全图版会让这部分**白白浪费** ⇒ 「退回」并不是零成本，它自己就会造成回归。
#   ⇒ 教训：**保守回退只有当回退路径不比原路径更贵时才真的"保守"。**
#   证据：logs/_thin_workload_dist.txt、logs/_thin_d0_ratio.txt。
#   已知局限（如实记录）：对**合成**的密集细条纹掩膜（2px 条纹 0.51x、5px 条纹 0.37x）
#   活跃集版更慢。该类形态无法由 `segment_defect` 产出（其形态学开运算会削掉细结构），
#   真实负载 80 例无一变慢。


def _zs_eval_full(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对**整幅图**做一次向量化 Zhang-Suen 求值，返回 (d0, d1) 两个布尔掩膜。

    完全是 `_thin_skeleton_zs_full` 单步逻辑的镜像，只是**同时**算出两套 cond 的删除判据
    （两者共用 B、A，只多两次布尔与）。供活跃集版处理**首轮**用。

    ★ 为什么需要它：活跃集版若首轮就对全部前景坐标做 8 次 gather，
      而前景又占满全图（如整幅实心），那一步会比全图版的 400 次 roll **更贵**
      —— 实测 `实心_2000x2900` 活跃集反而慢 3.4 倍（0.29x）。
      首轮改用本函数（连续内存的 roll）后该问题消失。
    """
    im = img
    p2 = np.roll(im, -1, 0)
    p3 = np.roll(np.roll(im, -1, 0), 1, 1)
    p4 = np.roll(im, 1, 1)
    p5 = np.roll(np.roll(im, 1, 0), 1, 1)
    p6 = np.roll(im, 1, 0)
    p7 = np.roll(np.roll(im, 1, 0), -1, 1)
    p8 = np.roll(im, -1, 1)
    p9 = np.roll(np.roll(im, -1, 0), -1, 1)
    seq = (p2, p3, p4, p5, p6, p7, p8, p9)
    B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
    A = np.zeros_like(B, dtype=np.int16)
    for i in range(8):
        A += ((seq[i] == 0) & (seq[(i + 1) % 8] == 1)).astype(np.int16)
    base = (im == 1) & (B >= 2) & (B <= 6) & (A == 1)
    d0 = base & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
    d1 = base & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
    return d0, d1


def _zs_ring_alive(alive: np.ndarray, ys: np.ndarray, xs: np.ndarray,
                   H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """返回被删像素的 1 邻域中**仍存活**者的坐标（模运算复刻 np.roll 的环绕）。"""
    ay: list[np.ndarray] = []
    ax: list[np.ndarray] = []
    for dy, dx in _ZS_OFFSETS:
        ny = (ys + dy) % H
        nx = (xs + dx) % W
        m = alive[ny, nx]
        ay.append(ny[m])
        ax.append(nx[m])
    if not ay:
        e = np.empty(0, dtype=np.int64)
        return e, e
    return np.concatenate(ay), np.concatenate(ax)


def _thin_skeleton_active(img: np.ndarray) -> np.ndarray:
    """活跃集（脏区）增量版 Zhang-Suen —— 与全图版**逐像素等价**。

    动机：全图版每轮对整幅图做 8 次 roll + 一轮布尔运算，实测占 measure 耗时的 97.6%。
    但一次删除只影响其 1 邻域 ⇒ 每轮真正需要重新求值的，只是「上一轮被删像素的 1 邻域」。

    ★★ 与历史失败路线的本质区别（务必看懂，否则会重踩）：
      历史第 1/5 条路线是「**crop 子图 + 边界补 0**」，破坏了 np.roll 的**环绕语义**
      （`logs/_thin_diag.txt` 已实证：第 0 行的"上邻居"取到末行）。
      本实现**完全不裁剪**：整幅图始终存在，只是在**坐标数组**上求值，
      邻域用模运算 `(y+dy) % H`、`(x+dx) % W` **复刻 np.roll 的环绕**。

    ★★ step0/step1 双判据陷阱（本节第一版就栽在这里）：
      Zhang-Suen 两个 step 用**不同的 cond**。若每个 step 只重算"当前判据"，
      则某像素在 step0 判为 False、且不在任何删除点的邻域内时，step1 会被跳过 ——
      但它在 cond1 下本应被删。
      ⇒ 正解：**一次求值同时算 d0 与 d1**（两者共用 B、A，只多两次布尔运算），
        并把 `d_other & ~d_this` 的坐标存进 P[other] 交给另一步。

    ★★ 首轮走全图向量化（第二个实测出来的坑）：
      首轮候选集 = 全部前景。若对 `|前景|` 个坐标做 8 次 gather，而前景又占满全图，
      那一步比全图版的 400 次 roll **更贵** —— 实测 `实心_2000x2900` 活跃集慢 3.4 倍。
      ⇒ 首轮 step 0 改用 `_zs_eval_full`（连续内存 roll），之后才切到活跃集路径。
      实测修掉后 `实心_2000x2900` 由 0.29x 转为显著加速。

    正确性不变量：
      若像素 p 不在任何删除点的 1 邻域内，则 p 的 0/1 邻域与上一 step 完全一致，
      此时 d0、d1 都未变。`U ∪ P[s]` 恰为「需重新求值 或 已知可删」的集合，
      故每个 step 删除的集合与原版逐位相同。

    ⚠️ 本函数只接受 0/1 的 uint8 图（调用方已做 `(mask > 0).astype(np.uint8)`）。
    """
    H, W = img.shape
    alive = img.astype(bool)
    empty = np.empty(0, dtype=np.int64)

    # ---------------- 第 1 轮 step 0：全图向量化（便宜） ----------------
    d0m, d1m = _zs_eval_full(img)

    Dy, Dx = np.nonzero(d0m)
    if Dy.size:
        alive[Dy, Dx] = False
    py, px = np.nonzero(d1m & ~d0m)
    # P[s] = 「上次求值时刻满足 step s 的 cond、但当时未删除」的坐标
    P = {0: (empty, empty), 1: (py.astype(np.int64), px.astype(np.int64))}
    U = _zs_ring_alive(alive, Dy, Dx, H, W) if Dy.size else (empty, empty)
    changed = bool(Dy.size)          # 第 1 轮 step 0 是否删过
    guard = 1

    def active_step(step: int) -> bool:
        """跑一个 step 的活跃集路径；返回本步是否删过。就地更新 P / U。"""
        nonlocal U
        uys, uxs = U
        pys, pxs = P[step]
        if uys.size and pys.size:
            cy = np.concatenate((uys, pys))
            cx = np.concatenate((uxs, pxs))
            key = np.unique(cy * W + cx)
            cy = key // W
            cx = key % W
        elif pys.size:
            cy, cx = pys, pxs
        elif uys.size:
            cy, cx = uys, uxs
        else:
            P[1 - step] = (empty, empty)
            U = (empty, empty)
            return False

        # 只保留仍存活的候选（等价于全图版的 (img == 1) 条件）
        keep = alive[cy, cx]
        cy = cy[keep]
        cx = cx[keep]
        if cy.size == 0:
            P[1 - step] = (empty, empty)
            U = (empty, empty)
            return False

        # 8 邻域：模运算复刻 np.roll 的环绕语义（**绝不裁剪**）
        nb = [alive[(cy + dy) % H, (cx + dx) % W] for dy, dx in _ZS_OFFSETS]
        B = np.zeros(cy.shape, dtype=np.int16)
        for a in nb:
            B += a
        A = np.zeros(cy.shape, dtype=np.int16)
        for i in range(8):
            A += (~nb[i] & nb[(i + 1) % 8])

        p2, p4, p6, p8 = nb[0], nb[2], nb[4], nb[6]
        common = (B >= 2) & (B <= 6) & (A == 1)
        d0 = common & (~(p2 & p4 & p6)) & (~(p4 & p6 & p8))
        d1 = common & (~(p2 & p4 & p8)) & (~(p2 & p6 & p8))

        ds = d0 if step == 0 else d1
        do = d1 if step == 0 else d0

        dyy, dxx = cy[ds], cx[ds]
        if dyy.size:
            alive[dyy, dxx] = False

        oy, ox = cy[do & ~ds], cx[do & ~ds]
        P[1 - step] = (oy, ox)
        U = _zs_ring_alive(alive, dyy, dxx, H, W) if dyy.size else (empty, empty)
        return bool(dyy.size)

    # ---------------- 第 1 轮 step 1（活跃集路径） ----------------
    changed = active_step(1) or changed

    # ---------------- 后续轮 ----------------
    while changed and guard < 200:
        changed = False
        guard += 1
        for step in (0, 1):
            if active_step(step):
                changed = True

    out = np.zeros((H, W), dtype=np.uint8)
    out[alive] = 255
    return out


def _thin_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    骨架化（分发器）。

    优先 cv2.ximgproc.thinning（contrib 模块）。
    本机实测 cv2 5.0.0 **不含 ximgproc**，因此必须自带实现。

    自带实现是标准 Zhang-Suen 细化，有两个版本（两者**逐像素等价**）：
      · `_thin_skeleton_active`   —— 活跃集增量版，用于大掩膜（瓶颈优化，见其 docstring）
      · `_thin_skeleton_zs_full`  —— 全图向量化版，用于小掩膜（活跃集固定开销不划算）
    分发阈值见 `_ZS_ACTIVE_MIN_PX`。

    踩坑记录：先前用手写的「形态学开运算相减」近似，会把结果叠加成
    2D 宽带而非 1px 线 —— 300px 的裂缝量出 64672px 长度（差 200 倍）。
    Zhang-Suen 是经过验证的经典算法，按它逐步删除边界点即可。
    """
    if mask is None or cv2.countNonZero(mask) == 0:
        return np.zeros_like(mask)

    # 方案 1：contrib 的 thinning（最可靠，本机不可用）
    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            return cv2.ximgproc.thinning(mask)
    except Exception:
        pass

    # 方案 2：标准 Zhang-Suen 细化（按掩膜大小选择实现）
    img = (mask > 0).astype(np.uint8)
    if img.size < _ZS_ACTIVE_MIN_PX:
        return _thin_skeleton_zs_full(img)
    return _thin_skeleton_active(img)


def _skeleton_length_px_pyslow(skel: np.ndarray) -> float:
    """**原实现（纯 Python set 循环）** —— 保留作回归对照，勿删。

    它逐点做 4 次哈希查找；大掩膜上实测 17~20 ms（占 measure 6~14%）。
    向量化版必须与它**位级等价**，由 `logs/_verify_skeleton_length.py` 对照验证。
    """
    ys, xs = np.nonzero(skel > 0)
    if ys.size == 0:
        return 0.0
    pts = set(zip(ys.tolist(), xs.tolist()))
    n_diag = 0
    n_orth = 0
    for (y, x) in pts:
        # 只看向「右/下/右下/左下」四个方向，避免重复计数
        for (dy, dx), w in (((0, 1), "o"), ((1, 0), "o"),
                            ((1, 1), "d"), ((1, -1), "d")):
            if (y + dy, x + dx) in pts:
                if w == "d":
                    n_diag += 1
                else:
                    n_orth += 1
    return n_orth * 1.0 + n_diag * float(np.sqrt(2.0))


def _shift_no_wrap(b: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """返回 `res[y, x] = b[y+dy, x+dx]`，**越界一律取 False（不环绕）**。

    ★ 与原实现语义严格一致：原实现用 `(y+dy, x+dx) in pts` 做集合查找，
      越界坐标根本不在集合里 ⇒ 等价于「越界为 False」。
      （**注意与 `_thin_skeleton` 的区别**：那里用 `np.roll`，是**会环绕**的。）
    """
    res = np.zeros_like(b)
    h, w = b.shape
    ys_src = slice(max(0, dy), h + min(0, dy))
    ys_dst = slice(max(0, -dy), h + min(0, -dy))
    xs_src = slice(max(0, dx), w + min(0, dx))
    xs_dst = slice(max(0, -dx), w + min(0, -dx))
    res[ys_dst, xs_dst] = b[ys_src, xs_src]
    return res


def _skeleton_length_px(skel: np.ndarray) -> float:
    """
    估算骨架的真实弧长（像素）。

    为什么不能直接数前景像素：8 邻域下对角线相邻的两个点，
    实际几何距离是 sqrt(2) 而不是 1。直接计数会把斜线的长度低估/高估。
    正确做法是分别统计「水平/垂直相邻对」与「对角相邻对」，加权求和。

    统计口径（与原实现逐条一致）：对每个前景点，看它**右 / 下 / 右下 / 左下**
    四个方向的邻居是否也是前景，每条连接只被计一次。

    ★ 2026-09-26 改为向量化（`logs/_bench_extra_stages.txt` 实测它占 measure 6~14%）。
      等价性不靠碰运气：原实现用普通下标 + set 查找 ⇒ **越界坐标天然不在集合里**，
      即**无环绕**；本实现用 `_shift_no_wrap` 精确复刻这一点（见 `logs/_verify_skeleton_length.py`）。
      返回值的运算顺序与浮点常数与原来一致 ⇒ **位级相同**。
    """
    b = skel > 0
    if not b.any():
        return 0.0
    # 右 / 下 记为正交；右下 / 左下 记为对角
    n_orth = (int(np.count_nonzero(b & _shift_no_wrap(b, 0, 1)))
              + int(np.count_nonzero(b & _shift_no_wrap(b, 1, 0))))
    n_diag = (int(np.count_nonzero(b & _shift_no_wrap(b, 1, 1)))
              + int(np.count_nonzero(b & _shift_no_wrap(b, 1, -1))))
    return n_orth * 1.0 + n_diag * float(np.sqrt(2.0))


def measure_line_like(mask: np.ndarray, calib: Calibration,
                      cls_name: str = "crack") -> tuple[float, float, float, float]:
    """
    线状缺陷：返回 (length_px, width_mean_px, width_max_px, area_px)。

    长度：骨架化后按邻接关系加权求弧长（见 _skeleton_length_px），
          不用像素计数，否则斜线长度会失真。
    宽度：距离变换给出每个前景像素到背景的最近距离 r，
         则该点处宽度 = 2r。**只在骨架点上取宽度**，
         因为骨架点的 2r 才是该截面真实的局部宽度；
         对全前景取平均会被拐角/端点拉偏。
    """
    area_px = float(cv2.countNonZero(mask))
    if area_px <= 0:
        return 0.0, 0.0, 0.0, 0.0

    skel = _thin_skeleton(mask)
    length_px = _skeleton_length_px(skel)
    if length_px <= 0:
        # 骨架失败（如极小块），退化为面积/等效直径
        return 0.0, 0.0, 0.0, area_px

    # 距离变换：每个前景像素到最近零像素的欧氏距离
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    skel_pts = dist[skel > 0]
    if skel_pts.size == 0:
        return length_px, 0.0, 0.0, area_px

    widths = 2.0 * skel_pts          # 局部宽度 = 2 * 半径
    width_mean_px = float(np.mean(widths))
    # 用 95 分位而非 max，避免单个毛刺决定结果（更符合「实测最大值」的稳健估计）
    width_max_px = float(np.percentile(widths, 95))
    return length_px, width_mean_px, width_max_px, area_px


# --------------------------------------------------------------------------
# 块状缺陷测量
# --------------------------------------------------------------------------
def measure_blob_like(mask: np.ndarray) -> tuple[float, float, float]:
    """块状缺陷：返回 (area_px, equiv_diameter_px, perimeter_px)。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    equiv = float(np.sqrt(4.0 * area / np.pi)) if area > 0 else 0.0
    return area, equiv, perim


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------
def measure_instance(image: np.ndarray, bbox_xyxy, cls_name: str,
                     calib: Calibration,
                     pad: int = 2) -> Measurement | None:
    """
    对单个检测框做几何量化。

    参数：
      image    : 原始图（BGR）
      bbox_xyxy: (x1,y1,x2,y2) 像素坐标
      cls_name : 统一类别名
      calib    : 尺度标定
      pad      : 裁剪时向外扩的像素，避免把缺陷边缘切掉
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None

    roi = image[y1:y2, x1:x2]
    mask, method = segment_defect(roi, cls_name)
    if cv2.countNonZero(mask) == 0:
        return Measurement(cls_name=cls_name, bbox_xyxy=(x1, y1, x2, y2),
                           method=method + "(empty)")

    mmpp = calib.mm_per_px
    m = Measurement(cls_name=cls_name, bbox_xyxy=(x1, y1, x2, y2), method=method)

    if cls_name in ("crack", "exposed_rebar", "rust"):
        L, wm, wM, area = measure_line_like(mask, calib, cls_name)
        m.length_px, m.width_mean_px, m.width_max_px, m.area_px = L, wm, wM, area
        m.length_mm = L * mmpp
        m.width_mean_mm = wm * mmpp
        m.width_max_mm = wM * mmpp
        m.area_mm2 = area * mmpp * mmpp
        m.elongation = (L / wm) if wm > 1e-6 else 0.0
    else:
        area, equiv, _ = measure_blob_like(mask)
        m.area_px = area
        m.equiv_diameter_px = equiv
        m.area_mm2 = area * mmpp * mmpp
        m.length_px = m.width_max_px = equiv

    m.area_m2 = m.area_mm2 / 1e6
    roi_area = float((x2 - x1) * (y2 - y1))
    m.area_ratio = (m.area_px / roi_area) if roi_area > 0 else 0.0

    # ---- 测量窗口可用性自检（2026-09-23 新增；同日经功能实测修正）----------
    # ks 直接来自分割方法名（`blackhat{ks}+otsu` / `meddev{ks}+otsu`），
    # 因此**不必改动任何分割逻辑**就能拿到像素上界。
    import re as _re
    _mk = _re.search(r"(?:blackhat|meddev)(\d+)", m.method or "")
    m.ks = int(_mk.group(1)) if _mk else 0
    _px = float(m.width_max_px if cls_name in ("crack", "exposed_rebar", "rust")
                else m.equiv_diameter_px)
    _hi = (m.ks - 1) if m.ks > 0 else None

    # ⚠️ 本节能覆盖什么、不能覆盖什么（2026-09-23 功能实测后如实标注）：
    #
    # ✅ 能覆盖：**量出来的尺寸落在 [3, ks−1] 之外**的两种情形
    #    · `_px < MIN_RELIABLE_PX`               —— 太小，低于最小可判读门槛
    #    · `_px > ks−1`                          —— 太大，超出本 ROI 的可靠上界
    #
    # ❌ **不能覆盖**：「测量塌陷」—— 形态学核 `ks` 小于目标宽度时，
    #    黑帽会把宽目标抹平，量出的值**塌成一个小而"合规"的数**
    #    （实测 gsd=0.1 下 2mm 的带应占 20px，被量成约 0.26mm≈2.6px，
    #      该值落在 [3, ks−1] 内 ⇒ 本检查放行）。
    #    ⇒ 要检测它需要**独立于测量的参照**。本轮曾试「bbox 厚度比值」，
    #      但在带 24px 内边距的框上**全部误杀**（4/4 用例），故**已回退**。
    #    ⇒ 该检测器**记为待办**：须在真实测试集上校准阈值后才可上线。
    #      在它落地之前，这条**只能靠使用前提约束**（框不要开得太紧，见 §9.1）。
    if _px <= 0:
        m.window_ok = False
        m.window_note = "分割未得到有效尺寸"
    elif _px < MIN_RELIABLE_PX:
        m.window_ok = False
        m.window_note = ("目标在图上仅 %.2fpx，低于最小可判读门槛 %gpx"
                         % (_px, MIN_RELIABLE_PX))
    elif _hi is not None and _px > _hi:
        m.window_ok = False
        m.window_note = ("量出尺寸 %.2fpx 超出本 ROI 的可靠上界 %dpx"
                         "（形态学核 ks=%d）" % (_px, _hi, m.ks))
    else:
        m.window_ok = True
        m.window_note = ""
    return m


def draw_measurement(image: np.ndarray, m: Measurement, color=(0, 0, 255)
                     ) -> np.ndarray:
    """把测量结果画到图上（cv2.putText 不支持中文，用英文+数值）。"""
    out = image.copy()
    x1, y1, x2, y2 = m.bbox_xyxy
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    if m.cls_name in ("crack", "exposed_rebar", "rust") and m.length_mm > 0:
        label = f"{m.cls_name} L={m.length_mm:.0f}mm W={m.width_max_mm:.2f}mm"
    elif m.equiv_diameter_px > 0:
        label = f"{m.cls_name} area={m.area_m2:.3f}m2"
    else:
        label = m.cls_name

    ty = max(12, y1 - 6)
    cv2.putText(out, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                color, 1, cv2.LINE_AA)
    return out
