# -*- coding: utf-8 -*-
"""
_verify_rectify.py —— rectify.py 的合成数据验证（带解析真值）

为什么用合成图而不是真实照片：
    真实照片没有真值。"校正得对不对"无法证伪。
    本脚本用**真实的针孔相机模型 + 已知偏航角**渲染砖墙，
    于是墙面上任意长度的像素数可以解析算出，校正前后都能对照。

核心命题（这是 measure.py 能否报毫米结论的前提）：
    P1  斜拍图上，同一块墙在画面左端与右端的 mm/px 相差可观
        （说明 measure.py 的单一全局 mm/px 系统性错）
    P2  rectify() 能把两族直线摆正（横缝水平、竖缝竖直）
    P3  rectify() 之后，左/中/右三处量出的 mm/px 一致
    P4  rectify() 自报的 scale_mm_per_px 与**解析几何**推出的真值一致
        —— P4 是唯一能抓住「尺度算错但看起来正常」的口径：
           它完全不走自相关，只用已知的平面->图单应算 Δ像素。

失败路径：
    F1  无结构图 -> 必须拒答（lines_not_credible），不得静默返回原图
    F2  正对拍摄标准砌法（running bond）-> 竖缝太短检不出，
        必须以 `insufficient_lines_one_sided` 明确说明，
        **不得**假装已正对（竖缝没检出就没有证据说明它不收敛）
    F3  空输入 -> empty_image
拼接：
    S1  有唯一锚点（非周期参照物）的 3 段 -> 必须拼成功
    S2  严格周期、无任何锚点的 3 段 -> 必须拒答，且点名「周期性混叠」

写图一律用 common.imwrite_u 并**检查返回值**。
（实测踩坑：cv2.imwrite 在中文路径下静默失败、只返回 False 不抛异常 ——
 本项目 02_code 位于「D:\\pythonstudy 备份\\创新题\\...」，
 早期本脚本所有 cv2.imwrite 全部静默失败，报告里「已存 xxx.png」是假陈述。）

输出写到 _verify_out/verify_report.txt 与 verify_report.json。
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
import rectify as R  # noqa: E402

OUT_DIR = HERE / "_verify_out"
OUT_DIR.mkdir(exist_ok=True)

BRICK_PITCH = 250.0   # 240 + 10
BRICK_COURSE = 63.0   # 53 + 10

LINES: list[str] = []
RESULTS: dict = {}
IMG_ERRORS: list[str] = []


def say(s: str = "") -> None:
    LINES.append(s)


def hdr(t: str) -> None:
    say("")
    say("=" * 74)
    say(t)
    say("=" * 74)


def save(name: str, img) -> None:
    """写图并**校验**，不放过静默失败。"""
    ok = imwrite_u(OUT_DIR / name, img)
    if not ok:
        IMG_ERRORS.append(name)
        say(f"      !! 写图失败：{name}")
    else:
        say(f"      （已写出 {name}  {img.shape[1]}x{img.shape[0]}）")


# --------------------------------------------------------------------------
# 合成砖墙：真实针孔模型
# --------------------------------------------------------------------------
_PATCH_SEED = 20260920


def _to_px(Hp2i, pts_world):
    """把墙面世界坐标（mm）的点集投到像素坐标，供 fillPoly 用。"""
    pts = np.asarray(pts_world, dtype=np.float64)
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    p = (Hp2i @ hom.T).T
    return np.int32(np.stack([p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]], axis=1))


def _patch_polys():
    """世界坐标下（mm）几个不规则修补块：多边形顶点、灰度。

    为什么必须是**锐利边缘的不规则多边形**，而不是几个高斯斑：
        第一版 `rich` 用的是 4 个低频正弦 + 3 个 σ=240~300mm 的高斯深斑，
        幅度只有 ±6%/26%。实测在 4000 个 ORB 特征点的预算里，这些"软"起伏
        完全被砖角挤掉 —— 于是 S3（有纹理、无锚点）的内点占比只有 245/1600
        = 15.3%，比 15% 的门槛只高 0.3 个百分点，是**侥幸通过**。
        真实墙面的可匹配特征来自补砖、抹灰修补、涂料边界这类**带硬边**的东西。
    """
    rng = np.random.default_rng(_PATCH_SEED)
    out = []
    for (cu, cv_, rad, val) in ((260.0, -150.0, 200.0, 116.0),
                                (1520.0, 170.0, 230.0, 198.0),
                                (2640.0, -80.0, 190.0, 124.0),
                                (3320.0, 210.0, 250.0, 206.0),
                                (4180.0, -30.0, 210.0, 132.0)):
        k = 6
        ang = np.sort(rng.uniform(0.0, 2.0 * np.pi, k))
        rr = rad * rng.uniform(0.55, 1.0, k)
        pts = np.stack([cu + rr * np.cos(ang), cv_ + rr * np.sin(ang)], axis=1)
        out.append((pts, val))
    return out


# ❸ 局部砌法扰动区：(中心 u, 中心 v, 半径, 新水平模数, 水平相位, 新竖向模数, 竖向相位)
# 真实的修补区里，砖被切断重砌，模数和相位都与周边对不上 —— 这是把"严格周期"
# 局部破坏掉的关键。只靠色斑不够：色斑不改变 FAST 角点的**几何布局**，
# 而 ORB 的匹配错误正来自"每个砖角的几何布局都一样"。
# 实测（_diag7/第一次 run6）只用色块时，配对 2-3 的内点占比仍只有 11.9~18.4%，
# 位移摊成 90 多个峰 —— 说明色块没能提供足够的几何唯一性。
_ZONE_SPEC = ((150.0, -190.0, 380.0, 232.0, 90.0, 57.0, 11.0),
              (1250.0, 150.0, 330.0, 268.0, 42.0, 68.0, 27.0),
              (2350.0, -120.0, 430.0, 236.0, 118.0, 55.0, 19.0),
              (3400.0, 185.0, 300.0, 262.0, 70.0, 71.0, 33.0))


def _make_marks(n: int = 220):
    """世界坐标下密集的小损伤：缺角、掉皮、灰浆污染、孔洞。

    尺寸 9~34mm（≈3~12px），灰度刻意分两档（暗 70~118 / 亮 205~224），
    使它们在同一面墙上就彼此不同。这些小结构的总量必须能跟"砖角"争
    ORB 的 4000 个特征点预算，否则匹配仍由周期性砖角主导。
    """
    rng = np.random.default_rng(20260921)
    out = []
    for _ in range(n):
        mu = rng.uniform(-1400.0, 4100.0)
        mv = rng.uniform(-720.0, 720.0)
        hs = rng.uniform(9.0, 34.0)
        g = float(rng.choice([70.0, 88.0, 104.0, 118.0, 205.0, 216.0, 228.0]))
        out.append((mu, mv, hs, g))
    return out


PATCH_POLYS = _patch_polys()
MARKS = _make_marks()


def _square_world(cu, cv_, hs):
    """世界坐标下以 (cu,cv) 为中心、半边长 hs 的正方形（45° 旋转，边缘更杂）。"""
    d = hs * 0.7071
    return [(cu - d, cv_ - d), (cu + d, cv_ - d),
            (cu + d, cv_ + d), (cu - d, cv_ + d)]


def make_brick_view(f: float, cx: float, cy: float, d: float, yaw_deg: float,
                    W: int, H: int, u0: float, v0: float,
                    bond: str = "running", seed: int = 0, rich: bool = False):
    """
    渲染「相机绕竖直轴偏航 yaw 后看到的砖墙」，并返回平面->图的解析单应 Hp2i。

    相机系：X 右，Y 下，Z 前。墙平面过点 (0,0,d)，法线 (sin,0,cos)。
    墙上一点 (u,v) 毫米 -> 相机系 P = (u cos(yaw), v, d - u sin(yaw))，
    再经 K=[[f,0,cx],[0,f,cy],[0,0,1]] 投影。于是平面->像素的单应是解析可写的。

    rich=True 时额外叠加**世界坐标锚定**的非周期结构（三样，缺一不可）：
        · 局部砌法扰动区：区内水平/竖向模数与相位都改了 —— 破坏"几何周期"
        · 带硬边的不规则修补块 —— 给 ORB 提供大尺度可区分结构
        · 220 个小损伤（缺角/掉皮/孔洞，灰度各异）—— 提供充足的非周期特征
        · 一块低频色差（模拟污渍/光照不均，这一项对匹配**没有**实质贡献，
          保留只是为了像真墙）
    为什么必须锚定在世界坐标上：分段拍摄时相邻两段看到的是同一片墙，
    若这些结构随图像坐标随机，重叠区里两段的外观就不一致，特征匹配必然失败 ——
    那是造假的测试场景。真实墙面的污渍与修补当然也是世界锚定的。
    """
    th = np.radians(yaw_deg)
    A = f * np.cos(th) - cx * np.sin(th)
    H_plane2img = np.array([[A, 0.0, cx * d],
                            [-cy * np.sin(th), f, cy * d],
                            [-np.sin(th), 0.0, d]], dtype=np.float64)
    T = np.array([[1.0, 0.0, -u0], [0.0, 1.0, -v0], [0.0, 0.0, 1.0]])
    Hp2i = H_plane2img @ T
    Hinv = np.linalg.inv(Hp2i)

    xs, ys = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    grid = np.stack([xs, ys, np.ones_like(xs)], axis=-1) @ Hinv.T
    w = grid[..., 2]
    u = grid[..., 0] / w
    v = grid[..., 1] / w

    r = np.floor(v / BRICK_COURSE)                    # 第几皮砖
    vv = v - r * BRICK_COURSE
    off = 0.0 if bond == "stack" else (r % 2) * (BRICK_PITCH / 2.0)
    uu = np.mod(u - off, BRICK_PITCH)
    mortar_h = BRICK_PITCH - 10.0
    mortar_v = BRICK_COURSE - 10.0

    if rich:
        # ❶ 局部砌法扰动：区内把水平/竖向模数与相位都换掉。
        #    这是**破坏几何周期**的唯一手段 —— 只加色块不改变 FAST 角点的布局，
        #    而 ORB 的匹配歧义正来自"每个砖角的布局都长得一样"。
        for (zu, zv, zr, pu, phu, pv, phv) in _ZONE_SPEC:
            ins = ((u - zu) ** 2 + (v - zv) ** 2) < (zr ** 2)
            off_z = 0.0 if bond == "stack" else (r % 2) * (pu / 2.0)
            uu = np.where(ins, np.mod(u - off_z + phu, pu), uu)
            vv = np.where(ins, np.mod(v + phv, pv), vv)
            mortar_h = np.where(ins, pu - 10.0, mortar_h)
            mortar_v = np.where(ins, pv - 10.0, mortar_v)

    is_mortar = ((uu > mortar_h) | (vv > mortar_v))

    rng = np.random.default_rng(seed)
    c = np.floor((u - off) / BRICK_PITCH)
    key = r.astype(np.int64) * 7919 + c.astype(np.int64) * 104729
    tint = 150.0 + np.mod(key, 41).astype(np.float64)      # 逐砖 150..190
    img = np.where(is_mortar, 62.0, tint)

    if rich:
        # ❷ 硬边结构：修补块 + 220 个小损伤。
        #    做法是把世界坐标的多边形**投影**到图上再 fillPoly ——
        #    平面多边形经单应仍是多边形，所以这样画出来的位置与形状都精确，
        #    而且天然跨段一致（同一世界坐标 -> 同一投影结果）。
        #    （早期版本试过"先在世界栅格里 fillPoly 再按 (u,v) 索引回来"，
        #      数学上等价但要开一张 6800x21600 的栅格，单张就 147MB，已废弃。）
        mk = np.zeros((H, W), np.uint8)
        for pts, val in PATCH_POLYS:
            cv2.fillPoly(mk, [_to_px(Hp2i, pts)], int(val))
        for (mu, mv, hs, g) in MARKS:
            cv2.fillPoly(mk, [_to_px(Hp2i, _square_world(mu, mv, hs))], int(g))
        hit = mk > 0
        img = np.where(hit, mk.astype(np.float64), img)

        # ❸ 低频色差（污渍/光照不均）
        #    如实记录：这一项对特征匹配**没有**实质贡献（σ 太软、被砖角挤掉），
        #    它只是为了渲染得像一面真墙。不要误以为"有污渍"就等于"能拼上"。
        stain = (0.06 * np.sin(u / 620.0 + 1.3)
                 + 0.05 * np.sin(v / 410.0 - 0.4)
                 + 0.04 * np.sin((u + v) / 950.0 + 2.1)
                 + 0.03 * np.sin((u - 2.2 * v) / 730.0))
        img = img * (1.0 + stain)

    img = img + rng.normal(0.0, 3.0, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), Hp2i


def visible_u_range(img, Hp2i):
    """该段画面实际覆盖的墙面 u 范围（mm）。"""
    h, w = img.shape[:2]
    inv = np.linalg.inv(Hp2i)
    us = []
    for (x, y) in ((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)):
        p = inv @ np.array([float(x), float(y), 1.0])
        us.append(p[0] / p[2])
    return min(us), max(us)


def stamp_anchor(img, Hp2i, u_w: float, v_w: float, kind: str = "pattern",
                 variant: int = 0) -> bool:
    """在墙面世界坐标 (u_w, v_w) 处画一张纸片。

    用途：给严格周期的砖墙一个**唯一锚点**，使拼接有解。
    这正是 stitch_segments 拒答时建议用户做的事，这里用合成图证明该建议可行。

    kind = "plain"   —— 纯色黑框白心方块（41x41px，≈117mm）。
        实测：只要它**确实被相邻两段都拍到**，就真能救回拼接
        （配对内点占比 21.4%/17.8%），但余量不大（距 15% 门槛仅 +6.4/+2.8 个百分点）。
        ⚠️ 之前记录过"纯色小纸片无效（12.3%）"——那是**验证台自身的 bug**：
        纸片只画进了接缝左侧那一段，右侧压根没有，等于纸上只有一半。已作废。
    kind = "pattern" —— 白底 + 黑白棋盘格 + 黑边框，尺寸按 **A4 横向**取
        （本场景 2.857mm/px 下 104x73px）。内部分块产生几十个位置各异的强角点，
        整张纸片的相对布局在世界上唯一。

    variant —— 纸片编号。不同接缝必须用不同的 variant（= 带编号），
        否则各接缝的纸片外观相同，会**互相当替身**：实测把相同图案分贴两个接缝时，
        真解与诱饵解在全量匹配上解释出 626 : 625 个内点（比值 1.00，完全平票），
        于是只能拒答 —— 纸片自己变成了一个新的周期。这正是建议里"带编号"
        那几个字不可省的原因。

    ⚠️ 用法上还有讲究：正确的做法是让**同一张纸片**只出现在**相邻两段**的重叠区里
    （每个接缝贴一张、位置固定在世界坐标上）。若在每一段都盖一张外观相同的纸片，
    它们在特征空间里会互相冒充 —— 见 build_segments 的 "repeated_plain" 说明。
    """
    p = Hp2i @ np.array([float(u_w), float(v_w), 1.0])
    if abs(p[2]) < 1e-12:
        return False
    x, y = int(round(p[0] / p[2])), int(round(p[1] / p[2]))
    h, w = img.shape[:2]
    if not (24 <= x < w - 24 and 24 <= y < h - 24):
        return False
    if kind == "plain":
        cv2.rectangle(img, (x - 16, y - 16), (x + 16, y + 16), (0, 0, 0), -1)
        cv2.rectangle(img, (x - 8, y - 8), (x + 8, y + 8), (255, 255, 255), -1)
        return True
    # "pattern"：一张 A4 纸片（横向），白底压黑白格 + 黑边。
    # 尺寸为什么按 A4 来定：本场景 2.857mm/px，A4 横向 297x210mm -> 104x73px，
    # 半宽/半高 = 52/37。这样测出来的结论才是用户照着做能得到的结果。
    half_w, half_h = 52, 37
    cv2.rectangle(img, (x - half_w, y - half_h), (x + half_w, y + half_h),
                  (255, 255, 255), -1)
    # 图案由 variant 决定：同一接缝的两段取到同一张图案，不同接缝取到不同图案
    rng = np.random.default_rng(4242 + 1000 * int(variant))
    cell = 9
    for i in range(11):
        for j in range(8):
            if rng.random() < 0.5:
                x0 = x - half_w + 1 + i * cell
                y0 = y - half_h + 1 + j * cell
                cv2.rectangle(img, (x0, y0), (x0 + cell - 2, y0 + cell - 2),
                              (0, 0, 0), -1)
    cv2.rectangle(img, (x - half_w, y - half_h), (x + half_w, y + half_h),
                  (0, 0, 0), 2)
    return True


def make_gt(f: float, cx: float, d: float, yaw_deg: float):
    """真值函数 gt_mmpp(x)：图像横坐标 x 处，竖直方向的 mm/px。"""
    th = np.radians(yaw_deg)
    A = f * np.cos(th) - cx * np.sin(th)

    def gt_mmpp(x: float) -> float:
        u = d * (x - cx) / (A + x * np.sin(th))
        return (d - u * np.sin(th)) / f
    return gt_mmpp


def autosize_T_and_canvas(H, W: int, Hh: int):
    """复现 _warp_with_autosize 的画布计算，返回 (T, out_w, out_h)。

    为什么要自己复现：解析真值需要把整条变换链（H -> T1 -> S -> T2）
    显式写出来，才能算出「墙上 250mm 在最终图里占多少像素」。
    这里只复现 5 行画布数学，并断言与模块自报的 canvas 一致。
    """
    corners = np.array([[0, 0], [W, 0], [W, Hh], [0, Hh]], dtype=np.float64)
    pts = np.hstack([corners, np.ones((4, 1))]) @ H.T
    pr = pts[:, :2] / pts[:, 2:3]
    x0, y0 = pr[:, 0].min(), pr[:, 1].min()
    x1, y1 = pr[:, 0].max(), pr[:, 1].max()
    ow, oh = int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0))
    sc = 1.0
    if max(ow, oh) > R.MAX_CANVAS_SIDE:
        sc = R.MAX_CANVAS_SIDE / float(max(ow, oh))
        ow, oh = int(ow * sc), int(oh * sc)
    T = np.array([[sc, 0, -x0 * sc], [0, sc, -y0 * sc], [0, 0, 1.0]])
    return T, ow, oh


# --------------------------------------------------------------------------
# 主场景参数
# --------------------------------------------------------------------------
F = 1400.0
CX, CY = 800.0, 600.0
D = 4000.0
YAW = 18.0
IMG_W, IMG_H = 1600, 1200
U0, V0 = 0.0, 0.0

GT = make_gt(F, CX, D, YAW)
MMPP_CENTER = D / F          # 2.857 mm/px（正对时）

hdr("0. 场景参数与解析真值")
say(f"f={F}px  d={D}mm  偏航 yaw={YAW}deg  画面 {IMG_W}x{IMG_H}  主点=({CX},{CY})")
say(f"正对时中心 mm/px = d/f = {MMPP_CENTER:.4f}")
say("")
say("解析真值 —— 同一面墙、同一竖直方向，在画面不同横坐标处的 mm/px：")
say(f"{'x(px)':>7} {'真值mm/px':>11} {'竖直砖距应为(px)':>17}")
probe_xs = [0, 200, 400, 800, 1200, 1400, 1599]
gt_vals = {}
for x in probe_xs:
    g = GT(x)
    gt_vals[x] = g
    say(f"{x:>7} {g:>11.4f} {BRICK_COURSE / g:>17.2f}")
ratio_gt = GT(0) / GT(IMG_W - 1)
say("")
say(f"**左端 / 右端 = {ratio_gt:.3f}x**  -> 相差 {(ratio_gt - 1) * 100:.1f}%")
say("这就是「单一全局 mm/px」失效的直接证据：同一道 0.3mm 裂缝，")
say("出现在画面左边会被量成 0.3mm，出现在右边会被量成 0.3/%.3f = %.3fmm。"
    % (ratio_gt, 0.3 / ratio_gt))
RESULTS["ground_truth"] = {"mmpp_center": MMPP_CENTER,
                           "mmpp_by_x": {str(k): v for k, v in gt_vals.items()},
                           "left_over_right": ratio_gt}

# --------------------------------------------------------------------------
# 1. 渲染斜拍图（两种砌法）
# --------------------------------------------------------------------------
hdr("1. 渲染合成斜拍砖墙图")
imgs, Hp2is = {}, {}
for bond in ("running", "stack"):
    img, Hp = make_brick_view(F, CX, CY, D, YAW, IMG_W, IMG_H, U0, V0, bond=bond)
    imgs[bond], Hp2is[bond] = img, Hp
    say(f"[{bond}] 渲染 {img.shape[1]}x{img.shape[0]}")
    save(f"synth_{bond}.png", img)

# --------------------------------------------------------------------------
# 2. P1：斜拍图上，左右两端测出的周期是否真的不同
# --------------------------------------------------------------------------
hdr("2. P1 验证 —— 未校正图上左右两端尺度不一致")
w = IMG_W
bands = {"left": (0, int(w * 0.22)), "mid": (int(w * 0.39), int(w * 0.61)),
         "right": (int(w * 0.78), w)}


def strip_period_px(img: np.ndarray, x0: int, x1: int, axis: str):
    """在指定列范围内量竖直砖距（像素）。用模块自己的 _modulus_px，保证口径一致。"""
    crop = img[:, max(0, x0):min(img.shape[1], x1)]
    if crop.size == 0:
        return None, {"status": "empty_crop"}
    m = R._modulus_px(crop, axis, max_period=max(120.0, 4.0 * BRICK_COURSE))
    return m.get("period_px"), m


p1 = {}
for bond in ("running", "stack"):
    say("")
    say(f"--- 砌法 = {bond} ---")
    row = {}
    for name, (x0, x1) in bands.items():
        pv, dv = strip_period_px(imgs[bond], x0, x1, "y")
        if pv is None:
            say(f"  {name:>5} [{x0:>4},{x1:>4})  竖直周期: 检出失败 {dv.get('status')}")
            row[name] = None
        else:
            mm = BRICK_COURSE / pv
            xc = (x0 + x1) / 2.0
            say(f"  {name:>5} [{x0:>4},{x1:>4})  周期={pv:7.2f}px  推出={mm:6.3f}mm/px"
                f"   真值={GT(xc):6.3f}   偏差={(mm / GT(xc) - 1) * 100:+6.1f}%")
            row[name] = {"period_px": round(pv, 3), "mmpp": round(mm, 4),
                         "gt": round(GT(xc), 4)}
    if row.get("left") and row.get("right"):
        r = row["left"]["mmpp"] / row["right"]["mmpp"]
        say(f"  => 实测左/右 = {r:.3f}x")
        row["ratio_left_over_right"] = round(r, 4)
    p1[bond] = row
RESULTS["P1_unrectified"] = p1

# --------------------------------------------------------------------------
# 3. 线族检出诊断
# --------------------------------------------------------------------------
hdr("3. 线族检出诊断 —— _split_line_families 能否找到两族直线")
fam = {}
for bond in ("running", "stack"):
    gray = cv2.cvtColor(imgs[bond], cv2.COLOR_BGR2GRAY)
    fam_v, fam_h, info = R._split_line_families(gray)
    enough = (len(fam_v) >= R.MIN_LINES_PER_FAMILY
              and len(fam_h) >= R.MIN_LINES_PER_FAMILY)
    say(f"  [斜拍/{bond}] 原始线段={info.get('n_segments_raw')}  "
        f"竖直族={info.get('n_vertical')}  水平族={info.get('n_horizontal')}  "
        f"主方向={info.get('dominant_angle_deg')}deg  canny={info.get('canny')}")
    say(f"      enough_lines = {enough}（两族各需 >= {R.MIN_LINES_PER_FAMILY} 条）")
    fam[bond] = {"n_vertical": info.get("n_vertical"),
                 "n_horizontal": info.get("n_horizontal"),
                 "dominant_angle_deg": info.get("dominant_angle_deg"),
                 "enough_lines": bool(enough)}
RESULTS["line_families"] = fam

# --------------------------------------------------------------------------
# 4. P2/P3/P4：校正是否生效、校正后尺度是否统一、自报尺度是否与解析真值一致
# --------------------------------------------------------------------------
hdr("4. P2/P3/P4 验证 —— 摆正、尺度统一、自报尺度可信")


def family_angles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fam_v, fam_h, _ = R._split_line_families(gray)
    out = {}
    for tag, f in (("v", fam_v), ("h", fam_h)):
        if len(f) == 0:
            out[tag] = None
            continue
        ang = np.degrees(np.arctan2(-f[:, 0], f[:, 1])) % 180.0
        out[tag] = (int(len(f)), round(float(np.median(ang)), 2))
    return out


rect_res = {}
for bond in ("running", "stack"):
    say("")
    say(f"--- 砌法 = {bond} ---")
    img, Hp2i = imgs[bond], Hp2is[bond]
    per = R.estimate_perspective(img)
    say(f"  校正前：enough_lines={per['enough_lines']} "
        f"credible={per['credible']} already_flat={per['already_flat']} "
        f"vp_v_dist_norm={per['vp_v_dist_norm']} vp_h_dist_norm={per['vp_h_dist_norm']}")

    try:
        res = R.rectify(img)
    except Exception:
        say("  !! rectify() 抛异常：")
        for ln in traceback.format_exc().splitlines():
            say("     " + ln)
        rect_res[bond] = {"exception": traceback.format_exc()}
        continue

    say(f"  rectify -> applied={res.applied} method={res.method} "
        f"conf={res.confidence} uniform_scale={res.uniform_scale}")
    say(f"      message: {res.message}")
    say(f"      advice : {res.advice}")
    say(f"      scale_mm_per_px = {res.scale_mm_per_px}")
    say(f"      canvas={res.detail.get('canvas')}  "
        f"axis_residual={res.detail.get('axis_residual')}")
    if not res.applied:
        rect_res[bond] = res.to_dict()
        continue

    save(f"rect_{bond}.png", res.image)

    # P2：摆正了没有
    ang = family_angles(res.image)
    say(f"      [P2 摆正] 校正后 竖缝 n={ang['v'][0] if ang['v'] else 0} "
        f"中位倾角={ang['v'][1] if ang['v'] else None}deg（应≈90）；"
        f"横缝 n={ang['h'][0] if ang['h'] else 0} "
        f"中位倾角={ang['h'][1] if ang['h'] else None}deg（应≈0）")

    # P3：三处尺度一致
    w2 = res.image.shape[1]
    b2 = {"left": (0, int(w2 * 0.22)), "mid": (int(w2 * 0.39), int(w2 * 0.61)),
          "right": (int(w2 * 0.78), w2)}
    vals = {}
    say("      [P3 尺度一致性] 校正后左/中/右三处：")
    for name, (x0, x1) in b2.items():
        pv, dv = strip_period_px(res.image, x0, x1, "y")
        if pv is None:
            say(f"          {name:>5} 检出失败 {dv.get('status')}")
            vals[name] = None
        else:
            vals[name] = BRICK_COURSE / pv
            say(f"          {name:>5} 周期={pv:7.2f}px  ->  {vals[name]:6.3f} mm/px")
    ok = [v for v in vals.values() if v]
    spread = (max(ok) / min(ok)) if len(ok) >= 2 else None
    if spread:
        say(f"          => 最大/最小 = {spread:.3f}x"
            f"（校正前左右为 {p1[bond].get('ratio_left_over_right')}x）")

    # P4：解析几何真值 vs 模块自报
    # 链条：plane ->(Hp2i 解析)-> 原图 ->(T1@H)-> A ->(S)-> A' ->(T2)-> 最终图
    mh, mv = res.detail["modulus_h"], res.detail["modulus_v"]
    mha, mva = res.detail["modulus_h_after"], res.detail["modulus_v_after"]
    p4 = {"analytic_mm_x": None, "analytic_mm_y": None, "reported": res.scale_mm_per_px}
    s_y = res.detail.get("y_scale_applied")
    if s_y:
        H = np.array(res.homography, dtype=np.float64)
        T1, aw, ah = autosize_T_and_canvas(H, IMG_W, IMG_H)
        A_chain = (T1 @ H) @ Hp2i
        S = np.array([[1.0, 0, 0], [0, s_y, 0], [0, 0, 1.0]])
        T2, fw, fh = autosize_T_and_canvas(S, aw, ah)
        # 校验复现的画布与模块自报的最终画布一致（不一致说明链条写错了）
        assert abs(res.detail["canvas"][0] - fw) <= 1 and \
            abs(res.detail["canvas"][1] - fh) <= 1, \
            f"画布复现不一致 {(fw, fh)} vs {res.detail['canvas']}"
        assert fw == res.image.shape[1] and fh == res.image.shape[0], \
            f"复现画布与返回图尺寸不一致 {(fw, fh)} vs {res.image.shape}"
        Htot = (T2 @ S) @ A_chain
        Pc = np.linalg.inv(Hp2i) @ np.array([CX, CY, 1.0])
        u_c, v_c = Pc[0] / Pc[2], Pc[1] / Pc[2]

        def px_of(du, dv):
            p0 = Htot @ np.array([u_c, v_c, 1.0])
            p1_ = Htot @ np.array([u_c + du, v_c + dv, 1.0])
            a, b = p0[:2] / p0[2], p1_[:2] / p1_[2]
            return float(np.hypot(*(b - a)))

        dx250, dy63 = px_of(BRICK_PITCH, 0), px_of(0, BRICK_COURSE)
        mm_x_true, mm_y_true = BRICK_PITCH / dx250, BRICK_COURSE / dy63
        say(f"      [P4 解析真值] 最终图中：250mm→{dx250:.3f}px、63mm→{dy63:.3f}px"
            f"  →  mm/px: x={mm_x_true:.4f}  y={mm_y_true:.4f}"
            f"  两轴各向异性={abs(mm_x_true / mm_y_true - 1) * 100:.2f}%")
        say(f"          最终图尺寸 {fw}x{fh}（模块自报 canvas={res.detail['canvas']}）")
        if res.scale_mm_per_px:
            err = abs(res.scale_mm_per_px / mm_x_true - 1) * 100
            say(f"          模块自报 {res.scale_mm_per_px:.4f}  vs 解析真值 "
                f"{mm_x_true:.4f}  →  偏差 {err:.2f}%"
                f"{'  ✓ 通过(<2%)' if err < 2 else '  ✗ 不通过'}")
            p4.update({"analytic_mm_x": round(mm_x_true, 4),
                       "analytic_mm_y": round(mm_y_true, 4),
                       "err_pct": round(err, 3)})
        else:
            say("          模块未给 scale_mm_per_px（刻意的拒答）")
        p4["final_canvas"] = [fw, fh]

    rect_res[bond] = {"result": res.to_dict(), "after_mmpp": vals, "spread": spread,
                      "families_after": ang, "P4": p4}
RESULTS["rectify"] = rect_res

# --------------------------------------------------------------------------
# 5. 失败路径
# --------------------------------------------------------------------------
hdr("5. 失败路径 —— 能不能「正确地拒绝」")

# F3 空
res_empty = R.rectify(None)
say(f"[F3 空图] applied={res_empty.applied} reason={res_empty.detail.get('error')}")
RESULTS["F3_empty"] = res_empty.to_dict()

# F1 无结构（纯噪声）
noise = np.random.default_rng(7).integers(0, 255, (600, 800, 3), dtype=np.uint8)
res_noise = R.rectify(noise)
say("")
say(f"[F1 噪声] rectify -> applied={res_noise.applied} method={res_noise.method} "
    f"conf={res_noise.confidence} reason={res_noise.detail.get('reason')}")
say(f"           message: {res_noise.message}")
say(f"           advice : {res_noise.advice}")
RESULTS["F1_noise"] = res_noise.to_dict()

# F2 正对 running bond（竖缝太短，检不出）
img_flat, Hp2i_flat = make_brick_view(F, CX, CY, D, 0.0, IMG_W, IMG_H, U0, V0,
                                      bond="running")
save("synth_flat_running.png", img_flat)
res_flat = R.rectify(img_flat)
per_flat = R.estimate_perspective(img_flat)
say("")
say(f"[F2 正对/running] estimate_perspective: enough={per_flat['enough_lines']} "
    f"n_v={per_flat['n_vertical']} n_h={per_flat['n_horizontal']} "
    f"already_flat={per_flat['already_flat']}")
say(f"           rectify -> applied={res_flat.applied} "
    f"reason={res_flat.detail.get('reason')}")
say(f"           message: {res_flat.message}")
say(f"           advice : {res_flat.advice}")
RESULTS["F2_flat_running"] = {"perspective": per_flat, "result": res_flat.to_dict()}

# F2b 正对 stack bond（竖缝连续，应能检出）
img_flat_s, _ = make_brick_view(F, CX, CY, D, 0.0, IMG_W, IMG_H, U0, V0,
                                bond="stack")
res_flat_s = R.rectify(img_flat_s)
say("")
say(f"[F2b 正对/stack] rectify -> applied={res_flat_s.applied} "
    f"method={res_flat_s.method} uniform_scale={res_flat_s.uniform_scale} "
    f"reason={res_flat_s.detail.get('reason')}")
say(f"           message: {res_flat_s.message}")
RESULTS["F2b_flat_stack"] = res_flat_s.to_dict()

# --------------------------------------------------------------------------
# 6. 拼接
# --------------------------------------------------------------------------
hdr("6. 拼接验证")
STEP_MM = 1285.0          # 相邻段平移，约 50% 重叠
SEG_W, SEG_H = 900, 600
SEG_CX, SEG_CY = SEG_W / 2.0, SEG_H / 2.0     # 段的主点居中（否则可视 u 范围不对称）
SEG_SPAN_MM = SEG_W * D / F                   # 一段覆盖的墙面宽度（mm）


def build_segments(rich: bool, anchor: str, tag: str):
    """渲染 3 段。

    rich   —— 墙上是否有世界锚定的非周期结构（局部砌法扰动 + 修补块 + 小损伤）
    anchor —— 纸片方案，四种：
        "none"             不贴任何纸片
        "per_joint_pattern" 在每个**相邻接缝**贴一张**带棋盘格图案**的纸片。
                            纸片位置固定在世界坐标上，只出现在相邻两段里 ——
                            这是拒答建议①的本意。
        "per_joint_plain"   同上，但纸片是**纯色**方块（用来隔离"带图案"这件事的作用）
        "repeated_plain"    在**每一段**都贴一张**外观相同**的纯色方块。
                            ⚠️ 这是第一版验证台犯的错：那些锚点都在各自段中心附近的
                            **同一画面位置**、外观完全相同，却对应相差 1285mm 的世界点 ——
                            等于在墙上贴了三张一模一样的纸片。那不是"唯一锚点"，
                            而是"周期更粗（1285mm）的另一个周期"，测错了对象。
                            保留它作为**负面对照**。
    """
    segs = []
    u_lo = u_hi = None
    # 纸片种类 + 是否"按接缝编号"
    # ⚠️ 别把 "repeated_plain" 也放进这个表里：它的语义是"每段各贴一张"，
    #    与"按接缝贴"完全不同。第一版误放进来了，于是 S6 静默退化成 S5 的复制，
    #    两组数字一模一样（401/308、1816x610）才露馅。
    spec = {"per_joint_numbered": ("pattern", True),
            "per_joint_samepat": ("pattern", False),
            "per_joint_plain": ("plain", False)}
    for i in range(3):
        seg, Hp = make_brick_view(F, SEG_CX, SEG_CY, D, 0.0, SEG_W, SEG_H,
                                  i * STEP_MM, 0.0, bond="stack", seed=i, rich=rich)
        if u_lo is None:
            u_lo, u_hi = visible_u_range(seg, Hp)
        # 接缝 j 的重叠区（绝对世界坐标）= 段 j 与段 j+1 的公共部分，取中点。
        # 一共只有 n-1 = 2 个接缝：最后一段之后没有下一段，在那里贴纸片只会凭空
        # 多出一个没有对应物的锚点（自己给自己制造诱饵）。
        if anchor in spec:
            kind, numbered = spec[anchor]
            # ⚠️ 关键一：同一张纸片必须画进**共享该接缝的两段**（段 j 与段 j+1）。
            # 第一版只把它画进段 j —— 等于纸上只有一半，右段压根没有这张纸，
            # 特征当然匹配不上。stamp_anchor 对"落在画面外"会自己返回 False，
            # 所以这里放心把两个接缝都试一遍。
            # ⚠️ 关键二：numbered=True 时每张纸片的图案不同（= 带编号）。
            # 若各接缝的纸片长得一样，它们会互相当替身 —— 见下方 S4 场景。
            for j in (i - 1, i):
                if not (0 <= j < 2):
                    continue
                j_u = j * STEP_MM + (STEP_MM + u_lo + u_hi) / 2.0
                ok = stamp_anchor(seg, Hp, j_u, 0.0, kind=kind,
                                  variant=(j if numbered else 0))
                if ok:
                    say(f"  [{tag}] 段 {i + 1}: 接缝{j + 1}纸片 u={j_u:.0f}mm"
                        f"（{kind}{'，图案编号=' + str(j) if numbered else '，图案固定'}）画入")
        elif anchor == "repeated_plain":
            j_u = i * STEP_MM + (STEP_MM + u_lo + u_hi) / 2.0
            ok = stamp_anchor(seg, Hp, j_u, 0.0, kind="plain")
            say(f"  [{tag}] 段 {i + 1}: 相同外观的纯色锚点 u={j_u:.0f}mm "
                f"画入={'成功' if ok else '失败(在画面外)'}")
        segs.append(seg)
        save(f"{tag}_seg{i}.png", seg)
    say(f"  [{tag}] 可视 u 范围 = [{u_lo:.0f}, {u_hi:.0f}]mm（跨度 "
        f"{u_hi - u_lo:.0f}mm），相邻平移 {STEP_MM}mm"
        f" → 重叠约 {100 * (1 - STEP_MM / (u_hi - u_lo)):.0f}%\n"
        f"         纹理={'有（砌法扰动+修补块+小损伤）' if rich else '无（纯周期砖格）'}，"
        f"纸片={anchor}")
    return segs


def run_stitch(segs, key, tag, expect):
    try:
        st = R.stitch_segments(segs)
    except Exception:
        say(f"  [{tag}] !! stitch_segments 抛异常：")
        for ln in traceback.format_exc().splitlines():
            say("       " + ln)
        RESULTS[key] = {"exception": traceback.format_exc()}
        return
    say(f"  [{tag}] ok={st.ok} conf={st.confidence} n_used={st.n_used}/{st.n_input}"
        f"  method={st.method}   （预期：{expect}）")
    say(f"           message: {st.message}")
    if st.advice:
        say(f"           advice : {st.advice}")
    # 把每个配对的「内点占比 vs 门槛」余量也打出来。
    # 为什么要打：S3 曾经以 15.3% 对 15% 的门槛"通过"，只高 0.3 个百分点，
    # 若只报「成功」，读者会以为拼接是稳健的，而实际上它是踩线侥幸。
    for pr in (st.detail.get("pairs") or []):
        if "inlier_ratio" not in pr:
            continue
        mg = (pr["inlier_ratio"] - 0.15) * 100
        say(f"           对 {pr['pair'][0]}-{pr['pair'][1]}: 匹配 {pr['n_matches']}、"
            f"内点 {pr['inliers']}（占比 {pr['inlier_ratio']:.1%}，"
            f"距 15% 门槛 {mg:+.1f} 个百分点）")
    if st.image is not None:
        save(f"stitched_{tag}.png", st.image)
        exp = SEG_W + 2 * STEP_MM * F / D
        say(f"           拼接宽度 {st.image.shape[1]}px，预期约 {exp:.0f}px，"
            f"偏差 {(st.image.shape[1] / exp - 1) * 100:+.1f}%")
    RESULTS[key] = st.to_dict()


say("S1 = 纯周期砖格、不贴纸片 → 应拒答，并点名是**周期歧义**（而非『拍糊了』）")
seg_s1 = build_segments(rich=False, anchor="none", tag="s1")
run_stitch(seg_s1, "S1_periodic_bare", "S1", "拒答（周期歧义）")

say("")
say("S2 = 真实感墙面（砌法扰动+修补块+小损伤）、不贴纸片 → 应成功。")
say("     最重要的一条：真实墙面本身就有非周期结构，多数情况**不需要**贴纸片")
seg_s2 = build_segments(rich=True, anchor="none", tag="s2")
run_stitch(seg_s2, "S2_rich_bare", "S2", "成功")

say("")
say("S3 = 纯周期砖格 + 每个接缝贴一张 A4 棋盘格纸片，**每张图案不同（=带编号）**")
say("     → 这是推荐做法，应成功")
seg_s3 = build_segments(rich=False, anchor="per_joint_numbered", tag="s3")
run_stitch(seg_s3, "S3_periodic_numbered", "S3", "成功")

say("")
say("S4 = 纯周期砖格 + 每个接缝贴一张 A4 棋盘格纸片，但**每张图案完全一样**")
say("     → 预期拒答：相同外观的纸片自己构成一个新周期，两解平票。")
say("     这一条就是建议里『带编号』那三个字的实测依据")
seg_s4 = build_segments(rich=False, anchor="per_joint_samepat", tag="s4")
run_stitch(seg_s4, "S4_periodic_samepat", "S4", "拒答（纸片自己成了新周期）")

say("")
say("S5 = 纯周期砖格 + 接缝处共用一张**纯色**小纸片（117mm）→ 如实报告。")
say("     预期能成功，但余量应明显小于带图案的 A4")
seg_s5 = build_segments(rich=False, anchor="per_joint_plain", tag="s5")
run_stitch(seg_s5, "S5_periodic_plain", "S5", "看结果")

say("")
say("S6 = 负面对照：纯周期砖格 + **每段各贴一张**外观相同的纯色方块 → 预期失败。")
say("     这是第一版验证台犯的错（把重复标记当成了唯一锚点）")
seg_s6 = build_segments(rich=False, anchor="repeated_plain", tag="s6")
run_stitch(seg_s6, "S6_periodic_repeated", "S6", "拒答（互相当替身）")


# --------------------------------------------------------------------------
# 6b. POS 先验（合成注入）—— 2026-09-25 新增
# --------------------------------------------------------------------------
# 目的：验证「把无人机 POS 当作单应初值」这一环真的能用，并验证**注入错误 POS 时
#       系统会兜底拒答**（而不是照着错 POS 拼出一个错误的立面还宣称成功）。
#
# ⚠️ 诚实边界：这是**合成注入**，不是真机航拍。报告里不得写「已实现无人机测绘」。
#
# 几何换算（与 make_brick_view 自洽）：
#   世界单位是 mm，相机离墙 D=4000mm，段间沿墙平移 STEP_MM=1285mm，无偏航。
#   ⇒ 第 i 段的 POS = {dx_m=i*STEP_MM/1000, d_m=D/1000, yaw=pitch=roll=0}
#     内参 K = [[F,0,cx],[0,F,cy],[0,0,1]]（cx=SEG_CX, cy=SEG_CY）
hdr("6b. POS 先验（合成注入）")

K_SYN = np.array([[F, 0.0, SEG_CX],
                  [0.0, F, SEG_CY],
                  [0.0, 0.0, 1.0]], dtype=np.float64)


def _syn_poses(n: int = 3, dx_m_override: dict | None = None) -> list:
    """按合成几何生成 n 段位姿（dx_m_override 可覆盖某段的 dx_m，用于注错）。"""
    ps = []
    for i in range(n):
        dx_m = i * STEP_MM / 1000.0
        if dx_m_override and i in dx_m_override:
            dx_m = dx_m_override[i]
        ps.append(dict(dx_m=dx_m, dy_m=0.0, d_m=D / 1000.0,
                       yaw=0.0, pitch=0.0, roll=0.0))
    return ps


def run_stitch_pose(segs, key, tag, expect, poses, K):
    """带 POS 的拼接（并打印 POS 相关诊断）。"""
    try:
        st = R.stitch_segments(segs, poses=poses, K=K)
    except Exception:
        say(f"  [{tag}] !! stitch_segments(poses=...) 抛异常：")
        for ln in traceback.format_exc().splitlines():
            say("       " + ln)
        RESULTS[key] = {"exception": traceback.format_exc()}
        return
    say(f"  [{tag}] ok={st.ok} conf={st.confidence} n_used={st.n_used}/{st.n_input}"
        f"  method={st.method}   （预期：{expect}）")
    say(f"           message: {st.message[:200]}")
    for pr in (st.detail.get("pairs") or []):
        bits = []
        if "pose_prior" in pr:
            bits.append(f"先验初值={'有' if pr['pose_prior'] else '无'}")
        if "pose_filter" in pr:
            pf = pr["pose_filter"]
            bits.append(f"先验筛后保留 {pf.get('kept')} 对"
                        + (f"（{pf.get('fallback')}）" if pf.get("fallback") else ""))
        if "pose_dev_px" in pr:
            bits.append(f"精修偏离 {pr['pose_dev_px']}px")
        if bits:
            say(f"           对 {pr['pair'][0]}-{pr['pair'][1]}: " + "；".join(bits))
    if st.image is not None:
        save(f"stitched_{tag}.png", st.image)
        say(f"           拼接宽度 {st.image.shape[1]}px")
    RESULTS[key] = st.to_dict()


say("S7 = 纯周期砖格 + **正确的 POS 先验**（合成注入）")
say("     预期：POS 把匹配限制到正确重叠区 ⇒ 有望拼成功（周期砖格本来会拒答）")
seg_s7 = build_segments(rich=False, anchor="none", tag="s7")
run_stitch_pose(seg_s7, "S7_pose_correct", "S7", "成功或如实报告", _syn_poses(), K_SYN)

say("")
say("S8 = 纯周期砖格 + **错误的 POS**（把第 3 段 dx 强行改成 5 米，实际约 2.57 米）")
say("     预期：**必须拒答**并点名『POS 先验与图像证据不一致』——")
say("     证明 POS 不可信时系统会兜底，而不是照着错 POS 拼出错立面")
seg_s8 = build_segments(rich=False, anchor="none", tag="s8")
run_stitch_pose(seg_s8, "S8_pose_wrong", "S8", "拒答（POS 不一致）",
                _syn_poses(dx_m_override={2: 5.0}), K_SYN)

say("")
say("S9 = 纯周期砖格 + **带噪声的 POS**（每段 dx 加 ±0.02m≈2cm 抖动）")
say("     预期：小噪声不应导致误拒 —— 用来标定”守卫阈值是否过紧“")
_rng9 = np.random.RandomState(20260925)
_noisy = _syn_poses()
for _p in _noisy:
    _p["dx_m"] += float(_rng9.uniform(-0.02, 0.02))
seg_s9 = build_segments(rich=False, anchor="none", tag="s9")
run_stitch_pose(seg_s9, "S9_pose_noisy", "S9", "不误拒（成功）或如实报告",
                _noisy, K_SYN)


st1 = R.stitch_segments([seg_s2[0]])
say("")
say(f"[单段] ok={st1.ok} method={st1.method} msg={st1.message}")
RESULTS["S_single"] = st1.to_dict()

# --------------------------------------------------------------------------
# 收尾
# --------------------------------------------------------------------------
hdr("完成")
if IMG_ERRORS:
    say(f"⚠️ 有 {len(IMG_ERRORS)} 张图写出失败：{IMG_ERRORS}")
else:
    say("所有图片均已写出并通过返回值校验。")
say(f"全部输出见 {OUT_DIR}")

(OUT_DIR / "verify_report.txt").write_text("\n".join(LINES), encoding="utf-8")
(OUT_DIR / "verify_report.json").write_text(
    json.dumps(RESULTS, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8")
print("\n".join(LINES))
