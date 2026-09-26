# -*- coding: utf-8 -*-
"""
_survey_demo.py —— 「整立面测绘」端到端演示（合成场景，可复现）

===== 这份脚本要证明什么 =====

用户问「是不是可以做一个比较完整的系统，比如加上无人机测绘」。
本脚本不声称用了无人机（**无设备、无真实航拍数据，绝不造假**），
而是做一件可验证的事：

    用合成渲染的「分段立面照片」走完整链路：
        分段拍摄 → 正射校正 → 多段拼接 → 整立面统一尺度 → 整墙能力评估

并回答「**这块墙，以这套拍摄配置，能测到什么**」——
把 §7.7 的几何判据从"单张图"升到"**整立面**"。

===== 为什么必须用合成场景 =====

实测：V3 测试集 285 张是**散拍照片**（greybrick / hrcds / rebar / urban 四个来源），
**互不构成同一立面的相邻段** ⇒ 无法用真实数据集验证拼接。
（这也解释了 `_verify_rectify.py` 当初为什么用合成场景：不是偷懒，是**没有别的办法**。）

===== 复用而非重写 =====

渲染器、拼接器全部复用 `_verify_rectify.py` 的 `make_brick_view` / `R.stitch_segments`，
本脚本只加「拼接后的整立面能力评估」这一层。

输出：logs/_survey_demo/{survey_report.txt, survey_report.json, *.png}
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

# 复用既有渲染器与拼接器
import _verify_rectify as V  # noqa: E402
import rectify as R  # noqa: E402
from advice import gsd_at_distance, advise_distance, LENS_PRESETS  # noqa: E402
from gsd import Calibration, screening_capability  # noqa: E402
from common import imwrite_u  # noqa: E402

OUT = Path(CODE).resolve().parent / "logs" / "_survey_demo"
OUT.mkdir(parents=True, exist_ok=True)

LINES: list[str] = []
RESULTS: dict = {}


def say(s: str = "") -> None:
    LINES.append(s)


def save_png(name: str, img) -> None:
    """落盘演示图。

    ⚠️ 必须用 `imwrite_u`：本机实测 `cv2.imwrite` 对含非 ASCII 的路径
    **返回 False 且不抛异常**（本工程路径含中文）⇒ 原写法下这些 PNG
    **一张都写不出来且完全无声**，而技术报告 §11.7 却把它们列为产物。
    2026-09-26 发现并修正；改后成功/失败都会打一行，不再静默。
    """
    if img is None:
        return
    p = OUT / name
    if imwrite_u(p, img):
        say(f"  [图] {p.name}")
    else:
        say(f"  [图][落盘失败] {p.name}")


def main() -> int:
    say("=" * 78)
    say("整立面测绘 · 端到端演示（合成场景）")
    say("=" * 78)
    say("")
    say("⚠️ 前提声明：本演示用**合成渲染的砖墙立面**，不是无人机航拍。")
    say("   本机无无人机设备、无真实航拍数据 ⇒ 不声称任何航拍实测。")
    say("")

    # ---------------------------------------------------------------- 1. 分段
    say("─" * 78)
    say("1. 分段拍摄（模拟沿墙面平移拍 3 段，相邻重叠约 50%）")
    say("─" * 78)
    try:
        segs = V.build_segments(rich=True, anchor="none", tag="survey")
    except Exception:
        say("!! 分段渲染失败：")
        for ln in traceback.format_exc().splitlines():
            say("   " + ln)
        return 1
    for i, s in enumerate(segs):
        save_png(f"seg{i}.png", s)
        say(f"   段{i+1}: {s.shape[1]}x{s.shape[0]}")
    say("")

    # ---------------------------------------------------------------- 2. 拼接
    say("─" * 78)
    say("2. 多段拼接 → 整立面图（stitch_segments，ORB+RANSAC+羽化）")
    say("─" * 78)
    try:
        st = R.stitch_segments(segs)
    except Exception:
        say("!! 拼接抛异常：")
        for ln in traceback.format_exc().splitlines():
            say("   " + ln)
        return 1
    say(f"   ok={st.ok}  conf={st.confidence}  n_used={st.n_used}/{st.n_input}"
        f"  method={st.method}")
    say(f"   message: {st.message}")
    if st.advice:
        say(f"   advice : {st.advice}")
    for pr in (st.detail.get("pairs") or []):
        if "inlier_ratio" in pr:
            mg = (pr["inlier_ratio"] - 0.15) * 100
            say(f"   对 {pr['pair'][0]}-{pr['pair'][1]}: 内点占比 "
                f"{pr['inlier_ratio']:.1%}（距 15% 门槛 {mg:+.1f} 个百分点）")
    RESULTS["stitch"] = st.to_dict()
    if st.image is None:
        say("!! 拼接未产出图像 ⇒ 无法进入能力评估（这本身是**正确的拒答**）")
        _dump()
        return 2
    canvas = st.image
    save_png("stitched_whole.png", canvas)
    say(f"   整立面图尺寸: {canvas.shape[1]}x{canvas.shape[0]}")
    say("")

    # ---------------------------------------------------------------- 3. 尺度
    say("─" * 78)
    say("3. 整立面统一尺度（mm/px）—— 从「拼图本身」反推")
    say("─" * 78)
    # 用已知渲染参数给出的世界尺度：F 焦距(px)、D 距离(mm) ⇒ 正对时 mm/px = D/F
    mm_per_px = float(V.D) / float(V.F)
    say(f"   渲染参数: 焦距 F={V.F:.0f}px, 距离 D={V.D:.0f}mm")
    say(f"   ⇒ 正对时整立面统一 mm/px = D/F = {mm_per_px:.4f} mm/px")
    calib = Calibration(mm_per_px=mm_per_px, method="synthetic_gt", confidence="high")
    # 覆盖宽度
    cover_m = canvas.shape[1] * mm_per_px / 1000.0
    say(f"   整立面覆盖宽度 = {canvas.shape[1]}px × {mm_per_px:.4f} = {cover_m:.2f} m")
    RESULTS["scale"] = dict(mm_per_px=mm_per_px, cover_m=round(cover_m, 3),
                            canvas_w=int(canvas.shape[1]), canvas_h=int(canvas.shape[0]))
    say("")

    # ---------------------------------------------------------------- 4. 能力
    say("─" * 78)
    say("4. ★ 整立面能力评估 —— 以这套拍摄配置，这面墙能测到什么")
    say("─" * 78)
    cap = screening_capability(calib)
    say(f"   {'目标':<24}{'实际尺寸':>10}{'图上像素':>10}{'能否测':>10}{'级别':>14}")
    for name, row in cap["capability"].items():
        say(f"   {name:<24}{row['target_mm']:>9.2f}mm{row['pixels']:>9.2f}px"
            f"{'是' if row['usable'] else '否':>10}{row['level']:>14}")
    RESULTS["capability"] = cap
    say("")

    # ---------------------------------------------------------------- 5. 建议
    say("─" * 78)
    say("5. 若判不了，该怎么补拍（advice.py 反解，与 gsd.calibrate_by_camera 互逆）")
    say("─" * 78)
    img_w = int(canvas.shape[1])
    for lens in ["主摄 24mm", "长焦 48mm"]:
        say(f"   ── 镜头：{lens}（整立面宽 {img_w}px）──")
        for mm, lab in [(0.3, "裂缝定级 0.3mm"), (1.0, "明显裂缝 1.0mm"),
                        (10.0, "小剥落 10mm")]:
            # 直接用 advise_distance（不手搓公式）⇒ 与 app.py 用的同一函数，口径必然一致
            a = advise_distance(mm, img_w, lens)
            # 往返校验：用反解出的距离再正算 GSD，应与 required_gsd 一致。
            # ⚠️ max_distance_m 在 advise_distance 内已 round(…,3) ⇒ 反算会有 ~1e-4 量级
            #    舍入差，故用相对容差 1% 判定，并如实把两个数都打出来（不假装严格相等）。
            gsd_check = gsd_at_distance(a.max_distance_m, img_w, LENS_PRESETS[lens])
            rel = abs(gsd_check - a.required_gsd) / max(a.required_gsd, 1e-12)
            flag = "OK" if rel < 0.01 else "!!"
            say(f"      {lab:<18} 需 GSD≤{a.required_gsd:.4f} ⇒ 距离≤{a.max_distance_m:.3f} m"
                f"（反算 GSD={gsd_check:.4f}，相对差 {rel:.1%} {flag}，可行={a.feasible}）")
    say("")

    # ---------------------------------------------------------------- 6. 结论
    say("─" * 78)
    say("6. 结论")
    say("─" * 78)
    say("   ① 分段→拼接→整立面→能力评估**链路打通**（合成场景，可复现）")
    say(f"   ② 整立面图 {canvas.shape[1]}x{canvas.shape[0]}，覆盖 {cover_m:.2f} m 宽")
    n_usable = sum(1 for r in cap["capability"].values() if r["usable"])
    n_total = len(cap["capability"])
    say(f"   ③ 该配置下 {n_usable}/{n_total} 类目标可判读")
    say("   ④ ⚠️ 真实航拍数据缺失 ⇒ 本演示**不等于**无人机实测，报告须如实声明")
    say("")
    say("完成。产物见 " + str(OUT))

    _dump()
    return 0


def _dump() -> None:
    txt = "\n".join(LINES)
    (OUT / "survey_report.txt").write_text(txt, encoding="utf-8")
    (OUT / "survey_report.json").write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
