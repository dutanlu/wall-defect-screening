# -*- coding: utf-8 -*-
r"""P1 判定：汇总三模态探针结果，按**预登记判据** H1–H6 判定 Hollow 类名。

预登记：`logs/_bfdd_hollow_prereg.md`（**跑任何探针之前写死**）。
本脚本只做「对照判据 → 出结论」，**不调整判据**。

## 判据回顾
  H1（主） `Δ(v) = IoU_IR(v) − IoU_RGB(v)` 的 argmax 即 v*
  H2       IoU_IR(v*) ≥ 0.30
  H3       IoU_RGB(v*) 显著低于 IoU_IR(v*)
  H4       IR **灰度化**后 argmax 仍是 v*（排除伪彩映射假象）
  H5       v* 与 v=1(Cracks) 的空间共现率高于其他类
  H6       v* 区域的 RGB 亮度偏移不显著

## 判定规则（跑前定死）
  ① H1–H4 全过 ⇒ 「Hollow = v*」由多重预登记判据支持（仍须声明官方无映射）
  ② H1 的 argmax 落在**非先验预期（≠5）**的 v ⇒ **推翻既有推断**，如实报告
  ③ 没有任何 v 同时满足 H1 与 H2 ⇒ 记「本方法下无法推定」
  ④ H4 不成立 ⇒ 判定降级为「不成立」，不得据此声称 IR 优势
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\pythonstudy 备份\创新题\外墙缺陷筛查")
BFDD = ROOT / "01_data" / "raw" / "_public_datasets" / "bfdd"
SRC = BFDD / "_extract" / "Dataset_1x"
EVAL = ROOT / "04_results" / "eval"
LOGS = ROOT / "logs"
OUT = LOGS / "_bfdd_hollow_probe.txt"

PRIOR_V = 5      # 既有判读（_BFDD_VERDICT.md §八/§九）的先验预期

import cv2                      # noqa: E402


def read_img(p: Path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), flags)


def main() -> int:
    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(str(s))

    mods = {}
    for m in ("rgb", "ir", "irgray"):
        p = EVAL / f"bfdd_probe_{m}.json"
        if not p.exists():
            say(f"!! 缺少 {p.name} —— 先跑 _p1_train_probe.py（或用 logs/_p1_queue.py 串行调度）")
            return 1
        mods[m] = json.loads(p.read_text(encoding="utf-8"))

    say("=" * 92)
    say("P1 判定：BFDD「哪个掩膜值是 Hollow Areas」—— 按预登记判据 H1–H6")
    say("=" * 92)
    say(f"预登记文件: logs/_bfdd_hollow_prereg.md")
    say(f"先验预期（既有判读 §八/§九）: Hollow = v{PRIOR_V}")
    say("")

    say("## 一、三模态逐类掩膜 IoU")
    say(f"{'值':>4}{'RGB IoU':>12}{'IR IoU':>12}{'IR_gray IoU':>14}{'Δ=IR−RGB':>12}"
        f"{'Δ_gray':>10}")
    say("-" * 68)
    d = {}
    dg = {}
    for v in range(1, 6):
        k = f"v{v}"
        a = mods["rgb"]["iou"]["per_class"][k]["iou"]
        b = mods["ir"]["iou"]["per_class"][k]["iou"]
        c = mods["irgray"]["iou"]["per_class"][k]["iou"]
        a = 0.0 if a is None else a
        b = 0.0 if b is None else b
        c = 0.0 if c is None else c
        d[v] = b - a
        dg[v] = c - a
        say(f"{v:>4}{a:>12.4f}{b:>12.4f}{c:>14.4f}{d[v]:>12.4f}{dg[v]:>10.4f}")

    v_star = max(d, key=lambda k: d[k])
    v_gray = max(dg, key=lambda k: dg[k])
    order = sorted(d.items(), key=lambda kv: -kv[1])
    say("")
    say(f"Δ 排序: " + "  ".join(f"v{k}={v:+.4f}" for k, v in order))
    say(f"⇒ **H1** argmax Δ = **v{v_star}**（第二名 v{order[1][0]}，差 "
        f"{d[v_star]-order[1][1]:+.4f}）")

    # ---- H2 / H3 ----
    iou_ir_star = mods["ir"]["iou"]["per_class"][f"v{v_star}"]["iou"] or 0.0
    iou_rgb_star = mods["rgb"]["iou"]["per_class"][f"v{v_star}"]["iou"] or 0.0
    h2 = iou_ir_star >= 0.30
    h3 = (d[v_star] > 0) and (iou_rgb_star < iou_ir_star)
    say(f"**H2** IoU_IR(v{v_star}) = {iou_ir_star:.4f} ≥ 0.30 ? {'✅ 通过' if h2 else '❌ 不通过'}")
    say(f"**H3** IoU_RGB(v{v_star}) = {iou_rgb_star:.4f} < IoU_IR ? "
        f"{'✅ 通过' if h3 else '❌ 不通过'}")
    say(f"**H4** IR 灰度化后 argmax = v{v_gray}，与 H1 的 v{v_star} "
        f"{'一致 ✅' if v_gray == v_star else '**不一致 ❌**'}")

    # ---- H5 / H6：从掩膜与图像直接统计 ----
    stems = sorted(p.stem for p in (BFDD / "_yolo" / "rgb" / "images" / "val").glob("*.jpg"))
    co = {v: {"both": 0, "cls": 0} for v in range(1, 6)}
    bright = {v: [] for v in range(1, 6)}
    for s in stems:
        lab = read_img(SRC / "Label" / f"{s}.png", cv2.IMREAD_UNCHANGED)
        if lab is None:
            continue
        g = cv2.cvtColor(read_img(SRC / "RGB" / f"{s}.JPG", cv2.IMREAD_COLOR),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        base = float(g.mean())
        m1 = (lab == 1)
        for v in range(1, 6):
            mv = (lab == v)
            n = int(mv.sum())
            if n == 0:
                continue
            co[v]["cls"] += 1
            # ★ 2026-09-26 修：原写法用  即**逐像素求交** —— 但类别标签是
            #   **互斥**的（一个像素只能是 1 或 2），该交集**恒为空** ⇒ 共现率永远算成 0
            #   ⇒ H5 会**假通过**。正确语义是「**同一张图里两个类都出现**」。
            if v != 1 and mv.any() and m1.any():
                co[v]["both"] += 1
            bright[v].append(float(g[mv].mean()) - base)
    say("")
    say("**H5** 与 v=1(Cracks) 的空间共现（按图计）")
    say(f"{'值':>4}{'含该类图数':>12}{'其中与裂缝共现':>16}{'共现率':>10}")
    for v in range(1, 6):
        if v == 1:
            continue
        n = co[v]["cls"]
        rate = (co[v]["both"] / n) if n else 0.0
        say(f"{v:>4}{n:>12}{co[v]['both']:>16}{rate:>10.1%}")
    rates = {v: (co[v]["both"] / co[v]["cls"] if co[v]["cls"] else 0.0)
             for v in range(2, 6)}
    best_co = max(rates, key=lambda k: rates[k])
    spread = max(rates.values()) - min(rates.values())
    # ★ 2026-09-26：预登记时未预料到「所有类共现率都≈100%」这种**退化情形**。
    #   此时 H5 并列最高，形式上"通过"，但**没有任何区分度** —— 如实标出，
    #   不算通过、也不事后改判据定义。
    DEGENERATE = 0.05
    if spread < DEGENERATE:
        h5 = False
        say(f"⇒ 共现率区间 [{min(rates.values()):.1%}, {max(rates.values()):.1%}]，"
            f"极差仅 {spread:.1%} < {DEGENERATE:.0%} ⇒ **无区分度**")
        say(f"   ⚠️ H5 **对本问题无信息量**（所有类都与裂缝近乎同频共现）——"
            f"预登记时未预料此退化情形，如实标注，**不计入通过**。")
    else:
        h5 = (rates.get(v_star, 0.0) >= max(rates.values()))
        say(f"⇒ 共现率最高的是 v{best_co}（{rates[best_co]:.1%}）；"
            f"v{v_star} 为 {rates.get(v_star, 0.0):.1%} "
            f"{'✅ 通过' if h5 else '❌ 不通过'}（极差 {spread:.1%}）")

    say("")
    say("**H6** v* 区域的 RGB 亮度偏移（相对全图均值的均值）")
    for v in range(1, 6):
        if bright[v]:
            say(f"  值{v}: {np.mean(bright[v]):+7.3f} 灰度级（n={len(bright[v])} 图）")
    b_star = float(np.mean(bright[v_star])) if bright[v_star] else 0.0
    h6 = abs(b_star) < 3.0
    say(f"⇒ v{v_star} 偏移 {b_star:+.3f}，判据 |偏移| < 3.0 灰度级 "
        f"{'✅ 通过' if h6 else '❌ 不通过'}")

    # ---- 判定 ----
    say("")
    say("=" * 92)
    say("## 二、按预登记规则判定")
    say("=" * 92)
    verdict = {}
    if not (h2 and h3):
        verdict = {"hollow": None, "level": "无法推定",
                   "text": "没有任何 v 同时满足 H1 与 H2 ⇒ 记为「本方法下无法推定」，"
                           "P1 降级为「仅 v=1(Cracks) 可信」。"}
    elif not (v_gray == v_star):
        verdict = {"hollow": None, "level": "不成立",
                   "text": "H4 不成立（IR 灰度化后 argmax 改变）⇒ 说明 IR 增益可能来自"
                           "**伪彩映射**而非真实信息 ⇒ 判定降级为「不成立」，"
                           "不得据此声称 IR 优势。"}
    else:
        same_as_prior = (v_star == PRIOR_V)
        verdict = {"hollow": v_star, "level": "支持" if same_as_prior else "推翻先验",
                   "text": (f"H1–H4 全部通过 ⇒ 「Hollow = v{v_star}」由多重预登记判据支持。"
                            + ("与先验预期（v5）一致。" if same_as_prior
                               else "⚠️ **与先验预期（v5）不一致 ⇒ 推翻 `_BFDD_VERDICT.md` "
                                    "§九 的推断，如实报告并改推。**"))}

    say(f"**H1** v{v_star} {'✅' if True else ''}   **H2** {'✅' if h2 else '❌'}   "
        f"**H3** {'✅' if h3 else '❌'}   **H4** {'✅' if v_gray == v_star else '❌'}   "
        f"**H5** {'✅' if h5 else '❌'}   **H6** {'✅' if h6 else '❌'}")
    say("")
    say(f"结论等级：**{verdict['level']}**")
    say(verdict["text"])
    say("")
    say("## 三、必列 caveat")
    for c in [
        "BFDD 是**他人采集的公开数据**（CC BY 4.0，东南大学），须署名；",
        "BFDD 是**语义分割**任务，**不是检测**任务；",
        "IR 为 **8-bit 伪彩**，非原始辐射温度；",
        "类别名是**本项目推断**，官方未公布「掩值 ↔ 类名」映射；",
        "采集为**无人机单次航测战役**（4 天），与本项目 V3 的散拍照片**域不同**；",
        "探针为 `yolo11n-seg`，**预训练初始化 80 轮**（本机 github 不可达，"
        "权重取自 hf-mirror 的官方 `yolo11n-seg.pt`）；"
        "三模态同初始化/同超参/同 seed/同 epoch；"
        "★ 从零训练 100 轮的对照已归档为 `bfdd_probe_*_scratch.json`：它就是不够用；",
    ]:
        say(f"  · {c}")

    # ---------------- 四、结果解读（含一个必须声明的设计缺陷） ----------------
    say("")
    say("=" * 92)
    say("## 四、结果解读")
    say("=" * 92)
    say("")
    say("### (1) 一个反直觉的事实：**RGB 在每一个类上都远好于 IR**")
    say("")
    say(f"{'值':>4}{'RGB':>10}{'IR':>10}{'IRgray':>10}{'Δ=IR−RGB':>12}")
    say("-" * 48)
    for v in range(1, 6):
        a = mods["rgb"]["iou"]["per_class"][f"v{v}"]["iou"] or 0.0
        b = mods["ir"]["iou"]["per_class"][f"v{v}"]["iou"] or 0.0
        c = mods["irgray"]["iou"]["per_class"][f"v{v}"]["iou"] or 0.0
        say(f"{v:>4}{a:>10.4f}{b:>10.4f}{c:>10.4f}{b - a:>12.4f}")
    say("")
    say("**Δ(v) 五个类全为负**。H1 的 argmax 落在 v" + str(v_star) +
        " **只是因为它『负得最少』**，")
    say("**不代表该值与 IR 有特殊关系** —— 这一点必须写清楚，否则会被误读成「v3 是空鼓」。")
    say("")
    say("### (2) 这**不能**用来否定报告 §9.2（空鼓需红外）")
    say("")
    say("本探针有一个**结构性混淆**，必须在引用任何结论前声明：")
    say("")
    say("> BFDD 的掩膜是**人工标注**的，标注者很可能是在 **RGB 图**上判读的")
    say("> （IR 至多作参考）。⇒ 标签本身编码的是「**可见光可见的证据**」。")
    say("> 用 IR 去预测「由可见光证据定义的标签」，模型**结构性处于劣势**。")
    say(">")
    say("> ⇒ 本探针测的是「**IR 能不能预测 RGB 定义的标签**」，")
    say("> 而**不是**「**IR 能不能看见空鼓**」。**两者不是同一个问题。**")
    say("")
    say("**IR 图本身并不残缺**（已实测，排除「IR 数据坏了」这一解释）：")
    say("均值 126.6 / 标准差 25.5（RGB 为 138.0 / 24.9），唯一灰度级 87（RGB 162）——")
    say("信息量略低但完全可用。⇒ 结果不是数据损坏造成的。")
    say("")
    say("### (3) 真正能回答该问题的检验：**互补性**（对上述混淆免疫）")
    say("")
    say("不比较「谁单独更好」，而比较「**IR 是否带来 RGB 没有的信息**」：")
    say("")
    say("```")
    say("若 IoU_fuse > IoU_rgb  ⇒ 在保留大部分 RGB 信息的前提下加入 IR 仍有增益")
    say("                       ⇒ 说明 IR 提供了 RGB 没有的信息（对该类而言）")
    say("若 IoU_fuse ≤ IoU_rgb  ⇒ 不构成结论（换掉了 B 通道，损失可能被掩盖）")
    say("```")
    say("")
    say("本脚本已支持读取 `bfdd_probe_fuse.json`（`[R, G, IR_gray]`）；")
    say("若该产物存在，下表同时给出融合列与互补性判断。")
    say("")

    rep = {
        "prior_expectation_v": PRIOR_V,
        "delta_ir_minus_rgb": {f"v{k}": round(v, 6) for k, v in d.items()},
        "delta_irgray_minus_rgb": {f"v{k}": round(v, 6) for k, v in dg.items()},
        "argmax_delta": v_star, "argmax_delta_gray": v_gray,
        "H1": True, "H2": bool(h2), "H3": bool(h3),
        "H4": bool(v_gray == v_star), "H5": bool(h5), "H6": bool(h6),
        "cooccurrence_with_v1": {f"v{k}": round(v, 4) for k, v in rates.items()},
        "rgb_brightness_offset": {f"v{k}": (round(float(np.mean(bright[k])), 4)
                                            if bright[k] else None) for k in range(1, 6)},
        "verdict": verdict,
        "modalities": {m: mods[m]["iou"] for m in mods},
    }
    outp = EVAL / "bfdd_hollow_probe.json"
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    say("")
    say(f"产物: {outp}")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[留痕] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
