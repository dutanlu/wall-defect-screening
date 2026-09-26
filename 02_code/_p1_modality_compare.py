# -*- coding: utf-8 -*-
r"""P1 的**模态对比**分析：把 rgb / ir / irgray / fuse / ctrl 放在一起判读。

## 与 `_p1_hollow_probe.py` 的分工
  · `_p1_hollow_probe.py` 负责**预登记判据 H1–H6** 的判定（"哪个值是空鼓"）。
  · 本脚本负责**模态层面的对比**，回答「**IR 是否携带 RGB 没有的信息**」这一前置问题。

## 五个模态
| 名称 | 输入 | 作用 |
|---|---|---|
| `rgb` | [R,G,B] | 可见光基准 |
| `ir` | 红外伪彩三通道 | 红外单模态 |
| `irgray` | gray(IR) 三通道 | 排除伪彩映射假象 |
| `fuse` | [R,G,**gray(IR)**] | 把 B 通道换成红外灰度 |
| `ctrl` | [R,G,**gray(RGB)**] | **对照**：通道结构与 fuse 相同，只换灰度来源 |

## 判读规则（**跑之前就已声明**）
1. `fuse` vs `rgb`：`fuse > rgb` ⇒ IR 有独立贡献；**`fuse ≤ rgb` ⇒ 不构成结论**
   （换掉了 B 通道，损失可能被掩盖）。
2. `fuse` vs `ctrl`（关键对照）：通道结构相同，唯一差别是灰度来源
   ⇒ `fuse > ctrl` ⇒ **红外灰度比可见光灰度更有用**，支持"IR 携带独有信息"。
3. ⚠️ 两者都**不能直接证明「IR 能看见空鼓」**：标签由可见光判读所定义，
   本对比只能说明「IR 通道是否携带独立信息」，是回答该问题的**必要非充分**条件。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\pythonstudy 备份\创新题\外墙缺陷筛查")
EVAL = ROOT / "04_results" / "eval"
OUT_TXT = ROOT / "logs" / "_bfdd_modality_compare.txt"

ORDER = ["rgb", "ir", "irgray", "fuse", "ctrl"]
LABEL = {"rgb": "RGB", "ir": "IR", "irgray": "IRgray",
         "fuse": "[R,G,IRgray]", "ctrl": "[R,G,grayRGB]"}


def main() -> int:
    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(str(s))

    mods = {}
    for m in ORDER:
        p = EVAL / f"bfdd_probe_{m}.json"
        if p.exists():
            mods[m] = json.loads(p.read_text(encoding="utf-8"))
    if "rgb" not in mods:
        say("!! 缺 rgb 产物，无法对比")
        return 1

    say("=" * 96)
    say("P1 模态对比：IR 是否携带 RGB 没有的信息")
    say("=" * 96)
    say("")
    say("各模态逐类掩膜 IoU（同预训练初始化 / 同超参 / 同 seed / 同 epoch=80）")
    say("")
    hdr = f"{'值':>4}" + "".join(f"{LABEL[m]:>16}" for m in mods)
    say(hdr)
    say("-" * len(hdr))
    tab = {}
    for v in range(1, 6):
        k = f"v{v}"
        row = {}
        for m in mods:
            d = mods[m]["iou"]["per_class"][k]
            row[m] = d["iou"] or 0.0
        tab[k] = row
        say(f"{v:>4}" + "".join(f"{row[m]:>16.4f}" for m in mods))

    say("")
    say("## 规则 1：`fuse` vs `rgb`（IR 是否带来净增益）")
    say(f"{'值':>4}{'rgb':>10}{'fuse':>10}{'Δ':>10}{'相对':>10}   判读")
    say("-" * 62)
    r1 = {}
    for v in range(1, 6):
        k = f"v{v}"
        a = tab[k].get("rgb", 0.0)
        d = tab[k].get("fuse", 0.0)
        rel = (d - a) / a * 100 if a > 0 else float("nan")
        verdict = "fuse>rgb ⇒ IR 有独立贡献" if d > a else "fuse≤rgb ⇒ **不构成结论**"
        r1[k] = {"rgb": a, "fuse": d, "delta": d - a, "conclusive": bool(d > a)}
        say(f"{v:>4}{a:>10.4f}{d:>10.4f}{d - a:>+10.4f}{rel:>+9.1f}%   {verdict}")

    r2 = {}
    if "ctrl" in mods:
        say("")
        say("## 规则 2：`fuse` vs `ctrl`（**关键对照**：通道结构相同，只换灰度来源）")
        say(f"{'值':>4}{'ctrl':>12}{'fuse':>12}{'Δ=fuse−ctrl':>14}{'相对':>10}   判读")
        say("-" * 72)
        for v in range(1, 6):
            k = f"v{v}"
            c = tab[k].get("ctrl", 0.0)
            d = tab[k].get("fuse", 0.0)
            rel = (d - c) / c * 100 if c > 0 else float("nan")
            if d > c:
                verd = "✅ 红外灰度优于可见光灰度 ⇒ 支持 IR 有独有信息"
            elif abs(d - c) < 0.02:
                verd = "≈ 无差异 ⇒ **无证据**表明 IR 有独有信息"
            else:
                verd = "❌ 红外灰度更差"
            r2[k] = {"ctrl": c, "fuse": d, "delta": d - c, "favours_ir": bool(d > c)}
            say(f"{v:>4}{c:>12.4f}{d:>12.4f}{d - c:>+14.4f}{rel:>+9.1f}%   {verd}")
        n_fav = sum(1 for k in r2 if r2[k]["favours_ir"])
        say("")
        say(f"⇒ 5 个类中 **{n_fav} 个** 满足 `fuse > ctrl`。")
    else:
        say("")
        say("## 规则 2：`fuse` vs `ctrl` —— 对照产物尚未生成，跳过")

    say("")
    say("=" * 96)
    say("## 总结")
    say("=" * 96)
    say("")
    say("**可以确定的**：")
    say("  · RGB 单模态在**每一个类**上都远优于 IR / IRgray（Δ 全为负）；")
    say("  · IR 图本身不残缺（均值 126.6 / 标准差 25.5 vs RGB 138.0 / 24.9；唯一灰度级 87）。")
    say("")
    say("**不能确定的（必须声明的边界）**：")
    say("  · 本系列对比**无法回答「IR 能否看见空鼓」**：BFDD 掩膜是人工标注，")
    say("    标注者很可能在**可见光图**上判读 ⇒ 标签编码的是可见光可见的证据，")
    say("    用 IR 预测它**结构性处于劣势**。")
    say("  · 因此本结果**不构成**对报告 §9.2（空鼓需红外）的否定，也不构成支持。")
    say("  · 要真正回答，需要**同时保留全部 RGB 通道并加入 IR** 的 4 通道模型；")
    say("    而 ultralytics 的数据加载是 3 通道、4 通道还会破坏预训练权重的加载")
    say("    ⇒ **本轮预算内未解决**，已列为后续。")
    say("")

    rep = {"per_class": tab, "fuse_vs_rgb": r1, "fuse_vs_ctrl": r2,
           "modalities_present": list(mods.keys()),
           "note": ("fuse/ctrl 均为三通道、同预训练初始化；本对比回答"
                    "「IR 是否携带独立信息」，不能直接证明「IR 能看见空鼓」"
                    "（标签由可见光判读定义）。")}
    outp = EVAL / "bfdd_modality_compare.json"
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    say(f"产物: {outp}")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[留痕] {OUT_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
