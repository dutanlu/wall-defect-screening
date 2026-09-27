# -*- coding: utf-8 -*-
r"""把 `rgbir` 纳入模态对比：在 `_p1_modality_compare.py` 的产物上**追加**一组判读。

## 为什么要单独一个脚本
`_p1_modality_compare.py` 的 ORDER 是硬编码的 5 个模态，且总结段里写死了
「要真正回答需要 4 通道模型 ⇒ **本轮预算内未解决**」这句话。
本脚本**不改**它（保持历史产物口径不变），而是：
  1. 读入它的产物 `bfdd_modality_compare.json`；
  2. 追加读取 `bfdd_probe_rgbir.json`；
  3. 写出**新**产物 `bfdd_modality_compare_rgbir.json` + 新结论文本。

## 新增的判读（**跑之前声明**）
| 对比 | 含义 | 判读 |
|---|---|---|
| `rgbir` vs `rgb` | **完整 RGB + 外加 IR** ⇒ 无「丢掉 B」的混淆 | `>` ⇒ IR 有独立增益（强证据）；`≈` ⇒ 无证据；`<` ⇒ 外加 IR 有害 |
| `rgbir` vs `fuse` | 两者都含 IR；差别是**是否保留 B** 与**是否多一路** | `>` ⇒ 保留 B 有价值 / 多通道有增益 |
| `rgbir` vs `ctrl` | 同上，对照的是「可见光灰度」 | 同向解读 |

⚠️ 三类结果都**仍不能**证明「IR 能看见空鼓」——BFDD 掩膜由可见光判读定义。
   本对比只回答「**IR 通道是否携带独立信息**」这一必要非充分条件。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\pythonstudy 备份\创新题\外墙缺陷筛查")
EVAL = ROOT / "04_results" / "eval"
OUT_TXT = ROOT / "logs" / "_bfdd_modality_compare_rgbir.txt"

LABELS = {"rgb": "RGB", "ir": "IR", "irgray": "IRgray",
          "fuse": "[R,G,IRgray]", "ctrl": "[R,G,grayRGB]",
          "rgbir": "[R,G,B,IR]★4ch"}


def main() -> int:
    base_p = EVAL / "bfdd_modality_compare.json"
    new_p = EVAL / "bfdd_probe_rgbir.json"
    if not base_p.exists():
        print(f"!! 缺 {base_p}")
        return 1
    if not new_p.exists():
        print(f"!! 缺 {new_p} —— 4 通道探针尚未产出，先跑 _bfdd_rgbir_probe.py")
        return 2

    base = json.loads(base_p.read_text(encoding="utf-8"))
    rgbir = json.loads(new_p.read_text(encoding="utf-8"))
    tab = base["per_class"]
    rgbir_pc = rgbir["iou"]["per_class"]

    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(str(s))

    say("=" * 100)
    say("模态对比（含 4 通道）：IR 是否携带 RGB 没有的信息")
    say("=" * 100)
    say("")
    say(f"4 通道探针：epochs={rgbir.get('epochs')}（落盘 {rgbir.get('epochs_landed')}）  "
        f"imgsz={rgbir.get('imgsz')}  batch={rgbir.get('batch')}  seed={rgbir.get('seed')}  "
        f"workers={rgbir.get('workers')}  用时={rgbir.get('train_minutes')} min")
    say(f"初始化：{rgbir.get('init')}")
    say(f"通道序：{rgbir.get('channel_order')}")
    say("")
    say("各模态逐类掩膜 IoU（同预训练初始化 / 同超参 / 同 seed / 同 epoch=80 / 同划分）")
    say("")
    mods = ["rgb", "ir", "irgray", "fuse", "ctrl", "rgbir"]
    hdr = f"{'值':>4}" + "".join(f"{LABELS[m]:>18}" for m in mods)
    say(hdr)
    say("-" * len(hdr))
    full = {}
    for v in range(1, 6):
        k = f"v{v}"
        row = {}
        for m in mods:
            if m == "rgbir":
                row[m] = rgbir_pc[k]["iou"] or 0.0
            else:
                row[m] = tab.get(k, {}).get(m) or 0.0
        full[k] = row
        say(f"{v:>4}" + "".join(f"{row[m]:>18.4f}" for m in mods))

    def cmp_block(name: str, ref_key: str, title: str):
        say("")
        say(f"## {title}")
        say(f"{'值':>4}{ref_key:>14}{'rgbir':>14}{'Δ=rgbir−'+ref_key:>16}{'相对':>10}   判读")
        say("-" * 82)
        res = {}
        n_gt = 0
        for v in range(1, 6):
            k = f"v{v}"
            a = full[k].get(ref_key, 0.0)
            d = full[k]["rgbir"]
            rel = (d - a) / a * 100 if a > 0 else float("nan")
            if d > a:
                verd = f"✅ rgbir 优于 {ref_key}"
                n_gt += 1
            elif abs(d - a) < 0.02:
                verd = "≈ 无差异（噪声内）"
            else:
                verd = f"❌ rgbir 更差"
            res[k] = {"ref": a, "rgbir": d, "delta": d - a, "better": bool(d > a)}
            say(f"{v:>4}{a:>14.4f}{d:>14.4f}{d - a:>+16.4f}{rel:>+9.1f}%   {verd}")
        say("")
        say(f"⇒ 5 个类中 **{n_gt} 个** 满足 `rgbir > {ref_key}`。")
        return res

    cmp_vs_rgb = cmp_block("vs_rgb", "rgb",
                           "规则 A：`rgbir` vs `rgb`（**关键**：完整 RGB + 外加 IR，无丢 B 混淆）")
    cmp_vs_fuse = cmp_block("vs_fuse", "fuse",
                            "规则 B：`rgbir` vs `fuse`（都含 IR；差别是保留 B + 多一路通道）")
    cmp_vs_ctrl = cmp_block("vs_ctrl", "ctrl",
                            "规则 C：`rgbir` vs `ctrl`（对照为可见光灰度）")

    say("")
    say("=" * 100)
    say("## 总结")
    say("=" * 100)
    say("")
    n_rgb = sum(1 for k in cmp_vs_rgb if cmp_vs_rgb[k]["better"])
    n_fuse = sum(1 for k in cmp_vs_fuse if cmp_vs_fuse[k]["better"])
    n_ctrl = sum(1 for k in cmp_vs_ctrl if cmp_vs_ctrl[k]["better"])
    say("**与既有结论的对照**（`fuse`/`ctrl` 都丢掉 B 通道 ⇒ 那两个对比有混淆）：")
    say(f"  · `rgbir > rgb`      ：{n_rgb}/5 个类")
    say(f"  · `rgbir > fuse`     ：{n_fuse}/5 个类")
    say(f"  · `rgbir > ctrl`     ：{n_ctrl}/5 个类")
    say("")
    say("**能确定的**：")
    say("  · 本对比**首次**给出了「保留全部 RGB 的同时外加 IR」这一**无混淆**条件下的结果；")
    say("  · 判据在跑之前已声明（见本脚本 docstring），不做事后挑选。")
    say("")
    say("**仍然不能确定的（必须声明的边界）**：")
    say("  · **不能**证明「IR 能看见空鼓」：BFDD 掩膜由可见光判读定义，")
    say("    标注者很可能在 RGB 上判读 ⇒ 用任何模态预测它都受此约束；")
    say("  · 单次运行、单 seed ⇒ 类间小的 Δ 不构成结论（沿用项目「Δ 须超噪声」纪律）；")
    say("  · BFDD 仅 152 张 val、5 类，绝对 IoU 低是数据规模与难度的产物，")
    say("    **跨模态的相对比较**才是本实验的读数。")
    say("")

    rep = {
        "per_class_all_modalities": full,
        "rgbir_vs_rgb": cmp_vs_rgb,
        "rgbir_vs_fuse": cmp_vs_fuse,
        "rgbir_vs_ctrl": cmp_vs_ctrl,
        "counts": {"vs_rgb": n_rgb, "vs_fuse": n_fuse, "vs_ctrl": n_ctrl},
        "rgbir_meta": {k: rgbir.get(k) for k in
                       ("epochs", "epochs_landed", "imgsz", "batch", "seed",
                        "workers", "conf", "train_minutes", "init",
                        "channel_order", "inflate_info")},
        "note": ("rgbir = 真 4 通道 [R,G,B,IR]，首层由 3 通道预训练权重膨胀。"
                 "本对比回答「IR 通道是否携带独立信息」，"
                 "不能直接证明「IR 能看见空鼓」（标签由可见光判读定义）。"),
    }
    outp = EVAL / "bfdd_modality_compare_rgbir.json"
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    say(f"产物: {outp}")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[留痕] {OUT_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
