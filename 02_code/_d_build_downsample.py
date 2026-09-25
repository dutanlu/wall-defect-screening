# -*- coding: utf-8 -*-
"""
_d_build_downsample.py —— D 路【实验组 B】数据构建：rust 降采样

===== 为什么做这个实验（与实验组 A 的区别）=====

实验组 A（v11s640_clsbal）已做：**逆频率重加权**，只改**损失权重**、不改数据分布。
实测结论：7 类逐类 AP **全部在噪声内**（见 04_results/eval/clsbal_ab_comparison.json）
⇒ 「重加权能救长尾」这个朴素观点在本项目上**未被证据支持**。

实验组 B（本脚本）：**直接改训练数据分布** —— 对 rust 做降采样。
这是比加权**更强**的干预：加权只是让模型"更在意"稀有类，
而降采样是**真的减少**多数类的样本量，从而改变梯度来源的构成。

二者结合 = 「损失层改权重」vs「数据层改分布」的对照，比单做一个更有信息量。

===== 关键设计：为什么只降"纯 rust 图" =====

实测：训练集 1995 张图中，**纯 rust 图 836 张承载 9432/9433 个 rust 实例**，
含 rust 的混合图仅 1 张、1 个实例。
⇒ 只删纯 rust 图可以**精确控制 rust 实例数，且几乎零副作用**（不伤其他类）。
若按标注框级降采样，会把共存图里的其他类一起删掉 ⇒ 破坏实验。

===== 两档强度（比单点更能画趋势）=====

  B1 温和：rust 9433 → ~3000（保留 32%）⇒ 不平衡比 45.6:1 → 14.5:1
  B2 激进：rust 9433 → ~1000（保留 10.6%）⇒ 不平衡比 45.6:1 →  4.8:1

两档都以**固定随机种子**抽取，抽出的图片列表落盘（可复现）。

===== 铁律 =====

1. **绝不改动 01_data/dataset/**（V3 正式数据集，已冻结）——
   只把子集"复制"到 01_data/_d_downsample/<name>/，训练时用临时 yaml 指过去。
2. val 集**不动**：要让 B1/B2 与原基线在**同一套 val** 上比较。
3. test 集**不动**：最终评估仍用 V3 的 285 图独立测试集（口径不变）。

用法：
  python _d_build_downsample.py            # 两档都建
  python _d_build_downsample.py --check    # 只打印统计，不写文件
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # 创新题/外墙缺陷筛查
DS = ROOT / "01_data" / "dataset"
OUT = ROOT / "01_data" / "_d_downsample"
RUST = 4
SEED = 42

# (名称, 目标 rust 实例数)
CONFIGS = [
    ("B1_rust3000", 3000),
    ("B2_rust1000", 1000),
]


def parse_label(p: Path) -> Counter:
    c = Counter()
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            c[int(ln.split()[0])] += 1
    return c


def collect():
    """返回 (纯rust图列表[(img, lbl, n_rust)], 非纯rust图列表, val/test 统计)"""
    lab_tr = DS / "labels" / "train"
    img_tr = DS / "images" / "train"
    pure, other = [], []
    for lbl in sorted(lab_tr.glob("*.txt")):
        img = img_tr / (lbl.stem + ".jpg")
        if not img.exists():
            for ext in (".png", ".jpeg", ".JPG", ".PNG"):
                cand = img_tr / (lbl.stem + ext)
                if cand.exists():
                    img = cand
                    break
        c = parse_label(lbl)
        if set(c.keys()) == {RUST}:
            pure.append((img, lbl, c[RUST]))
        else:
            other.append((img, lbl))
    return pure, other


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只统计不落盘")
    args = ap.parse_args()

    pure, other = collect()
    n_pure_inst = sum(n for _, _, n in pure)
    print("=" * 72)
    print("训练集构成")
    print("=" * 72)
    print("  纯 rust 图 : %4d 张，承载 rust 实例 %d" % (len(pure), n_pure_inst))
    print("  其它图     : %4d 张（含 rust 混合图 %d 张）"
          % (len(other), len(other) - (1995 - len(pure) - len(other)) if False else 0))
    print("  ⇒ 总图 %d 张" % (len(pure) + len(other)))

    if args.check:
        for name, target in CONFIGS:
            keep_frac = target / n_pure_inst
            n_keep = round(len(pure) * keep_frac)
            print("  [%s] 目标 rust=%d ⇒ 保留纯rust图 %d/%d (%.1f%%)，总图 %d"
                  % (name, target, n_keep, len(pure), 100 * keep_frac,
                     n_keep + len(other)))
        print("\n(--check 模式，未写任何文件)")
        return

    for name, target in CONFIGS:
        dst = OUT / name
        if dst.exists():
            shutil.rmtree(dst)
        (dst / "images" / "train").mkdir(parents=True)
        (dst / "labels" / "train").mkdir(parents=True)

        keep_frac = target / n_pure_inst
        n_keep = round(len(pure) * keep_frac)
        rng = random.Random(SEED)
        kept = rng.sample(pure, n_keep)
        kept.sort(key=lambda t: t[0].name)

        # 纯 rust 子集 + 全部其它图
        for img, lbl, _ in kept:
            shutil.copy2(img, dst / "images" / "train" / img.name)
            shutil.copy2(lbl, dst / "labels" / "train" / lbl.name)
        for img, lbl in other:
            shutil.copy2(img, dst / "images" / "train" / img.name)
            shutil.copy2(lbl, dst / "labels" / "train" / lbl.name)

        # val / test 原样（软链不可靠，直接指向同一目录由 yaml 完成）
        c = Counter()
        for lbl in (dst / "labels" / "train").glob("*.txt"):
            c.update(parse_label(lbl))
        names = {0: "crack", 1: "spalling", 2: "efflorescence", 3: "exposed_rebar",
                 4: "rust", 5: "delamination", 6: "moss"}
        print("\n[%s] 写出 → %s" % (name, dst))
        print("  图 %d 张 / 实例 %d" % (len(list((dst / "images" / "train").glob("*"))),
                                          sum(c.values())))
        for k in sorted(c, key=lambda x: -c[x]):
            print("      %-14s %5d" % (names[k], c[k]))
        insts = sorted(c.values())
        print("  不平衡比 max/min = %.1f : 1" % (insts[-1] / insts[0]))

        # 写出临时 yaml（train 指向子集；val/test 仍指 V3 正式集）
        yaml_txt = (
            "# 自动生成（_d_build_downsample.py）—— D 路实验组 B 专用，勿手工编辑\n"
            f"# 配置: {name}\n"
            "path: %s\n"
            "train: images/train\n"
            "val: %s\n"
            "test: %s\n"
            "nc: 7\n"
            "names:\n"
            "  0: crack\n  1: spalling\n  2: efflorescence\n  3: exposed_rebar\n"
            "  4: rust\n  5: delamination\n  6: moss\n"
        ) % (dst.as_posix(),
             (DS / "images" / "val").as_posix(),
             (DS / "images" / "test").as_posix())
        (dst / f"{name}.yaml").write_text(yaml_txt, encoding="utf-8")
        print("  yaml: %s" % (dst / f"{name}.yaml"))

    print("\n完成。注意：val/test 未复制，仍由 V3 正式集提供（保证同口径）。")


if __name__ == "__main__":
    main()
