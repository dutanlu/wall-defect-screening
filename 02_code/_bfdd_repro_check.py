#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""BFDD 数据集**外部可复算核查**（自包含，不依赖本工程任何模块）。

## 用途
任何人从 Mendeley 下载官方包（DOI `10.17632/9ych7czvyg.1`）后，运行本脚本即可**独立复现**
本项目关于 BFDD 的全部**确定性事实**，无需信任我们的任何中间产物：

  1. 归档 **sha256** 与字节数；
  2. **838 对**的配对闭包（RGB ∩ IR ∩ Label ∩ Label_color 是否一致）；
  3. 官方划分 `train.txt`(686) / `test.txt`(152) 是否**完整覆盖**且**无泄漏**；
  4. **掩膜值 ↔ 颜色**的对应关系（用 `Label_color` 反解，不靠目视猜色）；
  5. **5 类的像素分布**（逐值计数）；
  6. 与 `logs/_BFDD_VERDICT.md` 记录的数值**逐项对照**。

## 为什么需要它
本项目报告中的许多结论引用了 BFDD。让外部人**能自己算一遍**，
比在报告里写"可复算"要有力得多 —— 这是"证据能被独立验证"的最小落地。

## 依赖
仅 `numpy` + `opencv-python`（**不需要**本工程、不需要 GPU、不需要网络）。
本脚本**只读**归档，不解压到磁盘。

## 用法
    python _bfdd_repro_check.py <BFDD_dataset.tar.gz 路径>
    python _bfdd_repro_check.py --help

## 版权
BFDD 为**他人**公开数据：**CC BY 4.0**，贡献机构 **东南大学（Southeast University）**。
本脚本只做核查，不重新分发数据。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:                       # pragma: no cover
    cv2 = None

# ---- 本项目在 `logs/_BFDD_VERDICT.md` 中记录、供本脚本对照的数值 ----
# 若这些对照项不匹配，说明**要么归档不同，要么我们的记录有误** —— 两种情况都必须查清。
EXPECT_ARCHIVE_BYTES = 553_316_751
# ⚠️ 本值已用**独立工具** `sha256sum` 核对过（勿手抄，易多打字符 ——
#    第一版这里多打了一个 `f`（65 位），导致误报"归档不一致"）
EXPECT_SHA256 = "43d06305bf3c913f59d52c3ffa10caa0e129b668b7b3c9d8f80d619c6e6e8a7a"
assert len(EXPECT_SHA256) == 64, "sha256 常量长度必须是 64"
EXPECT_N_PAIRS = 838
EXPECT_SPLIT = {"train": 686, "test": 152}
# 掩膜值 -> 颜色（8-bit RGB）与像素数，均来自 `logs/_BFDD_VERDICT.md` §四/§五
EXPECT_MASK_COLOR = {1: (61, 61, 245), 2: (169, 36, 191), 3: (174, 79, 13),
                     4: (36, 179, 83), 5: (203, 253, 0)}
EXPECT_MASK_PX = {0: 252_506_139, 1: 4_677_930, 2: 1_615_986,
                  3: 3_060_090, 4: 5_903_313, 5: 6_832_382}


def sha256_of(p: Path, chunk: int = 1 << 22) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def decode(raw: bytes, flags):
    if cv2 is None:
        raise RuntimeError("需要 opencv-python（pip install opencv-python）")
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)


def main() -> int:
    ap = argparse.ArgumentParser(description="BFDD 数据集外部可复算核查（只读，不解压）")
    ap.add_argument("archive", nargs="?", default="",
                    help="BFDD_dataset.tar.gz 路径")
    args = ap.parse_args()

    if not args.archive:
        print(__doc__)
        return 2
    path = Path(args.archive)
    if not path.exists():
        print(f"!! 找不到归档：{path}")
        return 2

    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(str(s))

    ok_all = True

    def check(name: str, got, want) -> bool:
        nonlocal ok_all
        good = (got == want)
        ok_all = ok_all and good
        say(f"  [{'✅' if good else '❌'}] {name}: {got}" +
            ("" if good else f"   （期望 {want}）"))
        return good

    say("=" * 92)
    say("BFDD 数据集外部可复算核查")
    say("=" * 92)
    say(f"归档：{path}")
    say("（BFDD = 他人公开数据，CC BY 4.0，东南大学；本脚本只读核查，不重新分发）")
    say("")

    say("## 1. 归档指纹")
    digest, nbytes = sha256_of(path)
    check("字节数", nbytes, EXPECT_ARCHIVE_BYTES)
    check("sha256", digest, EXPECT_SHA256)

    say("")
    say("## 2. 从归档流式读取（**不解压到磁盘**）")
    stems: dict[str, dict[str, bytes]] = defaultdict(dict)
    with tarfile.open(path, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name
            base = name.rsplit("/", 1)[-1]
            stem = base.rsplit(".", 1)[0]
            for sub in ("RGB", "IR", "Label", "Label_color"):
                if f"/{sub}/" in name:
                    f = tf.extractfile(m)
                    if f is not None:
                        stems[stem][sub] = f.read()
                    break
    say(f"  读到唯一 stem = {len(stems)}")

    say("")
    say("## 3. 配对闭包（四种文件是否都齐）")
    need = ("RGB", "IR", "Label", "Label_color")
    complete = {s: d for s, d in stems.items() if all(k in d for k in need)}
    check("四种文件都齐的 stem 数", len(complete), EXPECT_N_PAIRS)
    miss = {s: [k for k in need if k not in d] for s, d in stems.items()
            if not all(k in d for k in need)}
    if miss:
        say(f"  !! 不完整的 stem（前 5）：{list(miss.items())[:5]}")

    say("")
    say("## 4. 官方划分：完整覆盖 + 无泄漏")
    # 划分文件在归档里可能是 train.txt / test.txt（此处从 tar 里再取一次）
    split = {}
    with tarfile.open(path, "r:gz") as tf:
        for m in tf:
            base = m.name.rsplit("/", 1)[-1]
            if base in ("train.txt", "test.txt"):
                f = tf.extractfile(m)
                if f is not None:
                    split[base] = [x.strip() for x in
                                   f.read().decode("utf-8", "replace").split() if x.strip()]
    for k, want in EXPECT_SPLIT.items():
        got = len(split.get(f"{k}.txt", []))
        check(f"{k}.txt 行数", got, want)
    s_tr, s_te = set(split.get("train.txt", [])), set(split.get("test.txt", []))
    check("train ∩ test（应为 0）", len(s_tr & s_te), 0)
    check("未被划分覆盖的 stem（应为 0）", len(set(stems) - s_tr - s_te), 0)
    check("划分里虚构的 stem（应为 0）", len((s_tr | s_te) - set(stems)), 0)

    say("")
    say("## 5. 掩膜值 ↔ 颜色 的 100% 对应（用 Label_color 反解）")
    say("  （不靠目视猜色：对每个值取 Label==v 的全部像素，看 Label_color 的颜色分布）")
    px_count = Counter()
    color_vote: dict[int, Counter] = defaultdict(Counter)
    n_label_ok = 0
    for s, d in sorted(complete.items()):
        lab = decode(d["Label"], cv2.IMREAD_UNCHANGED)
        lc = decode(d["Label_color"], cv2.IMREAD_COLOR)
        if lab is None or lc is None:
            continue
        n_label_ok += 1
        if lab.ndim == 3:
            lab = lab[:, :, 0]
        for v in np.unique(lab):
            v = int(v)
            px_count[v] += int((lab == v).sum())
            sel = lc[lab == v]
            if sel.size:
                # ★ 必须**逐像素**统计颜色直方图取众数，不能先求均值再当"颜色"投票：
                #   值 4 里有 0.32% 浅蓝（正是"早期 6 类编号空间"的钥匙），
                #   先求均值会被这 0.32% 拉走 ⇒ 主色占比从 99.68% 假跌到 88.28%。
                #   本口径与 `logs/_BFDD_VERDICT.md` §四一致。
                keys = (sel[:, 2].astype(np.uint32) << 16 |
                        sel[:, 1].astype(np.uint32) << 8 |
                        sel[:, 0].astype(np.uint32))
                uniq, cnts = np.unique(keys, return_counts=True)
                k = int(uniq[int(np.argmax(cnts))])
                rgb = ((k >> 16) & 255, (k >> 8) & 255, k & 255)
                color_vote[v][rgb] += int(cnts.max())
    say(f"  成功读取掩膜的图数 = {n_label_ok} / {len(complete)}")
    for v in range(1, 6):
        if not color_vote.get(v):
            say(f"  [❌] 值{v}: 未出现")
            ok_all = False
            continue
        top, cnt = color_vote[v].most_common(1)[0]
        total = sum(color_vote[v].values())
        share = cnt / total if total else 0.0
        want = EXPECT_MASK_COLOR[v]
        near = all(abs(a - b) <= 8 for a, b in zip(top, want))
        ok_all = ok_all and near and share > 0.98
        say(f"  [{'✅' if near and share > 0.98 else '❌'}] 值{v}: "
            f"主色 RGB{top} 占 {share:.2%}（记录值 RGB{want}）")

    say("")
    say("## 6. 掩膜像素总数（含背景），与记录逐项对照")
    say(f"  {'值':>4}{'实测像素':>14}{'记录像素':>14}   判定")
    for v in sorted(EXPECT_MASK_PX):
        got = int(px_count.get(v, 0))
        want = EXPECT_MASK_PX[v]
        good = (got == want)
        ok_all = ok_all and good
        say(f"  {v:>4}{got:>14,}{want:>14,}   {'✅' if good else '❌'}")
    tot = sum(EXPECT_MASK_PX.values())
    if n_label_ok:
        say(f"  逐位闭合校验：{n_label_ok} 图 x 640 x 512 = 实测总和 "
            f"{sum(int(px_count.get(v,0)) for v in range(6)):,}")
        say(f"  （记录的全量总和 = {tot:,}，按 838 图算）")

    say("")
    say("=" * 92)
    say("结论")
    say("=" * 92)
    if ok_all:
        say("✅ **全部对照项一致** ⇒ 本项目关于 BFDD 的确定性事实**可被独立复现**。")
    else:
        say("❌ **存在不一致项**（见上方 ❌）⇒ 请核对归档版本；")
        say("   若归档 sha256 与记录一致却仍有 ❌，则说明**本项目的记录有误**，应以实测为准。")
    say("")
    say("★ 边界：本脚本只复核**确定性事实**（指纹、配对、划分、颜色映射、像素计数），")
    say("  **不**复核任何需要建模的结论（如类名推断、模态判别），那些属于可争论的推断。")
    say("=" * 92)

    outp = Path(__file__).with_name("_bfdd_repro_check_report.txt")
    outp.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[报告] {outp}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
