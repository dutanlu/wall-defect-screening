# -*- coding: utf-8 -*-
"""
step2_dedup.py —— 去重（本项目的「防泄漏闸门」，必做）

背景（血泪教训）：
  基础题数据集 733 个文件里只有 549 张唯一图 —— Roboflow 导出时把
  rotated_by_* / vertical_flip_* / translation_* 等增强变体与原图并排存为独立文件。
  旧流程按「文件数」划分 train/val，导致同一张图的副本被劈到两侧，
  val 泄漏率 88.5%，mAP@0.5 被抬到 0.971（虚高）。
  修复后真实指标为 P 0.916 / R 0.799 / mAP50 0.854 / mAP50-95 0.676。

所以本项目在「下载之后、划分之前」强制插入本步骤。

三级去重：
  L1 文件 MD5      —— 完全相同的文件（字节级）
  L2 感知哈希 dHash —— 同图被 resize / 重压缩（Roboflow 常见）
  L3 报告不合并     —— 近似但非同一张（不同拍摄）只统计不删，避免误杀真实样本

策略：每组保留「信息量最大」的一张 —— 优先保留短边最大、再比文件最大。
      其余移动（不是删除）到 01_去重后/_被移除_重复/ 下，可随时复查。

用法：
  python step2_dedup.py
  python step2_dedup.py --root=D:/... --out=D:/... --near=3

输出：
  04_结果/评估/dedup_report.json
  logs/step2_dedup.log（由 PowerShell 重定向）
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

from common import (
    DEDUP_DIR,
    RAW_DIR,
    EVAL_DIR,
    content_hash,
    dump_json,
    ensure_dirs,
    hamming,
    imread_u,
    list_images,
    log,
    md5_of_file,
    argv_flag,
)


def main() -> None:
    root = Path(argv_flag("root", str(RAW_DIR)))
    out_root = Path(argv_flag("out", str(DEDUP_DIR)))
    near_thr = int(argv_flag("near", "3"))          # 汉明距离 <= 阈值视为近似同图
    # 是否把被判定为重复的副本从源目录「移走」。
    # 默认 False（复制）—— 因为默认移动会让源目录残缺且不可预期：
    # 保留项走 copy、隔离项走 move，跑完源目录只剩「凑巧被复制过的那批」，
    # 用户再想重跑就得重新解压。已实测踩到（2400 文件跑完 src 只剩 677）。
    move_dup = argv_flag("move") is not None
    keep_root = out_root / "images"
    drop_root = out_root / "_removed_duplicates"

    ensure_dirs(keep_root, drop_root, EVAL_DIR)

    log(f"扫描: {root}")
    log(f"隔离方式: {'移动（源目录会被删减）' if move_dup else '复制（源目录保持不动）'}")
    images = list_images(root)
    log(f"发现图像 {len(images)} 张（按文件计）")
    if not images:
        log("!! 没有图像，先跑 step0 下载")
        return

    # ---------- L1: 文件 MD5 ----------
    md5_buckets: dict[str, list[Path]] = defaultdict(list)
    for i, p in enumerate(images, 1):
        try:
            md5_buckets[md5_of_file(p)].append(p)
        except Exception as e:
            log(f"  读取失败 {p}: {e}")
        if i % 500 == 0:
            log(f"  MD5 计算 {i}/{len(images)}")

    exact_dups = {k: v for k, v in md5_buckets.items() if len(v) > 1}
    n_exact_removed = sum(len(v) - 1 for v in exact_dups.values())
    log(f"L1 完全重复：{len(exact_dups)} 组，冗余 {n_exact_removed} 个文件")

    # 每组保留一张（此时还不能定 keep，先做 L2 再决定，但为了算 dHash 只处理代表）
    # 优化：只对每组代表计算 dHash（重复文件内容相同，无需重复计算）
    representatives: list[Path] = [sorted(v)[0] for v in md5_buckets.values()]
    log(f"L1 后剩余唯一内容 {len(representatives)} 张")

    # ---------- L2: 感知哈希 ----------
    hashes: list[tuple[str, Path]] = []
    for i, p in enumerate(representatives, 1):
        img = imread_u(p)
        if img is None:
            continue
        h = content_hash(img)
        if h:
            hashes.append((h, p))
        if i % 500 == 0:
            log(f"  dHash 计算 {i}/{len(representatives)}")

    # 分桶加速：dHash 前 16 位做粗桶
    near_groups: list[list[tuple[str, Path]]] = []
    used = [False] * len(hashes)
    for i in range(len(hashes)):
        if used[i]:
            continue
        gi = [hashes[i]]
        used[i] = True
        for j in range(i + 1, len(hashes)):
            if used[j]:
                continue
            if hamming(hashes[i][0], hashes[j][0]) <= near_thr:
                gi.append(hashes[j])
                used[j] = True
        if len(gi) > 1:
            near_groups.append(gi)

    n_near = sum(len(g) - 1 for g in near_groups)
    log(f"L2 近似重复：{len(near_groups)} 组（阈值 hamming<={near_thr}），冗余 {n_near} 张")
    log(f"L2 后唯一图 {len(hashes) - n_near} 张")

    # ---------- 决定保留谁 ----------
    def quality(p: Path) -> tuple:
        img = imread_u(p)
        if img is None:
            return (0, 0, 0)
        h, w = img.shape[:2]
        return (min(h, w), max(h, w), p.stat().st_size)

    keep: list[Path] = []
    drop: list[tuple[Path, Path, str]] = []   # (被删, 保留对象, 原因)

    # 完全重复组
    for h, group in md5_buckets.items():
        group = sorted(group)
        best = max(group, key=quality)
        keep.append(best)
        for p in group:
            if p != best:
                drop.append((p, best, "L1_exact_md5"))

    # 近似组：从每组代表里再挑一个（注意代表可能已被 L1 选为 best 或未选）
    kept_set = set(keep)
    for g in near_groups:
        members = [p for _, p in g]
        # 仅处理仍然「存活」的成员（L1 未淘汰的）
        alive = [p for p in members if p in kept_set]
        if len(alive) <= 1:
            continue
        best = max(alive, key=quality)
        for p in alive:
            if p != best:
                kept_set.discard(p)
                drop.append((p, best, "L2_near_dhash"))

    keep = sorted(kept_set)
    log(f"最终保留 {len(keep)} 张唯一图；将{'移动' if move_dup else '复制'} "
        f"{len(drop)} 个冗余副本到 {drop_root}")

    # ---------- 落盘（图像与标签必须成对处理） ----------
    # 关键：样本 = (图像, 标签)。只处理图像会让后续 step3/step4 找不到标签，
    # 直接把数据集搬空。所以这里图像与标签总是成对处理。
    # 落盘策略由 move_dup 统一控制，避免「保留项复制 + 隔离项移动」的不对称行为。
    n_copied = 0
    common_root = _common_parent([p for p in keep] + [d[0] for d in drop]) or root
    n_label_copied = 0
    n_label_missing = []

    for p in keep:
        rel = _safe_rel(p, common_root)
        dst = keep_root / rel
        ensure_dirs(dst.parent)
        shutil.copy2(p, dst)
        n_copied += 1
        lab_src = _find_label(p, root)
        if lab_src is not None:
            lab_dst = keep_root / _label_rel(rel)
            ensure_dirs(lab_dst.parent)
            shutil.copy2(lab_src, lab_dst)
            n_label_copied += 1
        else:
            n_label_missing.append(str(p))

    for src, kept, reason in drop:
        rel = _safe_rel(src, common_root)
        dst = drop_root / rel
        ensure_dirs(dst.parent)
        if src.exists():
            try:
                if move_dup:
                    shutil.move(str(src), str(dst))
                else:
                    shutil.copy2(src, dst)
            except Exception:
                shutil.copy2(src, dst)
        lab_src = _find_label(src, root)
        if lab_src is not None and lab_src.exists():
            lab_dst = drop_root / _label_rel(rel)
            ensure_dirs(lab_dst.parent)
            try:
                if move_dup:
                    shutil.move(str(lab_src), str(lab_dst))
                else:
                    shutil.copy2(lab_src, lab_dst)
            except Exception:
                shutil.copy2(lab_src, lab_dst)

    log(f"成对搬运：图像 {n_copied} 张，标签 {n_label_copied} 个")
    if n_label_missing:
        log(f"!! 有 {len(n_label_missing)} 张保留图像找不到标签（前 10 个）：")
        for x in n_label_missing[:10]:
            log(f"     {x}")
        log("   -> 这些图在 step4 会被判为「无标签图像」而剔除，请确认是否符合预期")

    # 数据集级元数据也要跟着走：classes.txt / *.yaml 决定类别名，
    # 丢了它们下游只能看到 id0/id1，类别映射会全部失效（已实测踩到）。
    n_meta = 0
    for meta_name in ("classes.txt",):
        for m in root.rglob(meta_name):
            rel = _safe_rel(m, common_root)
            dst = keep_root / rel
            ensure_dirs(dst.parent)
            shutil.copy2(m, dst)
            n_meta += 1
    for pattern in ("*.yaml", "*.yml"):
        for m in root.rglob(pattern):
            if _is_skipped_rel(m, root):
                continue
            rel = _safe_rel(m, common_root)
            dst = keep_root / rel
            ensure_dirs(dst.parent)
            shutil.copy2(m, dst)
            n_meta += 1
    if n_meta:
        log(f"元数据文件已随行复制：{n_meta} 个（classes.txt / yaml）")
    else:
        log("!! 未找到 classes.txt 或 yaml，下游将无法解析类别名")
        log("   -> 若数据源确实没有，请在 keep_root 下手写一份 classes.txt：每行一个类名，顺序即 id")

    report = {
        "root": str(root),
        "n_files_scanned": len(images),
        "L1_exact_groups": len(exact_dups),
        "L1_removed": n_exact_removed,
        "L2_near_groups": len(near_groups),
        "L2_removed": n_near,
        "n_unique_kept": len(keep),
        "n_labels_copied": n_label_copied,
        "n_keep_images_without_label": len(n_label_missing),
        "keep_images_without_label": n_label_missing[:50],
        "n_moved_to_quarantine": len(drop),
        "dedup_ratio": round(1 - len(keep) / len(images), 4) if images else None,
        "keep_root": str(keep_root),
        "quarantine_root": str(drop_root),
        "L1_group_detail": [
            {"md5": k, "files": [str(x) for x in v]} for k, v in list(exact_dups.items())[:50]
        ],
        "L2_group_detail": [
            {"members": [str(p) for _, p in g]} for g in near_groups[:50]
        ],
        "removed_detail": [
            {"removed": str(a), "kept": str(b), "reason": r} for a, b, r in drop[:500]
        ],
    }
    dump_json(report, EVAL_DIR / "dedup_report.json")

    log("=" * 60)
    log(f"去重完成：{len(images)} 文件 → {len(keep)} 张唯一图")
    log(f"文件级冗余率：{report['dedup_ratio']:.1%}")
    log(f"保留：{keep_root}")
    log(f"隔离：{drop_root}")
    log("=" * 60)
    log("提示：请核对 L2 抽样的组，确认没有把「不同拍摄的真实样本」误判为重复。")
    log("      dHash 对小图较敏感，若误杀率高，可把 --near 调小到 1~2 重跑。")


def _safe_rel(p: Path, base: Path) -> Path:
    """求相对路径；若不在 base 下则退化为「父目录名/文件名」，避免路径塌陷丢文件。"""
    try:
        return p.relative_to(base)
    except ValueError:
        return Path(p.parent.name) / p.name


def _label_rel(img_rel: Path) -> Path:
    """把图像相对路径映射成标签相对路径：images -> labels，后缀改 .txt。"""
    parts = ["labels" if x == "images" else x for x in img_rel.parts]
    return Path(*parts).with_suffix(".txt")


def _find_label(img_path: Path, src_root: Path) -> Path | None:
    """定位图像对应的 YOLO 标签文件。"""
    c1 = img_path.with_suffix(".txt")
    if c1.exists():
        return c1
    s = str(img_path)
    c2 = Path(s.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
    if c2.exists():
        return c2
    hits = list(src_root.rglob(img_path.stem + ".txt"))
    return hits[0] if hits else None


def _is_skipped_rel(p: Path, base: Path) -> bool:
    """相对路径里是否含隔离区/隐藏目录段。"""
    try:
        rel = p.relative_to(base)
    except ValueError:
        return False
    from common import is_skipped
    return is_skipped(rel)


def _common_parent(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    parts = [list(p.parts) for p in paths]
    common: list[str] = []
    for tup in zip(*parts):
        if len(set(tup)) == 1:
            common.append(tup[0])
        else:
            break
    return Path(*common) if common else None


if __name__ == "__main__":
    main()
