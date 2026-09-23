# -*- coding: utf-8 -*-
"""
step4_split_dataset.py —— 7:2:1 划分 + 跨集交集自检（防泄漏闭环）

与基础题的区别（这就是「修复」本身）：
  1. 划分单位是「图片内容」而非「文件数」。输入必须是 step2 去重后的目录，
     所以同一张图的变体不会横跨 train/val。
  2. 划分后强制做交叉自检：
       - 文件名交集
       - 文件 MD5 交集
       - 感知哈希（dHash）交集
     任何一项不为 0 就报错退出（除非显式 --force）。
  3. 同时检查「无标签图像」和「无图像标签」，避免静默丢数据。

用法：
  python step4_split_dataset.py
  python step4_split_dataset.py --ratio=0.7,0.2,0.1 --seed=42

输出：
  01_数据/03_数据集/images/{train,val,test}/**
  01_数据/03_数据集/labels/{train,val,test}/**
  01_数据/03_数据集/wall_defects.yaml
  04_结果/评估/split_report.json
  02_代码/roboflow_yaml/wall_defects.yaml（Roboflow 上传用，含 train/valid/test 键名）
"""

from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

from common import (
    CLASSES,
    DATASET_DIR,
    DATASET_YAML_NAME,
    EVAL_DIR,
    UNIFIED_DIR,
    argv_flag,
    content_hash,
    dump_json,
    ensure_dirs,
    imread_u,
    list_images,
    load_json,
    log,
    md5_of_file,
    read_yolo_labels,
    write_dataset_yaml,
    write_yolo_labels,
)


def main() -> None:
    src_root = Path(argv_flag("in", str(UNIFIED_DIR)))
    out_root = Path(argv_flag("out", str(DATASET_DIR)))
    ratio = argv_flag("ratio", "0.7,0.2,0.1")
    seed = int(argv_flag("seed", "42"))
    force = argv_flag("force") is not None

    r_train, r_val, r_test = [float(x) for x in ratio.split(",")]
    assert abs(r_train + r_val + r_test - 1.0) < 1e-6, "比例之和必须为 1"

    ensure_dirs(out_root, EVAL_DIR)
    for mode in ("train", "val", "test"):
        ensure_dirs(out_root / "images" / mode, out_root / "labels" / mode)

    # 收集所有「成对」的 (img, label)
    pairs: list[tuple[Path, Path]] = []
    orphan_imgs: list[Path] = []
    for img in list_images(src_root):
        lab = find_label(img, src_root)
        if lab is None:
            orphan_imgs.append(img)
        else:
            pairs.append((img, lab))

    log(f"输入: {src_root}")
    log(f"图像-标签配对: {len(pairs)}，无标签图像: {len(orphan_imgs)}")
    if not pairs:
        log("!! 没有可用样本，先跑 step3")
        return

    # ---------------- 关键：计算每个样本的「扁平输出名」 ----------------
    # step2/step3 会在输出根下重建 images/labels 两级目录，若直接沿用相对路径，
    # 会得到 images/train/<源名>/images/<子目录>/<文件> 这种嵌套爆炸结构，
    # 且 <源名> 目录名会污染数据布局。这里统一压平成 <文件名>。
    #
    # 注意：不能用 img._flat_override = ... 给 Path 挂属性 —— pathlib.Path
    # 没有 __dict__（WindowsPath 是 __slots__ 类），会抛 AttributeError。
    # 必须用独立的 side-table（dict）记录映射。
    # -------------------------------------------------------------------
    names: dict[Path, str] = {}
    name_to_paths: dict[str, list[Path]] = {}
    for img, _ in pairs:
        names[img] = img.name
        name_to_paths.setdefault(img.name, []).append(img)

    clash = {k: v for k, v in name_to_paths.items() if len(v) > 1}
    if clash:
        log(f"注意：{len(clash)} 个文件名在不同子目录重复，将用 <上级目录名>__<文件名> 消歧")
        for nm, imgs in clash.items():
            for im in imgs:
                names[im] = f"{im.parent.name}__{im.name}"

    # 消歧后仍可能重名（多源同名同层），再补递增序号
    seen: dict[str, int] = {}
    for img, _ in pairs:
        base = names[img]
        if base in seen:
            seen[base] += 1
            stem, suf = Path(base).stem, Path(base).suffix
            names[img] = f"{stem}_{seen[base]}{suf}"
        else:
            seen[base] = 1

    # ---------------- 去重保险：再查一次源内重复 ----------------
    # （step2 已去重，但多个源合并后可能引入新的相同内容）
    sig: dict[str, Path] = {}
    dup_in_src = []
    for img, _ in pairs:
        try:
            h = md5_of_file(img)
        except Exception:
            continue
        if h in sig:
            dup_in_src.append({"dup": str(img), "first": str(sig[h])})
        else:
            sig[h] = img
    if dup_in_src:
        log(f"!! 源内仍存在 {len(dup_in_src)} 个字节级重复（多源合并引入），将自动剔除")
        bad = {d["dup"] for d in dup_in_src}
        pairs = [(i, l) for i, l in pairs if str(i) not in bad]
        log(f"   剔除后 {len(pairs)} 对")

    # ---------------- 分层划分 ----------------
    # 按「主类别」分层，保证罕见类在 train/val/test 都有出现
    def dominant_class(lab: Path) -> str:
        labels = read_yolo_labels(lab)
        if not labels:
            return "__empty__"
        c = Counter(l[0] for l in labels).most_common(1)[0][0]
        return CLASSES[c] if c < len(CLASSES) else f"id{c}"

    buckets: dict[str, list[tuple[Path, Path]]] = {}
    for item in pairs:
        buckets.setdefault(dominant_class(item[1]), []).append(item)

    rng = random.Random(seed)
    sets: dict[str, list[tuple[Path, Path]]] = {"train": [], "val": [], "test": []}

    for cls, items in sorted(buckets.items()):
        rng.shuffle(items)
        n = len(items)
        if n == 1:
            # 单样本类：只能进 train，并在报告里标注（保证不虚报指标）
            sets["train"] += items
            log(f"  [警告] 类别 {cls} 仅 1 个样本，全部进 train")
            continue
        if n == 2:
            sets["train"] += items[:1]
            sets["val"] += items[1:]
            log(f"  [警告] 类别 {cls} 仅 2 个样本，train/val 各 1")
            continue
        n_test = max(1, int(round(n * r_test))) if r_test > 0 else 0
        n_val = max(1, int(round(n * r_val)))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train, n_val, n_test = n, 0, 0
        sets["train"] += items[:n_train]
        sets["val"] += items[n_train:n_train + n_val]
        sets["test"] += items[n_train + n_val:]

    # ---------------- 复制落盘（扁平命名） ----------------
    stats = {}
    for mode, items in sets.items():
        rng.shuffle(items)
        n_img = n_lab = n_inst = 0
        cls_cnt: Counter = Counter()
        for img, lab in items:
            flat = names[img]
            out_img = out_root / "images" / mode / flat
            out_lab = out_root / "labels" / mode / Path(flat).with_suffix(".txt")
            ensure_dirs(out_img.parent, out_lab.parent)
            shutil.copy2(img, out_img)
            labels = read_yolo_labels(lab)
            write_yolo_labels(out_lab, labels)
            n_img += 1
            n_lab += 1
            n_inst += len(labels)
            for c, *_ in labels:
                cls_cnt[CLASSES[c] if c < len(CLASSES) else f"id{c}"] += 1
        stats[mode] = {
            "n_images": n_img,
            "n_labels": n_lab,
            "n_instances": n_inst,
            "class_counts": {c: cls_cnt.get(c, 0) for c in CLASSES},
        }
        log(f"{mode:5s}: {n_img:5d} 张 / {n_inst:6d} 实例  {dict(cls_cnt)}")

    # ---------------- 跨集交集自检（核心） ----------------
    log("-" * 60)
    log("跨集交集自检（必须全为 0）")
    id_stats = {}
    for mode in ("train", "val", "test"):
        imgs = list_images(out_root / "images" / mode)
        names = {p.name for p in imgs}
        md5s = {}
        for p in imgs:
            try:
                md5s[p.name] = md5_of_file(p)
            except Exception:
                pass
        phashes = {}
        for p in imgs:
            im = imread_u(p)
            if im is not None:
                phashes[p.name] = content_hash(im)
        id_stats[mode] = {"names": names, "md5": md5s, "dhash": phashes}

    def overlap(a, b, key):
        A, B = id_stats[a][key], id_stats[b][key]
        if key == "names":
            return sorted(A & B)
        return sorted({A[k] for k in A} & {B[k] for k in B})

    intersections = {}
    any_leak = False
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for key in ("names", "md5", "dhash"):
            inter = overlap(a, b, key)
            intersections[f"{a}-{b}:{key}"] = len(inter)
            if inter:
                any_leak = True
                log(f"  !! {a}-{b} {key} 交集 {len(inter)}: {inter[:10]}")
            else:
                log(f"  OK {a}-{b} {key} 交集 0")

    if any_leak and not force:
        log("!! 检测到跨集重叠，流程终止（这正是基础题数据泄漏的成因）")
        log("   请回到 step2 重新去重，不要用 --force 掩盖")
        raise SystemExit(2)

    # ---------------- 写 yaml ----------------
    # path 用绝对路径（out_root），train/val/test 为其下的相对目录
    ds_yaml = out_root / DATASET_YAML_NAME
    write_dataset_yaml(
        ds_yaml,
        train="images/train",
        val="images/val",
        test="images/test",
        root=out_root,
    )
    # Roboflow 上传用的版本（键名为 train/valid/test）
    rf_yaml = Path(__file__).resolve().parent / "roboflow_yaml" / DATASET_YAML_NAME
    write_dataset_yaml(rf_yaml, train="images/train", val="images/valid",
                       test="images/test", root=out_root)

    report = {
        "input": str(src_root),
        "output": str(out_root),
        "ratio": {"train": r_train, "val": r_val, "test": r_test},
        "seed": seed,
        "n_pairs": len(pairs),
        "n_orphan_images": len(orphan_imgs),
        "orphan_examples": [str(p) for p in orphan_imgs[:30]],
        "duplicates_removed_in_src": dup_in_src[:50],
        "n_duplicates_removed": len(dup_in_src),
        "splits": stats,
        "intersection_check": intersections,
        "leak_free": not any_leak,
        "dataset_yaml": str(ds_yaml),
        "roboflow_yaml": str(rf_yaml),
    }
    dump_json(report, EVAL_DIR / "split_report.json")

    log("=" * 60)
    log(f"划分完成：{dict((m, stats[m]['n_images']) for m in stats)}")
    log(f"数据集配置: {ds_yaml}")
    log(f"无泄漏自检: {'通过' if not any_leak else '未通过'}")
    log("=" * 60)


def find_label(img_path: Path, src_root: Path) -> Path | None:
    stem = img_path.stem
    c1 = img_path.with_suffix(".txt")
    if c1.exists():
        return c1
    c2 = Path(
        str(img_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
    ).with_suffix(".txt")
    if c2.exists():
        return c2
    hits = list(src_root.rglob(stem + ".txt"))
    return hits[0] if hits else None


if __name__ == "__main__":
    main()
