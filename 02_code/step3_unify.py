# -*- coding: utf-8 -*-
"""
step3_unify_classes.py —— 多源类别体系归一

问题：不同数据集的类别命名/粒度都不同，直接混在一起训练必然错乱。
  jq3/building_defect            : crack, corrosion, mold, peeling, spalling
  surface-defects-in-heritage    : Crack, Corrosion, Vegetation, Blackening, Brick Spalling
  paint-defects                  : delamination, efflorescence, exposed rebar, scaling, ...
  ...

本脚本把每个源的类别名映射到项目统一的 6 类：
  crack / spalling / efflorescence / exposed_rebar / rust / delamination

映射表由 ROBOFLOW_MAP 提供，可在此文件里增改。未命中的类别会：
  - 默认「丢弃该类实例」（可配置）
  - 并在报告里列出，提示你补映射，绝不静默吞掉

用法：
  python step3_unify_classes.py --src=jq3_building_defect --in=D:/... --out=D:/...
  python step3_unify_classes.py            # 处理 00_原始下载 下所有子目录

输出：
  01_数据/02_统一类别/<src>/**
  04_结果/评估/class_unify_report.json
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

from common import (
    CLASSES,
    CLASS_TO_ID,
    DEDUP_DIR,
    UNIFIED_DIR,
    EVAL_DIR,
    argv_flag,
    clean_class_name as norm,
    dump_json,
    ensure_dirs,
    is_skipped,
    list_images,
    log,
    read_lines_u,
    read_text_u,
    read_yolo_labels,
    write_yolo_labels,
)

# --------------------------------------------------------------------------
# 源 -> 统一类 映射表（键统一按小写去空格匹配）
# 规则：源类别名（小写） -> 统一类名，或 None 表示丢弃
# --------------------------------------------------------------------------
ROBOFLOW_MAP: dict[str, dict[str, str | None]] = {
    # ----------------------------------------------------------------------
    # 主力数据源：ModelScope 古建筑青砖表面损伤数据集（YOLO 格式，Apache-2.0）
    # https://www.modelscope.cn/datasets/destinylhj/greybrick-yolo
    # 学术出处：Gray Brick Wall Surface Damage Detection of Traditional Chinese
    #          Buildings in Macau（澳门岭南青砖建筑，375 张标注图 / 8 类）
    # 原始类别：CRACK / W_E / ALKALI / MISS / MOSS
    # 用户决策（2026-09-20）：MOSS 保留为独立第 7 类，不丢弃、不并入。
    # ----------------------------------------------------------------------
    "greybrick-yolo": {
        "crack": "crack",
        "w_e": "efflorescence",       # Water Effect：水渍/渗水 ↔ 泛碱渗水（同属水致劣化）
        "w e": "efflorescence",       # norm 会把 W_E 归一成 "w e"
        "alkali": "efflorescence",    # 泛碱
        "miss": "spalling",           # 砖块缺失 ↔ 剥落掉块
        "moss": "moss",               # 苔藓附着，独立第 7 类
    },
    # https://universe.roboflow.com/jq3/building_defect
    "jq3_building_defect": {
        "crack": "crack",
        "corrosion": "rust",
        "mold": "efflorescence",
        "peeling": "delamination",
        "spalling": "spalling",
    },
    # https://universe.roboflow.com/defects-dataset/surface-defects-in-heritage
    "surface-defects-in-heritage": {
        "crack": "crack",
        "corrosion": "rust",
        "vegetation": None,          # 植被不是外墙结构缺陷，丢弃
        "blackening": "efflorescence",
        "brick spalling": "spalling",
        "spalling": "spalling",
    },
    # https://universe.roboflow.com/main-zxmvk/paint-defects-dfbjj
    "paint-defects": {
        "delamination": "delamination",
        "efflorescence": "efflorescence",
        "exposed rebar": "exposed_rebar",
        "exposed_rebar": "exposed_rebar",
        "rust": "rust",
        "rusting": "rust",
        "scaling": "spalling",
        "peeling": "delamination",
        "crack": "crack",
        "cracking": "crack",
        "spalling": "spalling",
        "blistering": "delamination",
        "damp": "efflorescence",
        "moisture": "efflorescence",
    },
    # ----------------------------------------------------------------------
    # 露筋/结构裂缝补充源：jiange1236/StructuralCrackDataset（Apache-2.0）
    # 由 step5_fetch_rebar.py 落盘。该源的标签已是「项目 7 类」id 空间
    # （0=crack / 1=spalling / 3=exposed_rebar），故这里做恒等映射。
    #
    # 用户决策（2026-09-20）：只并入裂缝框。
    #   - exposed_rebar：该源仅 9 框（2 张图），远不足以训练一个类别，硬塞进主集
    #     只会作为噪声拉低整体指标，故显式丢弃、继续隔离（技术报告 §4.4）。
    #   - spalling：该源仅 2 框（1 张图），同样显式丢弃。
    # 被丢弃的实例会在报告的 unmapped_classes 里以「xxx (显式丢弃)」留痕，不静默吞掉。
    # 实测该源 27 图 / 90 框 → 存活 25 图 / 79 个 crack 框。
    # ----------------------------------------------------------------------
    "rebar_structural": {
        "crack": "crack",
        "spalling": None,            # 仅 2 框，隔离
        "exposed_rebar": None,       # 仅 9 框，隔离
        # 注意：common.clean_class_name 会把下划线归一成空格（W_E -> "w e"），
        # 故两种写法都必须登记，否则该条会被判成「未映射、需补表」，
        # 而不是「显式丢弃」——审计口径会失真（已实测踩到）。
        "exposed rebar": None,
        "efflorescence": "efflorescence",
        "rust": "rust",
        "delamination": "delamination",
        "moss": "moss",
    },
    # 通用兜底（命中不到具体源时使用）
    "__generic__": {
        "crack": "crack",
        "cracks": "crack",
        "cracking": "crack",
        "craquelure": "crack",
        "spall": "spalling",
        "spalling": "spalling",
        "peel": "delamination",
        "peeling": "delamination",
        "paint peel": "delamination",
        "delamination": "delamination",
        "blistering": "delamination",
        "rust": "rust",
        "rusting": "rust",
        "corrosion": "rust",
        "exposed rebar": "exposed_rebar",
        "exposed_rebar": "exposed_rebar",
        "rebar": "exposed_rebar",
        "efflorescence": "efflorescence",
        "mold": "efflorescence",
        "mould": "efflorescence",
        "damp": "efflorescence",
        "moisture": "efflorescence",
        "water stain": "efflorescence",
        "water": "efflorescence",
        "alkali": "efflorescence",
        "moss": "moss",
        "mold moss": "moss",
        # --- 古建筑青砖数据集（greybrick-yolo）的 5 个原始类名 ---
        # 必须同时放进兜底表：pick_map() 是按「源目录名」选表的，
        # 而源目录常被重命名（如扁平化后叫 images），此时会落到兜底表。
        # 只写进 "greybrick-yolo" 条目会导致 W_E / MISS 全部未映射（已实测踩到）。
        "w_e": "efflorescence",       # norm 会把 W_E 变成 "w e"，故两种写法都登记
        "w e": "efflorescence",
        "water effect": "efflorescence",
        "miss": "spalling",           # 砖块缺失/破损 → 剥落掉块
        "missing": "spalling",
        "vegetation": None,
        "shadow": None,
    },
}


# norm 直接复用 common.clean_class_name（已处理 BOM），此处不再重复定义


def pick_map(src_name: str) -> dict[str, str | None]:
    """优先精确匹配源名，其次做包含匹配，最后用通用表。"""
    key = norm(src_name)
    for k, v in ROBOFLOW_MAP.items():
        if k == "__generic__":
            continue
        if norm(k) == key:
            return v
    for k, v in ROBOFLOW_MAP.items():
        if k == "__generic__":
            continue
        nk = norm(k)
        if nk and (nk in key or key in nk):
            return v
    return ROBOFLOW_MAP["__generic__"]


def build_name_lookup(label_root: Path) -> dict[int, str]:
    """
    从数据集的 classes.txt / *.yaml 建立 id -> 类名。
    找不到时返回空 dict，调用方会退化为「按 id 顺序猜测」并告警。
    """
    for ct in label_root.rglob("classes.txt"):
        try:
            lines = read_lines_u(ct)
            if lines:
                log(f"  类别名来源: {ct}")
                return {i: n for i, n in enumerate(lines)}
        except Exception:
            pass
    for y in list(label_root.rglob("*.yaml")) + list(label_root.rglob("*.yml")):
        try:
            text = read_text_u(y)
        except Exception:
            continue
        if "names" not in text:
            continue
        m: dict[int, str] = {}
        in_names = False
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("names:"):
                in_names = True
                inline = s[len("names:"):].strip()
                if inline.startswith("["):
                    items = [x.strip().strip("'\"") for x in inline.strip("[]").split(",")]
                    for i, n in enumerate(items):
                        if n:
                            m[i] = n
                    if m:
                        log(f"  类别名来源: {y}")
                        return m
                continue
            if in_names:
                if not s or s.startswith("#"):
                    continue
                if ":" in s and not line.startswith((" ", "\t")):
                    break
                if ":" in s:
                    k, v = s.split(":", 1)
                    try:
                        m[int(k.strip())] = v.strip().strip("'\"")
                    except ValueError:
                        pass
        if m:
            log(f"  类别名来源: {y}")
            return m
    return {}


def process_one(src_dir: Path, out_root: Path) -> dict:
    """处理单个源目录，返回统计。"""
    log(f"[源] {src_dir.name}")
    name_lookup = build_name_lookup(src_dir)
    mapping = pick_map(src_dir.name)
    cache: dict = {}

    raw_counts: Counter = Counter()
    mapped_counts: Counter = Counter()
    unmapped: Counter = Counter()            # 映射表里查不到 -> 需要补表（真缺陷）
    explicitly_dropped: Counter = Counter()  # 映射表里显式为 None -> 按设计丢弃（非缺陷）
    n_img = 0
    n_lab = 0
    n_inst_in = 0
    n_inst_out = 0
    n_dropped = 0
    n_images_without_label = 0
    n_images_all_dropped = 0

    images = list_images(src_dir)

    # 输出子目录名：扁平布局时 src_dir 常叫 images/dedup，用其父名或自身更有意义
    out_name = src_dir.name
    if out_name.lower() in {"images", "dedup", "01_去重后", "去重后"}:
        out_name = src_dir.name if src_dir.parent == Path(src_dir.anchor) else src_dir.parent.name
        out_name = f"{src_dir.name}_unified"
    out_src = out_root / out_name
    for img_path in images:
        n_img += 1
        # 找对应 label
        lab = find_label(img_path, src_dir, cache)
        if lab is None:
            n_images_without_label += 1
            continue
        labels = read_yolo_labels(lab)
        n_lab += 1
        new_labels = []
        for c, cx, cy, w, h in labels:
            n_inst_in += 1
            raw_name = name_lookup.get(c, f"id{c}")
            raw_counts[raw_name] += 1
            target = mapping.get(norm(raw_name))
            if target is None:
                # 未命中：既不在映射表里，也不是显式 None
                if norm(raw_name) in mapping:
                    n_dropped += 1              # 映射表里显式为 None：按设计丢弃
                    explicitly_dropped[raw_name] += 1
                else:
                    unmapped[raw_name] += 1      # 映射表里查不到：需要补映射
                    n_dropped += 1
                continue
            new_labels.append((CLASS_TO_ID[target], cx, cy, w, h))
            mapped_counts[target] += 1
            n_inst_out += 1

        if not new_labels:
            # 注意：这些实例在上面已按「未映射 / 显式丢弃」逐个计入 n_dropped，
            # 此处若再 += len(labels) 会重复计数（实测：整图被丢弃的样本被算两遍）。
            # 故单列一个「整图丢弃」计数，实例数不重复加。
            # 另：标签文件为空（0 实例）的图也走这一支 —— 这是 wall_defects 的
            # 背景/无缺陷图，会在报告里以 n_images_all_dropped 显形，
            # 正好解释「677 张去重后唯一图 → 649 对可用样本」的差额，不再静默消失。
            n_images_all_dropped += 1
            continue

        # 复制图像 + 写新标签，保持 images/labels 布局
        rel = img_path.relative_to(src_dir)
        # 把路径里的 images 段换成 labels 段
        parts = list(rel.parts)
        parts = ["labels" if p == "images" else p for p in parts]
        out_img = out_src / "images" / rel
        out_lab = out_src / "labels" / Path(*parts).with_suffix(".txt")
        ensure_dirs(out_img.parent, out_lab.parent)
        shutil.copy2(img_path, out_img)
        write_yolo_labels(out_lab, new_labels)

    log(f"  图像 {n_img}，标签 {n_lab}，实例 {n_inst_in} → {n_inst_out}"
        f"（丢弃实例 {n_dropped}，整图丢弃 {n_images_all_dropped}）")
    if explicitly_dropped:
        log(f"  按映射表显式丢弃的类别（设计决策，非缺陷）: {dict(explicitly_dropped)}")
    if unmapped:
        log(f"  !! 未映射类别（需补 ROBOFLOW_MAP）: {dict(unmapped)}")

    return {
        "source": src_dir.name,
        "out_name": out_name,
        "n_images": n_img,
        "n_labels": n_lab,
        "n_images_without_label": n_images_without_label,
        "n_instances_in": n_inst_in,
        "n_instances_out": n_inst_out,
        "n_dropped": n_dropped,
        "n_images_all_dropped": n_images_all_dropped,
        "explicitly_dropped_classes": dict(explicitly_dropped),
        "class_names_found": dict(name_lookup),
        "mapping_used": {k: v for k, v in mapping.items()},
        "raw_class_counts": dict(raw_counts),
        "unified_class_counts": dict(mapped_counts),
        "unmapped_classes": dict(unmapped),
    }


def find_label(img_path: Path, src_root: Path, cache: dict | None = None) -> Path | None:
    """
    在若干常见布局里寻找图像对应的标签文件。

    顺序：
      1. 同目录同名 .txt（最稳）
      2. 把路径中的 images 段换成 labels 段
      3. 兜底：全树同名 txt（慢，仅在 1/2 都失败时用；用 cache 避免重复扫描）
    """
    c1 = img_path.with_suffix(".txt")
    if c1.exists():
        return c1

    s = str(img_path)
    for a, b in (("/images/", "/labels/"), ("\\images\\", "\\labels\\")):
        if a in s:
            c2 = Path(s.replace(a, b)).with_suffix(".txt")
            if c2.exists():
                return c2

    # 兜底：建立一次全树索引，后续 O(1) 查询
    if cache is None:
        cache = {}
    if "_txt_index" not in cache:
        cache["_txt_index"] = {}
        for t in src_root.rglob("*.txt"):
            if t.name == "classes.txt":
                continue
            # 跳过隔离区/备份目录，否则重复样本的标签会混进来
            if is_skipped(t.relative_to(src_root)):
                continue
            cache["_txt_index"].setdefault(t.stem, t)
    return cache["_txt_index"].get(img_path.stem)


def detect_sources(in_root: Path) -> list[Path]:
    """
    判断输入根下面「哪些子目录算一个独立数据源」。

    这是本项目最容易搞错的一处逻辑，必须区分两种情况：
      A) 多源嵌套：<root>/<源名>/images/*.jpg + <root>/<源名>/labels/*.txt
         -> 每个 <源名> 是一个源
      B) 单源扁平：<root>/images/*.jpg + <root>/labels/*.txt + <root>/classes.txt
         -> <root> 本身是一个源；此时绝不能把 images/labels 当源，
            否则 classes.txt 会落在扫描范围之外，类别名全部退化成 id0/id1

    判别依据：若子目录名属于 {images, labels, valid, test, train, val} 这类
    结构关键字，则视为「单源扁平」布局。
    """
    STRUCT_KEYS = {"images", "labels", "train", "val", "valid", "test"}
    subs = [d for d in sorted(in_root.iterdir()) if d.is_dir() and not d.name.startswith("_")]
    sub_names = {d.name.lower() for d in subs}

    # 情况 B：根下直接有 images/ 或 labels/
    if sub_names & {"images", "labels"}:
        return [in_root]

    # 情况 A：子目录是数据源名（且不是结构关键字）
    meaningful = [d for d in subs if d.name.lower() not in STRUCT_KEYS]
    return meaningful if meaningful else [in_root]


def main() -> None:
    in_root = Path(argv_flag("in", str(DEDUP_DIR / "images")))
    out_root = Path(argv_flag("out", str(UNIFIED_DIR)))
    one = argv_flag("src")

    ensure_dirs(out_root, EVAL_DIR)

    # 输入可能是「多个源子目录」或「一个扁平目录」
    if one:
        targets = [Path(one)] if Path(one).exists() else [in_root / one]
    elif in_root.exists():
        targets = detect_sources(in_root)
    else:
        targets = [in_root]

    log(f"输入根: {in_root}")
    log(f"待处理源: {[t.name for t in targets]}")

    reports = []
    for t in targets:
        if not t.exists():
            log(f"跳过（不存在）: {t}")
            continue
        reports.append(process_one(t, out_root))

    # 汇总
    total = Counter()
    for r in reports:
        for k, v in r["unified_class_counts"].items():
            total[k] += v

    summary = {
        "input_root": str(in_root),
        "output_root": str(out_root),
        "unified_classes": CLASSES,
        "per_source": reports,
        "total_class_counts": {c: total.get(c, 0) for c in CLASSES},
        "total_instances": sum(total.values()),
        "sources_with_unmapped": [r["source"] for r in reports if r["unmapped_classes"]],
    }
    dump_json(summary, EVAL_DIR / "class_unify_report.json")

    log("=" * 60)
    log("类别归一完成，统一 7 类实例数：")
    for c in CLASSES:
        log(f"  {c:16s} {total.get(c, 0):6d}")
    if summary["sources_with_unmapped"]:
        log(f"!! 有源存在未映射类别，请补 ROBOFLOW_MAP：{summary['sources_with_unmapped']}")
    log("=" * 60)


if __name__ == "__main__":
    main()
