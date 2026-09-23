# -*- coding: utf-8 -*-
"""
step1_scan_images.py —— 原始数据体检

在去重之前，先把「手上到底有什么」摸清楚，全部落盘成 JSON：
  - 每张图的宽高、文件大小、格式
  - 分辨率分布（直接支撑「距离域偏移」这个创新点的论证：
    公开数据集多为近距高分辨率，部署场景是远距低 GSD）
  - 标签统计：类别实例数、框的绝对像素尺寸分布、小目标占比

用法：
  python step1_scan_images.py
  python step1_scan_images.py --root=D:/xxx --out=D:/yyy

输出：
  04_结果/评估/数据体检_image_stats.json
  04_结果/评估/数据体检_label_stats.json
  logs/step1_scan_images.log（由 PowerShell 重定向）
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from common import (
    CLASSES,
    CLASS_TO_ID,
    RAW_DIR,
    AUDIT_DIR,
    EVAL_DIR,
    list_images,
    load_json,
    read_yolo_labels,
    read_lines_u,
    read_text_u,
    imread_u,
    dump_json,
    log,
    argv_flag,
    label_path_for,
    is_tiny_box,
    ensure_dirs,
)


def main() -> None:
    root = Path(argv_flag("root", str(RAW_DIR)))
    # 体检报告属于数据阶段产物，默认落在 01_data/audit（不是 04_results/eval）
    out_dir = Path(argv_flag("out", str(AUDIT_DIR)))
    ensure_dirs(out_dir)

    log(f"扫描根目录: {root}")
    if not root.exists():
        log("!! 目录不存在，请先下载数据（见 step0_download_roboflow.py）")
        return

    images = list_images(root)
    log(f"发现图像: {len(images)} 张")

    # ---------------- 图像侧统计 ----------------
    img_records = []
    res_counter: Counter = Counter()
    ext_counter: Counter = Counter()
    total_bytes = 0
    unreadable = []

    for i, img_path in enumerate(images, 1):
        img = imread_u(img_path)
        if img is None:
            unreadable.append(str(img_path))
            continue
        h, w = img.shape[:2]
        size = img_path.stat().st_size
        total_bytes += size
        img_records.append(
            {
                "path": str(img_path),
                "name": img_path.name,
                "w": int(w),
                "h": int(h),
                "bytes": int(size),
                "ext": img_path.suffix.lower(),
            }
        )
        res_counter[f"{w}x{h}"] += 1
        ext_counter[img_path.suffix.lower()] += 1
        if i % 500 == 0:
            log(f"  已扫描 {i}/{len(images)}")

    sides = sorted(min(r["w"], r["h"]) for r in img_records) if img_records else []
    def pct(lst, q):
        if not lst:
            return None
        k = max(0, min(len(lst) - 1, int(round(q * (len(lst) - 1)))))
        return lst[k]

    img_summary = {
        "root": str(root),
        "n_images": len(img_records),
        "n_unreadable": len(unreadable),
        "unreadable": unreadable[:50],
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024 / 1024, 2),
        "ext_distribution": dict(ext_counter),
        "top_resolutions": res_counter.most_common(20),
        "n_distinct_resolutions": len(res_counter),
        "min_side_px": {
            "min": sides[0] if sides else None,
            "p10": pct(sides, 0.10),
            "p50": pct(sides, 0.50),
            "p90": pct(sides, 0.90),
            "max": sides[-1] if sides else None,
        },
        "n_small_lt_640": sum(1 for s in sides if s < 640),
        "n_small_lt_416": sum(1 for s in sides if s < 416),
    }
    dump_json(img_summary, out_dir / "数据体检_image_stats.json")

    # ---------------- 标签侧统计 ----------------
    # 尝试多种常见布局，自动定位 images / labels
    label_stats = analyze_labels(root, img_records)

    log("---- 图像统计 ----")
    log(f"  数量 {img_summary['n_images']} 张，合计 {img_summary['total_mb']} MB")
    log(f"  不可读 {img_summary['n_unreadable']} 张")
    log(f"  分辨率种类 {img_summary['n_distinct_resolutions']} 种")
    log(f"  短边 p10/p50/p90 = {sides and (pct(sides,0.1), pct(sides,0.5), pct(sides,0.9))}")
    log("---- 标签统计 ----")
    log(f"  找到标签文件 {label_stats['n_label_files']} 个")
    log(f"  总实例数 {label_stats['n_instances']}")
    log(f"  类别分布 {label_stats['class_counts']}")
    log(f"  原始类别名 {label_stats['raw_class_names']}")
    log(f"  小目标(<6px)实例占比 {label_stats['tiny_ratio']}")
    log(f"完成，结果写入 {out_dir}")

    # 打印标签摘要，方便直接观察
    dump_json(label_stats, out_dir / "数据体检_label_stats.json")


def analyze_labels(root: Path, img_records: list[dict]) -> dict:
    """
    自动识别 YOLO 布局并统计标签。
    支持：
      <root>/images/**.jpg + <root>/labels/**.txt
      <root>/train/images + <root>/train/labels，val/ test/ 同理
    """
    by_name = {Path(r["path"]).name: r for r in img_records}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    txts = sorted([p for p in root.rglob("*.txt") if p.name != "classes.txt"])

    class_counts: Counter = Counter()
    raw_names: set[str] = set()
    box_wh_px: list[tuple[float, float]] = []
    per_file_cls: dict[str, set] = defaultdict(set)
    n_instances = 0
    tiny = 0
    matched = 0
    orphan_labels = []

    # 若存在 data.yaml / 自定义 yaml，尝试读取类别名
    name_map = try_read_names(root)

    for tp in txts:
        # 找同名图像
        owner = None
        for e in exts:
            cand = tp.with_suffix(e).name
            if cand in by_name:
                owner = by_name[cand]
                break
        if owner is None:
            orphan_labels.append(str(tp))
            if len(orphan_labels) > 200:
                continue
        else:
            matched += 1

        labels = read_yolo_labels(tp)
        for c, cx, cy, w, h in labels:
            n_instances += 1
            class_counts[c] += 1
            per_file_cls[tp.stem].add(c)
            if name_map and c in name_map:
                raw_names.add(name_map[c])
            if owner is not None:
                W, H = owner["w"], owner["h"]
                bw, bh = w * W, h * H
                box_wh_px.append((bw, bh))
                if is_tiny_box(w, h, W, H, min_px=6.0):
                    tiny += 1

    sides_px = sorted(min(a, b) for a, b in box_wh_px)
    areas_px = sorted(a * b for a, b in box_wh_px)

    def pct(lst, q):
        if not lst:
            return None
        k = max(0, min(len(lst) - 1, int(round(q * (len(lst) - 1)))))
        return round(lst[k], 2)

    # 归一化类别分布
    cls_named = {}
    for cid, n in sorted(class_counts.items()):
        nm = (name_map or {}).get(cid, CLASSES[cid] if cid < len(CLASSES) else f"id{cid}")
        cls_named[f"{cid}:{nm}"] = n

    return {
        "n_label_files": len(txts),
        "n_matched_to_image": matched,
        "n_orphan_labels": len(orphan_labels),
        "orphan_examples": orphan_labels[:30],
        "n_instances": n_instances,
        "class_counts": cls_named,
        "raw_class_names": sorted(raw_names),
        "box_min_side_px": {
            "min": pct(sides_px, 0.0),
            "p10": pct(sides_px, 0.10),
            "p50": pct(sides_px, 0.50),
            "p90": pct(sides_px, 0.90),
            "max": pct(sides_px, 1.0),
        },
        "box_area_px": {
            "p10": pct(areas_px, 0.10),
            "p50": pct(areas_px, 0.50),
            "p90": pct(areas_px, 0.90),
        },
        "n_tiny_lt6px": tiny,
        "tiny_ratio": round(tiny / n_instances, 4) if n_instances else None,
        "multi_class_files": sum(1 for v in per_file_cls.values() if len(v) > 1),
    }


def try_read_names(root: Path) -> dict:
    """尽力从 data.yaml / classes.txt / *.yaml 里读出 类别id->名称 映射。"""
    # classes.txt：每行一个类名，顺序即 id
    for ct in root.rglob("classes.txt"):
        try:
            lines = read_lines_u(ct)
            if lines:
                return {i: n for i, n in enumerate(lines)}
        except Exception:
            pass

    # *.yaml 里的 names: 块
    for y in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
        try:
            text = read_text_u(y)
        except Exception:
            continue
        if "names" not in text:
            continue
        m: dict = {}
        in_names = False
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("names:"):
                in_names = True
                # 行内列表形式 names: ['a','b']
                inline = s[len("names:"):].strip()
                if inline.startswith("["):
                    items = [x.strip().strip("'\"") for x in inline.strip("[]").split(",")]
                    for i, n in enumerate(items):
                        if n:
                            m[i] = n
                    if m:
                        return m
                continue
            if in_names:
                if not s or s.startswith("#"):
                    continue
                if ":" in s and not line.startswith((" ", "\t")):
                    break  # 回到顶层键，结束
                if ":" in s:
                    k, v = s.split(":", 1)
                    try:
                        m[int(k.strip())] = v.strip().strip("'\"")
                    except ValueError:
                        pass
        if m:
            return m
    return {}


if __name__ == "__main__":
    main()
