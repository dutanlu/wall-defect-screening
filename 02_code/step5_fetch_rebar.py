# -*- coding: utf-8 -*-
"""
step5_fetch_rebar.py —— 补充数据源：工程结构裂缝数据集（含「露筋」标注）

数据源：ModelScope `jiange1236/StructuralCrackDataset`（Apache-2.0）
学术描述：建筑结构裂缝标注数据集（含裂缝类型 + 构件类型 + 病害描述文本）

为什么需要这个脚本：
  主数据源 greybrick-yolo 只覆盖 crack/spalling/efflorescence/moss 四类，
  **`exposed_rebar`（露筋）样本数为 0**。露筋是本项目分级逻辑里最敏感的一类
  （钢筋失去保护层必然持续锈蚀 → 直接判 danger），没有数据就无法训练。

标注格式差异（关键，决定了必须写转换脚本）：
  该数据集是 **label-studio** 导出格式，不是 YOLO：
    - `annotations.csv` 每行一张图，含 `rectTool` / `polygonTool` 两个 JSON 字段
    - 坐标为**百分比**（0-100），不是归一化 0-1，且 y 轴向下
    - 一个 JSON 字段里是**标注列表**，每个元素带 `rectanglelabels` / `polygonlabels`
  因此需要：
    1. 解析 CSV → 逐图取所有框
    2. 类别名映射到本项目的 7 类体系
    3. 百分比 → YOLO 归一化(xywh) 并做越界裁剪
    4. 只保留映射成功的类（未映射的记数并报告，**不静默吞掉**）

输出（YOLO 布局，供 step2/step3 的既有流水线继续消费）：
  01_data/raw/rebar_structural/images/**.{jpg,png}
  01_data/raw/rebar_structural/labels/**.txt
  01_data/raw/rebar_structural/_fetch_report.json

用法：
  python step5_fetch_rebar.py --list      # 只看清单，不下载
  python step5_fetch_rebar.py             # 下载 + 解压 + 转换
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from common import (
    CLASSES,
    RAW_DIR,
    argv_flag,
    dump_json,
    ensure_dirs,
    imread_u,
    log,
    write_yolo_labels,
)

# --------------------------------------------------------------------------
DS_NAMESPACE = "jiange1236"
DS_NAME = "StructuralCrackDataset"
DS_REVISION = "master"
API_BASE = f"https://www.modelscope.cn/api/v1/datasets/{DS_NAMESPACE}/{DS_NAME}"
TREE_URL = f"{API_BASE}/repo/tree?Revision={DS_REVISION}&Recursive=true"
FILE_URL = f"{API_BASE}/repo?Revision={DS_REVISION}&FilePath={{fname}}"

OUT_ROOT = RAW_DIR / "rebar_structural"
ZIP_DIR = OUT_ROOT / "_zips"
SRC_DIR = OUT_ROOT / "src"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "Referer": "https://www.modelscope.cn/datasets"}

# --------------------------------------------------------------------------
# 类别映射：该数据集的原始标签 -> 本项目 7 类体系
# --------------------------------------------------------------------------
# 实测（annotations.csv 逐条读出）出现的原始标签：
#   "裂缝" / "露筋" / "保护层脱落"
# CSV 的 cracksClassification 列还有更细的裂缝分类：
#   受拉裂缝 / 受压裂缝 / 剪切裂缝 / 温度裂缝 / 沉降裂缝 / 收缩裂缝
#   / 钢筋露筋锈蚀 / 保护层脱落
# 注意：labels 里可能出现「裂缝」泛称，也可能是细分类，两种都要映射。
LABEL_MAP: dict[str, str] = {
    # ---- 露筋（本次补充的**唯一目标类**）----
    "露筋": "exposed_rebar",
    "钢筋露筋锈蚀": "exposed_rebar",     # 露筋与锈蚀在工程上强相关，合并
    "钢筋外露": "exposed_rebar",
    "exposed rebar": "exposed_rebar",
    # ---- 保护层脱落 -> 剥落缺失 ----
    "保护层脱落": "spalling",
    "sag of protecting coating": "spalling",
    # ---- 裂缝各类（含泛称）----
    "裂缝": "crack",
    "tensile cracks": "crack",
    "compression cracks": "crack",
    "shear cracks": "crack",
    "temperature cracks": "crack",
    "settlement cracks": "crack",
    "shrinkage cracks": "crack",
    "受拉裂缝": "crack",
    "受压裂缝": "crack",
    "剪切裂缝": "crack",
    "温度裂缝": "crack",
    "沉降裂缝": "crack",
    "收缩裂缝": "crack",
}

# 只保留这些类写进数据集（其余记数报告后丢弃）
KEEP_CLASSES: set[str] = {"exposed_rebar", "spalling", "crack"}


def http_get(url: str, timeout: int = 60) -> bytes:
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers=UA)
    with op.open(req, timeout=timeout) as r:
        return r.read()


def list_files() -> list[dict]:
    raw = http_get(TREE_URL)
    j = json.loads(raw.decode("utf-8", errors="replace"))
    files = (j.get("Data") or {}).get("Files") or []
    return [f for f in files if isinstance(f, dict)]


def download(url: str, dst: Path) -> int:
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers=UA)
    total = 0
    tmp = dst.with_suffix(dst.suffix + ".part")
    with op.open(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            blk = r.read(1 << 16)
            if not blk:
                break
            f.write(blk)
            total += len(blk)
    tmp.replace(dst)
    return total


def norm_label(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").split())


def parse_rects(rect_json: str, img_w: int, img_h: int) -> list[tuple[str, float]]:
    """
    解析 label-studio 的 rectTool 字段。

    返回 [(class_name, cx, cy, w, h), ...]，坐标为 YOLO 归一化 0-1。
    原始坐标是**百分比**(0-100)，需 /100。
    """
    out: list[tuple[str, float, float, float, float]] = []
    if not rect_json or not rect_json.strip():
        return out
    try:
        items = json.loads(rect_json)
    except Exception:
        return out
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        labels = it.get("rectanglelabels") or []
        if not labels:
            continue
        try:
            x = float(it.get("x", 0)) / 100.0          # 左上角 x（比例）
            y = float(it.get("y", 0)) / 100.0          # 左上角 y
            w = float(it.get("width", 0)) / 100.0
            h = float(it.get("height", 0)) / 100.0
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        cls_raw = norm_label(labels[0])
        mapped = None
        for k, v in LABEL_MAP.items():
            if norm_label(k) == cls_raw:
                mapped = v
                break
        if mapped is None:
            continue
        # label-studio 的 x/y 是左上角 -> 转中心点
        cx = x + w / 2.0
        cy = y + h / 2.0
        # 裁剪到 [0,1]（百分比坐标可能有轻微越界）
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        w2, h2 = x2 - x1, y2 - y1
        if w2 <= 1e-6 or h2 <= 1e-6:
            continue
        out.append((mapped, x1 + w2 / 2, y1 + h2 / 2, w2, h2))
    return out


def parse_polys(poly_json: str) -> list[tuple[str, float, float, float, float]]:
    """
    解析 label-studio 的 polygonTool 字段。

    返回 [(cls, cx, cy, w, h), ...]（YOLO 归一化）。
    多边形取**外接矩形** —— 因为我们做的是目标检测（bbox），不是分割。
    这是有损的（多边形可能是不规则裂缝），我们的数据管线本就面向 bbox。

    ⚠️ 第一版只解析了 rectTool，漏掉了本数据集里的 4 个多边形「裂缝」标注
       —— 已实测发现并补上。标注解析必须两个字段都覆盖。
    """
    out: list[tuple[str, float, float, float, float]] = []
    if not poly_json or not poly_json.strip():
        return out
    try:
        items = json.loads(poly_json)
    except Exception:
        return out
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        labels = it.get("polygonlabels") or []
        pts = it.get("points") or []
        if not labels or not isinstance(pts, list) or len(pts) < 3:
            continue
        cls_raw = norm_label(labels[0])
        mapped = None
        for k, v in LABEL_MAP.items():
            if norm_label(k) == cls_raw:
                mapped = v
                break
        if mapped is None:
            continue
        # 注意：多边形坐标同样是百分比(0-100)
        try:
            xs = [float(p[0]) / 100.0 for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
            ys = [float(p[1]) / 100.0 for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
        except (TypeError, ValueError):
            continue
        if not xs or not ys:
            continue
        x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
        y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
        w, h = x2 - x1, y2 - y1
        if w <= 1e-6 or h <= 1e-6:
            continue
        out.append((mapped, x1 + w / 2, y1 + h / 2, w, h))
    return out


def main() -> None:
    do_list = argv_flag("list") is not None
    ensure_dirs(OUT_ROOT, ZIP_DIR, SRC_DIR)

    log(f"数据源: {DS_NAMESPACE}/{DS_NAME}  (Apache-2.0)")
    files = list_files()
    log(f"远端文件 {len(files)} 个")
    for f in files:
        p = f.get("Path") or ""
        sz = f.get("Size") or 0
        log(f"  {p}  ({sz/1024:.0f} KB)")

    if do_list:
        log("--list 模式，未下载")
        return

    # ---- 下载 ----
    # ⚠️ 该数据集**没有 zip 打包**：27 张图 + 3 个标注文件都是独立对象。
    # 因此必须按后缀分类下载，不能只挑 .zip（第一版就是只挑 zip，
    # 结果图片一张没下来、转换阶段直接报「图像目录不存在」——已实测踩到）。
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    META_EXT = {".csv", ".json", ".jsonl", ".py", ".md", ".txt"}

    zips = [f for f in files if (f.get("Path") or "").lower().endswith(".zip")]
    metas = [f for f in files if Path(f.get("Path") or "").suffix.lower() in META_EXT]
    imgs = [f for f in files if Path(f.get("Path") or "").suffix.lower() in IMG_EXT]

    log(f"待下载: zip {len(zips)} / 元数据 {len(metas)} / 图像 {len(imgs)}")

    for f in zips:
        name = Path(f["Path"]).name
        dst = ZIP_DIR / name
        if dst.exists() and dst.stat().st_size == (f.get("Size") or 0):
            log(f"已存在且大小一致，跳过: {name}")
            continue
        try:
            n = download(FILE_URL.format(fname=f["Path"]), dst)
            log(f"已下载 {name}  {n/1024/1024:.2f} MB")
        except Exception as e:
            log(f"!! 下载失败 {name}: {e!r}")

    # 元数据与图像：保持远端相对路径
    for f in metas + imgs:
        rel = f["Path"]
        dst = SRC_DIR / rel
        ensure_dirs(dst.parent)
        want = f.get("Size") or 0
        if dst.exists() and (want == 0 or dst.stat().st_size == want):
            continue
        try:
            data = http_get(FILE_URL.format(fname=rel))
            dst.write_bytes(data)
            ok = "" if (want == 0 or len(data) == want) else f"  !! 大小不符(期望{want})"
            log(f"已下载 {rel}  ({len(data)} B){ok}")
        except Exception as e:
            log(f"!! 下载失败 {rel}: {e!r}")

    # 解压
    for z in sorted(ZIP_DIR.glob("*.zip")):
        dest = SRC_DIR / z.stem
        ensure_dirs(dest)
        try:
            with zipfile.ZipFile(z) as zf:
                bad = zf.testzip()
                if bad:
                    log(f"!! {z.name} 损坏于 {bad}")
                    continue
                zf.extractall(dest)
            n = sum(1 for _ in dest.rglob("*") if _.is_file())
            log(f"已解压 {z.name} -> {dest}  ({n} 文件)")
        except Exception as e:
            log(f"!! 解压失败 {z.name}: {e!r}")

    # ---- 定位 annotations.csv ----
    csvs = list(SRC_DIR.rglob("annotations.csv"))
    if not csvs:
        log("!! 未找到 annotations.csv，无法转换")
        return
    ann = csvs[0]
    log(f"标注文件: {ann}")

    img_root = ann.parent / "images"
    if not img_root.exists():
        log(f"!! 图像目录不存在: {img_root}")
        return

    # 读 CSV（image 列形如 images/xxx.png）
    rows: list[dict] = []
    with open(ann, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    log(f"CSV 记录数: {len(rows)}")

    out_img = OUT_ROOT / "images"
    out_lab = OUT_ROOT / "labels"
    ensure_dirs(out_img, out_lab)

    stats = Counter()
    dropped_labels: Counter = Counter()
    per_class_img = Counter()
    n_ok = 0
    n_no_img = 0
    n_no_box = 0

    for row in rows:
        rel_img = (row.get("image") or "").strip()
        if not rel_img:
            continue
        src_img = ann.parent / rel_img
        if not src_img.exists():
            # CSV 里是 images/xxx，可能在别的层级
            cands = list(SRC_DIR.rglob(Path(rel_img).name))
            if not cands:
                n_no_img += 1
                continue
            src_img = cands[0]

        img = imread_u(src_img)
        if img is None:
            n_no_img += 1
            continue
        h, w = img.shape[:2]

        boxes = parse_rects(row.get("rectTool") or "", w, h)
        boxes += parse_polys(row.get("polygonTool") or "")

        # 统计未映射的标签（不静默吞掉）
        for raw_field in (row.get("rectTool") or "", row.get("polygonTool") or ""):
            if not raw_field.strip():
                continue
            try:
                items = json.loads(raw_field)
            except Exception:
                continue
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                for key in ("rectanglelabels", "polygonlabels"):
                    for lb in (it.get(key) or []):
                        if norm_label(lb) not in {norm_label(k) for k in LABEL_MAP}:
                            dropped_labels[lb] += 1

        keep = [b for b in boxes if b[0] in KEEP_CLASSES]

        # 复制图像（加前缀避免与其它源撞名）
        dst_img = out_img / f"sc_{src_img.stem}{src_img.suffix.lower()}"
        if not dst_img.exists():
            try:
                import shutil
                shutil.copy2(src_img, dst_img)
            except Exception as e:
                log(f"!! 复制失败 {src_img.name}: {e!r}")
                continue

        if keep:
            labs = [(CLASSES.index(c), cx, cy, bw, bh) for c, cx, cy, bw, bh in keep]
            write_yolo_labels(out_lab / f"sc_{src_img.stem}.txt", labs)
            for c, *_ in keep:
                stats[c] += 1
            per_class_img.update({c for c, *_ in keep})
            n_ok += 1
        else:
            # 没有可用框 → 写空标签（YOLO 语义：负样本/背景图）
            (out_lab / f"sc_{src_img.stem}.txt").write_text("", encoding="utf-8")
            n_no_box += 1

    report = {
        "source": f"{DS_NAMESPACE}/{DS_NAME}",
        "license": "Apache-2.0",
        "annotation_file": str(ann),
        "csv_rows": len(rows),
        "images_written": n_ok + n_no_box,
        "images_with_boxes": n_ok,
        "images_empty_label": n_no_box,
        "images_missing": n_no_img,
        "instances_per_class": {k: stats.get(k, 0) for k in CLASSES if stats.get(k)},
        "instances_per_class_cn": {
            k: stats[k] for k in stats
        },
        "images_per_class": {k: per_class_img.get(k, 0)
                            for k in CLASSES if per_class_img.get(k)},
        "unmapped_labels": dict(dropped_labels),
        "label_map_used": LABEL_MAP,
        "keep_classes": sorted(KEEP_CLASSES),
        "note": ("本数据集为 label-studio 导出格式（rectTool 百分比坐标），"
                 "已转换为 YOLO 归一化 xywh。只保留 KEEP_CLASSES 中的类，"
                 "其余标签记入 unmapped_labels 供核对，不静默丢弃。"),
    }
    dump_json(report, OUT_ROOT / "_fetch_report.json")

    log("=" * 70)
    log(f"图像写出 {n_ok + n_no_box}（其中 {n_ok} 张有可用框，{n_no_box} 张空标签）")
    log(f"缺图 {n_no_img}")
    log("各类实例数:")
    for k in CLASSES:
        if stats.get(k):
            log(f"  {k:16s} {stats[k]}")
    if dropped_labels:
        log("未映射标签（已丢弃，供核对）:")
        for k, v in dropped_labels.most_common():
            log(f"  {k!r} x{v}")
    log(f"报告: {OUT_ROOT / '_fetch_report.json'}")
    log("下一步：step2_dedup.py --root=..\\01_data\\raw\\rebar_structural\\images")
    log("=" * 70)


if __name__ == "__main__":
    main()
