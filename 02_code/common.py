# -*- coding: utf-8 -*-
"""
common.py —— 外墙缺陷筛查项目公共工具

集中解决三件事：
1. 中文路径读写（cv2.imread 在中文路径下返回 None，必须走 np.fromfile + cv2.imdecode）
2. 项目路径常量（全部相对本文件定位，换机器只需保证目录结构一致）
3. 统一 JSON / 日志落盘，保证每一步都可追溯

所有脚本都应 `from common import *` 而不是各自重复造轮子。
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# 路径常量（全部使用英文目录名，避免中文路径在工具链各处的兼容问题）
# --------------------------------------------------------------------------
# common.py 位于 <项目根>/02_code/common.py
CODE_DIR: Path = Path(__file__).resolve().parent
PROJ_ROOT: Path = CODE_DIR.parent

DATA_DIR: Path = PROJ_ROOT / "01_data"
RAW_DIR: Path = DATA_DIR / "raw"
DEDUP_DIR: Path = DATA_DIR / "dedup"
UNIFIED_DIR: Path = DATA_DIR / "unified"
DATASET_DIR: Path = DATA_DIR / "dataset"
# 数据阶段的报告（体检/去重/统一/切分）落在数据目录下，不要占用 04_results/eval
# —— 04_results/eval 的语义是「模型评估指标」，混入数据报告会让产物归属混乱。
AUDIT_DIR: Path = DATA_DIR / "audit"
WEIGHTS_DIR: Path = PROJ_ROOT / "03_weights"
RESULT_DIR: Path = PROJ_ROOT / "04_results"
TRAIN_DIR: Path = RESULT_DIR / "train"
EVAL_DIR: Path = RESULT_DIR / "eval"
VIS_DIR: Path = RESULT_DIR / "vis"
ABLATION_DIR: Path = RESULT_DIR / "ablation"
QUANT_DIR: Path = PROJ_ROOT / "05_quantify_grade"
DEPLOY_DIR: Path = PROJ_ROOT / "06_deploy"
REPORT_DIR: Path = PROJ_ROOT / "07_report"
LOG_DIR: Path = PROJ_ROOT / "logs"

# 数据集配置文件名（英文，避免 ultralytics 在中文文件名上的解析问题）
DATASET_YAML_NAME: str = "wall_defects.yaml"

# --------------------------------------------------------------------------
# 布局自适应（2026-09-25 新增）：兼容「工程包目录重命名」后的拷贝
# --------------------------------------------------------------------------
# 源工程布局：01_data/dataset, 03_weights, 04_results, 05_quantify_grade,
#             06_deploy, 07_report
# 评测包布局：dataset,        weights,    results,   grading_rules,
#             deploy,         report
# 两者常量名完全一致、仅目录名不同，故此处做一次探测：经典目录不存在、
# 而重命名目录存在时，把常量改指后者。源工程内经典目录存在 ⇒ 行为不变。
def _adapt_layout() -> None:
    globals_ = globals()

    def pick(canon: "Path", alt_name: str) -> "Path":
        alt = PROJ_ROOT / alt_name
        return alt if (not canon.exists() and alt.exists()) else canon

    # 注意：不能写 `DATA_DIR = pick(DATA_DIR, ...)` —— 赋值会让 DATA_DIR
    # 在函数内变成局部变量，右侧读取时 UnboundLocalError（已实测踩到）。
    # 仅当「经典 01_data/dataset 不存在、而 <root>/dataset 存在」时才改写
    # DATASET_DIR；否则保持原值（源工程即此分支）。
    # 早期版本在这里无条件 globals_["DATASET_DIR"] = data_dir，
    # 结果把源工程的 DATASET_DIR 从 01_data/dataset 错改成 01_data（已实测踩到）。
    classic_data = globals_["DATA_DIR"]
    alt_data = PROJ_ROOT / "dataset"
    if not classic_data.exists() and alt_data.exists():
        globals_["DATA_DIR"] = alt_data
        globals_["DATASET_DIR"] = alt_data
        globals_["RAW_DIR"] = alt_data / "raw"
        globals_["DEDUP_DIR"] = alt_data / "dedup"
        globals_["UNIFIED_DIR"] = alt_data / "unified"
        globals_["AUDIT_DIR"] = alt_data / "audit"
    globals_["WEIGHTS_DIR"] = pick(globals_["WEIGHTS_DIR"], "weights")
    globals_["RESULT_DIR"] = pick(globals_["RESULT_DIR"], "results")
    globals_["TRAIN_DIR"] = globals_["RESULT_DIR"] / "train"
    globals_["EVAL_DIR"] = globals_["RESULT_DIR"] / "eval"
    globals_["VIS_DIR"] = globals_["RESULT_DIR"] / "vis"
    globals_["ABLATION_DIR"] = globals_["RESULT_DIR"] / "ablation"
    globals_["QUANT_DIR"] = pick(globals_["QUANT_DIR"], "grading_rules")
    globals_["DEPLOY_DIR"] = pick(globals_["DEPLOY_DIR"], "deploy")
    globals_["REPORT_DIR"] = pick(globals_["REPORT_DIR"], "report")


_adapt_layout()

# --------------------------------------------------------------------------
# 统一类别体系（7 类，全项目唯一真源，任何脚本不得另行硬编码）
# --------------------------------------------------------------------------
# 变更记录：2026-09-20 由 6 类扩展为 7 类，新增 moss。
# 原因：主力数据源 greybrick-yolo 含 MOSS（苔藓附着）类且有 122 个实例，
#      用户决定保留为独立类别而非丢弃/并入，故类别体系扩为 7 类。
CLASSES: list[str] = [
    "crack",          # 裂缝 / 开裂
    "spalling",       # 剥落 / 掉块 / 砖块缺失 / 混凝土脱落
    "efflorescence",  # 泛碱 / 渗水 / 水渍 / 潮湿
    "exposed_rebar",  # 露筋        （V3 已并入：HRCDS 源 1389 框）
    "rust",           # 锈迹 / 钢筋锈蚀（V3 已并入：Urban 等源 13685 框）
    "delamination",   # 空鼓 / 分层 / 起皮
    "moss",           # 苔藓 / 生物附着  ← 2026-09-20 新增
]
CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
CLASS_CN: dict[str, str] = {
    "crack": "裂缝",
    "spalling": "剥落缺失",
    "efflorescence": "泛碱渗水",
    "exposed_rebar": "露筋",
    "rust": "锈迹",
    "delamination": "空鼓分层",
    "moss": "苔藓附着",
}

IMG_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 扫描图像时永远跳过的目录名（去重隔离区、备份、缓存等）。
# 漏掉这个会导致 step4 把「被移除的重复图」当成正常样本重新放进训练集 —— 已实测踩到。
SKIP_DIR_NAMES: set[str] = {
    "_removed_duplicates", "_quarantine", "_rejected",
    "__pycache__", ".ipynb_checkpoints", "_backup", "_old",
}
SKIP_DIR_PREFIXES: tuple[str, ...] = ("_", ".")


# --------------------------------------------------------------------------
# 中文路径安全读写（本项目最关键的一处封装，勿绕过）
# --------------------------------------------------------------------------
def imread_u(path) -> np.ndarray | None:
    """读取图像，兼容中文/空格路径。失败返回 None。"""
    p = Path(path)
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_u(path, img) -> bool:
    """写入图像，兼容中文/空格路径。成功返回 True。"""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix if p.suffix else ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(p))
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# 文本安全读写（兼容 BOM / UTF-16 / GBK）
# --------------------------------------------------------------------------
def read_text_u(path) -> str:
    """
    读文本文件，自动兼容 UTF-8 BOM / UTF-16 / GBK。

    为什么需要：用户用记事本或 PowerShell `Set-Content -Encoding UTF8`
    编辑 classes.txt 时会写入 BOM(U+FEFF)，导致 'crack' 变成 '\\ufeffcrack'
    而匹配不上映射表 —— 本项目在冒烟测试里已实测踩到这个坑。
    """
    p = Path(path)
    raw = p.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_lines_u(path) -> list[str]:
    """读文本并按行返回，自动处理编码与行尾，过滤空行。"""
    return [l.strip() for l in read_text_u(path).splitlines() if l.strip()]


def clean_class_name(s: str) -> str:
    """类名归一化：去 BOM、去首尾空白、统一小写、把下划线/连字符/多空格压成单空格。"""
    return " ".join(
        str(s).lstrip("\ufeff").strip().lower().replace("_", " ").replace("-", " ").split()
    )


# --------------------------------------------------------------------------
# 哈希与去重
# --------------------------------------------------------------------------
def md5_of_file(path, chunk: int = 1 << 20) -> str:
    """文件级 MD5（用于识别完全相同的副本，比像素级更快）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            blk = f.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def content_hash(img: np.ndarray, size: int = 32) -> str:
    """
    像素级感知哈希（用于兜住「同一张图但被重编码/改分辨率」的近似副本）。

    做法：缩放 -> 统一转 BGR 8bit -> 取均值中心裁剪 -> dHash。
    这能识别 Roboflow 增强副本里「resize / 轻微重压缩」的一类，
    而纯 MD5 对这类无能为力。
    """
    if img is None:
        return ""
    try:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return ""
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = g[:, 1:] > g[:, :-1]
    return "".join("1" if b else "0" for b in diff.flatten())


def hamming(a: str, b: str) -> int:
    """两个等长二进制字符串的汉明距离。"""
    if not a or not b or len(a) != len(b):
        return 1 << 30
    return sum(1 for x, y in zip(a, b) if x != y)


# --------------------------------------------------------------------------
# 记录与日志（每步都要留痕，报告要能追溯）
# --------------------------------------------------------------------------
def log(msg: str) -> None:
    """带时间戳打印，方便 PowerShell 重定向到文件后回看。"""
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def dump_json(obj, path, indent: int = 2) -> None:
    """
    JSON 落盘（UTF-8，中文不转义），供后续脚本与报告复用。

    兼容 numpy 标量：管线里的数值大多来自 numpy（conf、area_ratio、
    assess_interpretability 的 feasible 等），而标准 json 只认 Python 原生类型。
    典型报错是 `TypeError: Object of type bool is not JSON serializable` ——
    那个 "bool" 其实是 numpy.bool_，肉眼很难定位。统一在这里兜底。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"不可序列化类型: {type(o).__name__}")

    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=_default)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs(*dirs) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 数据集自检
# --------------------------------------------------------------------------
def list_images(d) -> list[Path]:
    """
    递归列出目录下的所有图像。

    会跳过隔离区/备份/隐藏目录（见 SKIP_DIR_NAMES、SKIP_DIR_PREFIXES），
    否则去重隔离出来的「重复图」会被下游当成正常样本重新吸回数据集。
    """
    p = Path(d)
    if not p.exists():
        return []
    out: list[Path] = []
    for f in p.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in IMG_EXTS:
            continue
        if is_skipped(f.relative_to(p)):
            continue
        out.append(f)
    return sorted(out)


def is_skipped(rel: Path) -> bool:
    """相对路径中任一目录段属于跳过规则，则整个文件跳过。"""
    for part in rel.parts[:-1]:          # 最后一段是文件名，不看
        if part in SKIP_DIR_NAMES:
            return True
        if part.startswith(SKIP_DIR_PREFIXES):
            return True
    return False


def label_path_for(img_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    """YOLO 布局下，由 images 路径推出对应 labels 路径。"""
    rel = img_path.relative_to(images_dir)
    return labels_dir / rel.with_suffix(".txt")


def read_yolo_labels(txt_path) -> list[tuple[int, float, float, float, float]]:
    """读 YOLO 标签，返回 [(cls, cx, cy, w, h), ...]（归一化坐标）。"""
    out = []
    p = Path(txt_path)
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                c = int(float(parts[0]))
                vals = [float(v) for v in parts[1:5]]
            except ValueError:
                continue
            if any(v < 0 or v > 1 for v in vals):
                continue
            out.append((c, *vals))
    return out


def write_yolo_labels(txt_path, labels) -> None:
    p = Path(txt_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for c, cx, cy, w, h in labels:
            f.write(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def is_tiny_box(w_norm: float, h_norm: float, img_w: int, img_h: int,
                min_px: float = 6.0) -> bool:
    """
    判断是否为「过小框」—— 归一化边长还原成像素后，任一边小于 min_px 即为过小。
    小目标是我们这个课题的核心难点，需要在数据阶段就把分布摸清。
    """
    return (w_norm * img_w) < min_px or (h_norm * img_h) < min_px


# --------------------------------------------------------------------------
# 简易命令行解析（避免依赖 argparse 的样板，够用即可）
# --------------------------------------------------------------------------
def argv_flag(name: str, default=None):
    """
    解析命令行参数。支持两种写法：

      --key=value    →  返回 "value"
      --key          →  返回 "true"（裸 flag，即开关）

    未提供则返回 default。

    为什么必须支持裸 flag（2026-09-20 修正的真 bug）：
      `evaluate.py --all` 原本**静默失效** —— 它只匹配 `--key=value`，
      裸 `--all` 匹配不到，于是 do_all 恒为 False，脚本打印用法后
      **以 exit=0 正常退出**，看起来「跑过了」其实一个模型都没评估。
      这类「参数没被读到但不报错」的坑在本项目已出现第二次
      （前一次是 bench_pipeline.py 的 `--json`），故在此根治。
    """
    prefix = f"--{name}="
    bare = f"--{name}"
    for a in sys.argv[1:]:
        if a.startswith(prefix):
            return a[len(prefix):]
        # 裸 flag：精确等于 --name（排除 --name=value 已在上分支处理）
        if a == bare:
            return "true"
    return default


def resolve_weight(name: str) -> str:
    """
    把权重名解析成「本地可用路径」。

    为什么需要：本机网络环境下 github.com 的 CONNECT 隧道返回 502，
    Ultralytics 默认会去 github release 下载预训练权重并失败。
    所以统一先从 03_weights/ 取本地权重；找不到才回退到权重名（触发官方下载）。

    已归档的本地权重：yolov8n / yolov8s / yolo11n / yolo11s
    （yolo11 系列来自 hf-mirror.com，其余复用基础题已有文件）
    """
    p = Path(name)
    if p.is_absolute() and p.exists():
        return str(p)
    local = WEIGHTS_DIR / p.name
    if local.exists():
        return str(local)
    return name


def write_dataset_yaml(path, train: str, val: str, test: str | None = None,
                       names: dict | None = None, root: str | None = None) -> None:
    """
    写 dataset yaml。

    关键：`path` 必须写**绝对路径**。
    Ultralytics 对 `path: .` 的解析依赖 yaml 所在位置与当前工作目录，
    在非标准目录下会解析到错误位置（本项目已实测：
    它跑去 02_code/images/val 找图，直接 FileNotFoundError）。
    写绝对路径可以彻底规避这个坑。

    Windows 反斜杠在 yaml 里要转成正斜杠，否则会被当作转义字符。
    """
    names = names or {i: c for i, c in enumerate(CLASSES)}
    p = Path(path).resolve()
    if root is None:
        root = p.parent
    root_s = Path(root).resolve().as_posix()

    lines = [
        "# 外墙缺陷筛查数据集配置（由 common.write_dataset_yaml 生成）",
        "# 注意：path 为绝对路径，避免 Ultralytics 在中文/非标准目录下解析失败",
        f"path: {root_s}",
        f"train: {train}",
        f"val: {val}",
    ]
    if test:
        lines.append(f"test: {test}")
    lines.append("")
    lines.append("names:")
    for i in sorted(names):
        lines.append(f"  {i}: {names[i]}")
    lines.append("")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# dataset yaml 可移植性自愈（2026-09-25 新增）
# --------------------------------------------------------------------------
def ensure_dataset_yaml(path=None) -> "Path":
    """保证 dataset yaml 的 `path:` 指向**当前**数据集目录，而不是训练机的旧路径。

    为什么需要：
        write_dataset_yaml 刻意写绝对路径（见该函数 docstring 的实测说明），
        但绝对路径一旦随工程包拷到别的机器／别的盘符就失效。
        工程包里的 yaml 是训练时生成的，`path` 记的是当时的路径；
        评委拷走后 evaluate.py / train.py 直接读它 ⇒ FileNotFoundError。

    做法：
        读 yaml → 取 `path:` 与 `train:`/`val:`/`test:` → 若
        `(path / train)` 不存在，而 `DATASET_DIR / train` 存在，则用当前
        DATASET_DIR 重写 yaml 的 `path:`（其余字段原样保留）。

    返回重写后的 yaml 路径。幂等；yaml 不存在或路径本就正确时不动作。
    """
    import re

    p = Path(path) if path is not None else (DATASET_DIR / DATASET_YAML_NAME)
    if not p.exists():
        return p

    text = p.read_text(encoding="utf-8")
    m = re.search(r"^path:\s*(.+?)\s*$", text, flags=re.M)
    if m is None:
        return p

    old_root = Path(m.group(1).strip().strip('"').strip("'"))
    # 取一个划分名用于探测（优先 train）
    m_split = re.search(r"^train:\s*(.+?)\s*$", text, flags=re.M)
    probe_rel = m_split.group(1).strip().strip('"').strip("'") if m_split else "images/train"

    if (old_root / probe_rel).exists():
        return p                      # 旧路径仍可用，不动
    if not (DATASET_DIR / probe_rel).exists():
        return p                      # 当前路径也没有，别乱改

    new_root = DATASET_DIR.resolve().as_posix()
    text2 = re.sub(r"^path:\s*.+?\s*$", "path: " + new_root,
                   text, count=1, flags=re.M)
    if text2 != text:
        p.write_bytes(text2.replace("\r\n", "\n")
                      .replace("\n", "\r\n").encode("utf-8"))
        print(f"[ensure_dataset_yaml] 已把 {p.name} 的 path 修正为: {new_root}")
    return p
