# -*- coding: utf-8 -*-
"""rt_common.py —— 无人机实时识别实验的公共层。

归属：09_uav_realtime/（独立实验目录）。本文件**不修改、不导入副作用**任何
既有源文件，只**只读复用** 02_code/ 中的既有接口。

职责边界（三件事，别的都不做）：
  1. load_config()  —— 读 rt_config.yaml，校验，并**把 infer.* 归一化成
     pipeline.run_one 真正消费的 args 字典**。这一步是本模块最容易出错、
     也最值得集中的地方：YAML 里用的是人类友好键名（env_class / rectify），
     而 run_one 消费的是旧口径键名（env / rectify），必须在此处做映射。
  2. 绘图工具       —— 中文标签覆盖层（PIL）+ HUD；**检测框本身不重画**，
     直接用 run_one 返回的 annotated（保证样式与主链路逐像素一致）。
  3. LatencyStats   —— 分位统计（P50/P90/P99）。实时性看长尾，均值会骗人。

★ 纪律（源自项目长期踩坑，务必遵守）：
  - 绝不硬编码类别名：一律从 common.CLASSES / CLASS_CN 取（唯一真源）。
  - 中文文本绝不用 cv2.putText（它只支持 ASCII，中文会变 "?"）⇒ 用 PIL。
  - 非 ASCII 路径绝不交给 cv2.imwrite ⇒ 用 common.imwrite_u。
  - 计时后必须 _sync()，否则 GPU 上测到的是「发起调用」而非「算完」。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ 复用既有接口
# 把既有源码目录加进 sys.path：本模块是独立目录，不能假设调用方已把源码目录入路径。
#
# ★ 必须同时兼容「源目录布局」与「交付包布局」，否则包内跑不起来：
#     源目录：  <root>/09_uav_realtime/  +  <root>/02_code/
#     交付包：  <root>/uav_realtime/     +  <root>/code/
#   命名不同是历史原因（交付包把 02_code 重命名为 code），
#   因此这里按「候选目录逐个探测」而不是写死一个名字。
#   —— 这正是 `_pack_vs_src.py` 之外、必须**在包自身目录内跑一次入口**的原因：
#      源目录能跑 ≠ 包里能跑。
_HERE = Path(__file__).resolve().parent
_CODE_CANDIDATES = (_HERE.parent / "02_code", _HERE.parent / "code")
for _cand in _CODE_CANDIDATES:
    if (_cand / "common.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break
else:
    raise SystemExit(
        "找不到既有源码目录（期望 common.py 位于 %s 之一）。\n"
        "请确认工程结构：源目录 = 02_code/，交付包 = code/。"
        % " 或 ".join(str(c) for c in _CODE_CANDIDATES)
    )

from common import (  # noqa: E402
    CLASSES,
    CLASS_CN,
    CLASS_COLORS,
    imwrite_u,
    log,
    default_weight,
    resolve_weight,
)

try:
    import yaml
except ImportError:  # pragma: no cover
    raise SystemExit(
        "缺少 PyYAML。请安装：\n"
        "  D:\\下载\\python.exe -m pip install \"PyYAML>=6.0\" -i https://pypi.org/simple"
    )


# ============================================================================
# 1) 配置加载与 args 归一化
# ============================================================================

# YAML 键 → run_one args 键 的映射表。
# 左列是本配置文件的现代键名，右列是 pipeline.run_one 实际 args.get(...) 的键名。
# 为什么需要它：run_one 的 args 是历史演化的产物（同时存在 env / env_class 两种
# 叫法，jgj125 是字符串控制位而非布尔），直接透传会静默走到默认值 —— 这类
# 「键名不匹配导致参数静默失效」是本工程真实踩过的坑（如 argv_flag 裸 flag 曾失效）。
_ARG_KEY_MAP = {
    "env_class": "env",        # run_one 读的是 args["env"]（中文描述串）
}

# 必填/校验规则：(键路径, 期望类型, 允许范围或 None)
_NUMERIC_CHECKS = [
    ("infer.conf", (int, float), (0.0, 1.0)),
    ("infer.imgsz", (int,), None),
    ("capture.target_fps", (int, float), (0.0, None)),
    ("capture.max_frames", (int,), (0, None)),
    ("capture.duration_sec", (int, float), (0.0, None)),
    ("capture.warmup_frames", (int,), (0, None)),
]


class ConfigError(SystemExit):
    """配置错误。用 SystemExit 是为了让 CLI 直接以非零码退出并打印原因。"""


def _dig(d: dict, path: str, default=None):
    """按 "a.b.c" 取嵌套字典值。"""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_config(path) -> dict:
    """读 YAML 配置并做基本校验。返回原始 dict（未归一化）。"""
    p = Path(path)
    if not p.is_absolute():
        p = _HERE / p
    if not p.exists():
        raise ConfigError(f"[配置错误] 找不到配置文件：{p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ConfigError(f"[配置错误] {p} 解析结果不是字典，检查 YAML 缩进。")

    # --- 数值范围校验 ---
    for key, types, rng in _NUMERIC_CHECKS:
        val = _dig(cfg, key)
        if val is None:
            continue
        if not isinstance(val, types):
            raise ConfigError(
                f"[配置错误] {key} 类型应为 {types}，实际 {type(val).__name__}={val!r}")
        if rng is not None:
            lo, hi = rng
            if lo is not None and val < lo:
                raise ConfigError(f"[配置错误] {key}={val} 小于下限 {lo}")
            if hi is not None and val > hi:
                raise ConfigError(f"[配置错误] {key}={val} 大于上限 {hi}")

    cfg["_config_path"] = str(p)
    return cfg


def resolve_source(cfg: dict, source_key: str | None = None) -> dict:
    """按预设名取出视频源配置，并合上 stream / capture 的公共段。

    返回含 kind / uri / 各自参数 的扁平 dict。
    """
    src = cfg.get("source") or {}
    presets = src.get("presets") or {}
    key = source_key or src.get("preset")
    if not key:
        raise ConfigError("[配置错误] 未指定视频源：请设 source.preset 或用 --source=xxx")
    if key not in presets:
        raise ConfigError(
            f"[配置错误] source.presets 里没有 '{key}'。可用：{sorted(presets.keys())}")

    one = dict(presets[key] or {})
    kind = one.get("kind")
    if kind not in ("file", "stream", "device", "wifi_fpv"):
        raise ConfigError(
            f"[配置错误] preset[{key}].kind='{kind}' 非法，"
            f"只能是 file / stream / device / wifi_fpv")

    uri = one.get("uri")
    # wifi_fpv 源用 port 而非 uri 定位，不做 uri 校验
    if kind == "wifi_fpv":
        if not one.get("port"):
            raise ConfigError(
                f"[配置错误] preset[{key}] 是 wifi_fpv 源但未填 port。\n"
                f"  请先跑 uav_probe.py 实测出无人机推流端口，再填进来。")
        one["key"] = key
        one["stream"] = dict(src.get("stream") or {})
        one["capture"] = dict(cfg.get("capture") or {})
        return one

    # device 的 uri 允许是 "0"（字符串），但 file 必须有真实路径
    if kind == "file":
        if not uri or not str(uri).strip():
            raise ConfigError(
                f"[配置错误] preset[{key}] 是 file 源但 uri 为空。"
                f"请填入视频文件路径（注释里已说明该预设会显式报错而非静默跳过）。")

        # ★ uri 支持「多候选」（用 | 分隔），取第一个真实存在的。
        #   为什么需要：同一份 rt_config.yaml 要能在两种布局下都跑通 ——
        #     源目录：  ../logs/_test_video/xxx.mp4
        #     交付包：  ./_test_video/xxx.mp4
        #   写死单一路径必然在另一种布局下报「文件不存在」。
        cands = [c.strip() for c in str(uri).split("|") if c.strip()]
        tried = []
        fp = None
        for cand in cands:
            p = Path(cand)
            if p.is_absolute():
                if p.exists():
                    fp = p
                    break
                tried.append(str(p))
                continue
            # 相对路径依次按「本目录」与「工程根」解析
            for base in (_HERE, _HERE.parent):
                q = (base / cand).resolve()
                if q.exists():
                    fp = q
                    break
                tried.append(str(q))
            if fp is not None:
                break
        if fp is None:
            raise ConfigError(
                f"[配置错误] preset[{key}] 视频文件不存在。已尝试以下候选：\n  "
                + "\n  ".join(tried))
        one["uri_resolved"] = str(fp)

    one["key"] = key
    one["stream"] = dict(src.get("stream") or {})
    one["capture"] = dict(cfg.get("capture") or {})
    return one


def build_args(cfg: dict) -> dict:
    """把 infer.* 段归一化成 pipeline.run_one 消费的 args 字典。

    ★ 这是本模块的口径关口：只有经此函数产出的 args 才允许传给 run_one。
      直接手搓 args 会导致键名漂移、参数静默失效。
    """
    infer = dict(cfg.get("infer") or {})
    cap = dict(cfg.get("capture") or {})

    # 模型解析：留空则走主链路默认权重（default_weight 里定义了候选优先级）
    model = str(infer.get("model") or "").strip()
    if not model:
        model = default_weight("v11s640")
    else:
        # 允许写裸名（如 v11s640_best.pt），由 resolve_weight 定位
        if not Path(model).exists():
            model = resolve_weight(model)

    args: dict = {
        "conf": float(infer.get("conf", 0.25)),
        "imgsz": int(infer.get("imgsz", 640)),
        "device": str(infer.get("device", "cuda:0")),
        "model": model,
        # 标定（与 pipeline.run_one 同名）
        "distance": float(infer.get("distance", 0.0) or 0.0),
        "focal": float(infer.get("focal", 0.0) or 0.0),
        "calib_object_px": float(infer.get("calib_object_px", 0.0) or 0.0),
        "calib_object_mm": float(infer.get("calib_object_mm", 0.0) or 0.0),
        "calib_object_name": str(infer.get("calib_object_name", "") or ""),
        "calib_brick": bool(infer.get("calib_brick", False)),
        "rectify": bool(infer.get("rectify", False)),
        "rectify_force": False,
        "jgj125": "main-rebar,slab-tension" if infer.get("jgj125", True) else "",
        # 批量/视频场景不落盘单张 annotated（由本模块统一管理落盘）
        "save_annotated": False,
        "out": str(_HERE / "out_frames"),
        # 采样参数给下游可能用到（run_one 本身不读，但保留便于追溯）
        "target_fps": float(cap.get("target_fps", 0.0) or 0.0),
    }

    # --- 应用键名映射（现代名 → run_one 旧名）---
    for new_key, old_key in _ARG_KEY_MAP.items():
        if new_key in infer:
            args[old_key] = infer[new_key]

    # env 兜底：run_one 期望中文描述串
    if "env" not in args or not args["env"]:
        args["env"] = "二类环境（露天/潮湿）"

    # ★ env 合法性校验 —— 把「静默失效」变成「响亮报错」。
    # grade.grade_all 对未知 env_class 不会抛错，而是回落到默认裂缝限值，
    # 于是报告里的合规判定会悄悄基于错误的标准。此处必须拦住。
    _VALID_ENV = (
        "一类环境（室内干燥）",
        "二类环境（露天/潮湿）",
        "三类环境（干湿交替/海风）",
    )
    if args["env"] not in _VALID_ENV:
        raise ConfigError(
            f"[配置错误] infer.env_class='{args['env']}' 不是合法环境类别。\n"
            f"  必须逐字取以下之一：\n    " + "\n    ".join(_VALID_ENV) +
            "\n  （写错不会报错、会静默用错限值，故在此拦截）")

    # ★ 标定链合法性预检 —— 提前拦，不要在跑到中途才崩。
    # 背景：get_calibration 的最后兜底是 calibrate_by_camera(distance, w, focal)，
    # 而它 assert distance>0 且 focal>0。若前面几路标定（标定物/校正/砖缝）都不
    # 生效，就会走到这里抛 ValueError，**在视频跑到一半时把整个实验打断**。
    # 故凡是没提供标定物、也没开砖缝、也没走校正时，distance/focal 必须为正。
    has_primary_calib = (
        args["calib_object_px"] > 0
        or args["calib_brick"]
        or args["rectify"]
    )
    if not has_primary_calib and (args["distance"] <= 0 or args["focal"] <= 0):
        raise ConfigError(
            "[配置错误] 标定参数不合法，运行到中途会崩：\n"
            f"  distance={args['distance']}  focal={args['focal']}\n"
            "  当前既未提供标定物（calib_object_px）、也未开砖缝法（calib_brick）、"
            "也未开正射校正（rectify），\n"
            "  因此会落到相机参数法兜底，而它要求 distance>0 且 focal>0。\n"
            "  ⇒ 请填正数（外墙巡检常用：distance=15~25，focal=24），"
            "或改用标定物法/砖缝法。")

    return args


# ============================================================================
# 2) 绘图工具
# ============================================================================

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\simsun.ttc",   # 宋体
]

_font_cache: dict[int, object] = {}


def _get_font(size: int = 18):
    """取中文字体（带缓存）。找不到任何中文字体时返回 None，调用方需降级。"""
    if size in _font_cache:
        return _font_cache[size]
    try:
        from PIL import ImageFont
    except ImportError:
        _font_cache[size] = None
        return None
    for fp in _FONT_CANDIDATES:
        if Path(fp).exists():
            try:
                f = ImageFont.truetype(fp, size)
                _font_cache[size] = f
                return f
            except Exception:
                continue
    _font_cache[size] = None
    return None


def cn_label(cls_name: str) -> str:
    """类别 → 中文显示名。缺项时回退英文名，绝不抛异常。"""
    return CLASS_CN.get(cls_name, cls_name)


def draw_cn_overlay(bgr: np.ndarray, texts: list[tuple[str, tuple[int, int, int]]],
                    origin=(8, 8), size: int = 18) -> np.ndarray:
    """在 BGR 图上用 PIL 绘制中文文本行（解决 cv2.putText 不支持中文）。

    texts  : [(文本, BGR颜色), ...]，逐行从上往下。
    origin : 左上角起始坐标。
    返回新图（不改原图）。
    """
    if not texts:
        return bgr
    font = _get_font(size)
    if font is None:
        # 降级：用 cv2 画英文/数字（中文会缺字，但至少不崩）
        import cv2
        out = bgr.copy()
        x0, y0 = origin
        for i, (t, col) in enumerate(texts):
            cv2.putText(out, t, (x0, y0 + (i + 1) * (size + 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        return out

    from PIL import Image, ImageDraw
    out = bgr.copy()
    img = Image.fromarray(out[:, :, ::-1])   # BGR → RGB
    draw = ImageDraw.Draw(img)
    x0, y0 = origin
    line_h = size + 8
    for i, (t, col) in enumerate(texts):
        rgb = (col[2], col[1], col[0])       # BGR → RGB
        y = y0 + i * line_h
        # 描边 + 填充，保证在任何底色上可读
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((x0 + dx, y + dy), str(t), font=font, fill=(0, 0, 0))
        draw.text((x0, y), str(t), font=font, fill=rgb)
    return np.array(img)[:, :, ::-1].copy()  # RGB → BGR


# 与 pipeline.py 完全相同的 7 类色板（BGR）。★ 新增类别必须同步这里，
# 否则 .get 会静默回退成裂缝红 —— 6→7 类扩容时真实漏过一次。


def class_color(cls_name: str):
    return CLASS_COLORS.get(cls_name, (0, 0, 255))


def draw_det_labels(bgr: np.ndarray, measurements: list[dict],
                    size: int = 16) -> np.ndarray:
    """在每个检测框上方画「中文类名 + 置信度」。

    注意：框本身由 draw_boxes() 用 measure.draw_measurement 画出（与主链路
    同一函数，样式一致）；本函数只补中文标签层。

    measurements 里每项需含 cls_name / bbox_xyxy / conf（conf 是 run_one
    显式 merge 进去的运行时字段，见 pipeline.py 第 355 行）。
    """
    if not measurements:
        return bgr
    from PIL import Image, ImageDraw
    font = _get_font(size)
    out = bgr.copy()

    if font is None:
        import cv2
        for m in measurements:
            x1, y1 = int(m["bbox_xyxy"][0]), int(m["bbox_xyxy"][1])
            cv2.putText(out, f"{m['cls_name']} {m.get('conf', 0):.2f}",
                        (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, class_color(m["cls_name"]), 1, cv2.LINE_AA)
        return out

    img = Image.fromarray(out[:, :, ::-1])
    draw = ImageDraw.Draw(img)
    W, H = img.size
    placed: list[tuple[int, int, int, int]] = []   # 已占用的标签矩形，用于避让

    def _overlaps(rect) -> bool:
        x1, y1, x2, y2 = rect
        for a1, b1, a2, b2 in placed:
            if not (x2 < a1 or x1 > a2 or y2 < b1 or y1 > b2):
                return True
        return False

    for m in measurements:
        bbox = m["bbox_xyxy"]
        x1, y1 = int(bbox[0]), int(bbox[1])
        name = cn_label(m["cls_name"])
        conf = float(m.get("conf", 0.0) or 0.0)
        txt = f"{name} {conf:.2f}"
        try:
            l, t, r, b = draw.textbbox((0, 0), txt, font=font)
            tw, th = r - l, b - t
        except Exception:
            tw, th = len(txt) * size, size + 4
        tw += 6
        th += 4

        # 水平避让：默认贴框左上；超出右边界则右对齐到画面内
        tx = x1
        if tx + tw > W:
            tx = max(W - tw - 1, 0)
        # 垂直避让：默认放框上方；与已有标签重叠则上移，顶到边界则改放框内顶部
        ty = max(y1 - th - 2, 0)
        tries = 0
        while _overlaps((tx, ty, tx + tw, ty + th)) and tries < 8:
            ty -= th + 2
            if ty < 0:
                ty = min(y1 + 2, H - th - 1)
                break
            tries += 1
        ty = max(0, min(ty, H - th - 1))

        col = class_color(m["cls_name"])
        draw.rectangle([tx, ty, tx + tw, ty + th], fill=(0, 0, 0))
        draw.text((tx + 3, ty + 1), txt, font=font, fill=(col[2], col[1], col[0]))
        placed.append((tx, ty, tx + tw, ty + th))
    return np.array(img)[:, :, ::-1].copy()


def draw_boxes(bgr: np.ndarray, measurements: list[dict],
               grades: list[dict] | None = None) -> np.ndarray:
    """画检测框 + 度量标注。

    ★ 为什么本层要自己画框，而不是用 run_one 返回的 `annotated`：
      run_one 的 `annotated` 字段是**文件路径字符串**（`str | None`），
      不是图像数组 —— 它内部画好 `vis` 后只在 `--save-annotated` 打开时
      落盘并返回路径，图像本身并不返回。实测确认（pipeline.py:317-326、362）。
      若误当数组使用，会静默拿到 None 并回退成原帧 —— **框完全不显示**，
      而且不报错。这一点已在本模块首轮实测中真实触发过。
      ⇒ 正确做法：本层用同一个 measure.draw_measurement 重画，
        保证与主链路样式逐像素一致，且不需要逐帧写盘读盘。

    grades 用于判定 severity：命中 danger 的框强制画红（与主链路同规则）。
    """
    if not measurements:
        return bgr
    from measure import draw_measurement  # 延迟导入：避免模块级副作用
    out = bgr
    for m in measurements:
        cls_name = m.get("cls_name", "")
        col = class_color(cls_name)
        if grades:
            for g in grades:
                if tuple(g.get("bbox_xyxy", ())) == tuple(m.get("bbox_xyxy", ())):
                    if g.get("severity") == "danger":
                        col = (0, 0, 255)
                    break
        out = draw_measurement(out, _DictMeasurement(m), col)
    return out


class _DictMeasurement:
    """把 run_one 返回的 measurement dict 适配成 draw_measurement 需要的对象。

    draw_measurement 只读取属性（不要的是 dataclass 类型本身），
    故用轻量适配器即可，避免反向 import measure.Measurement 造成耦合。
    """

    __slots__ = ("cls_name", "bbox_xyxy", "length_mm", "width_max_mm",
                 "equiv_diameter_px", "area_m2")

    def __init__(self, d: dict):
        self.cls_name = d.get("cls_name", "")
        self.bbox_xyxy = tuple(int(round(v)) for v in d.get("bbox_xyxy", (0, 0, 0, 0)))
        self.length_mm = float(d.get("length_mm", 0.0) or 0.0)
        self.width_max_mm = float(d.get("width_max_mm", 0.0) or 0.0)
        self.equiv_diameter_px = float(d.get("equiv_diameter_px", 0.0) or 0.0)
        self.area_m2 = float(d.get("area_m2", 0.0) or 0.0)


def draw_banner(bgr: np.ndarray, calib: dict, risk: dict,
                interp: dict | None = None) -> np.ndarray:
    """左上角信息条：GSD / 可判读性 / 风险等级。

    用 PIL 画以支持中文（主链路用 cv2.putText 只能画 ASCII）。
    """
    mm = calib.get("mm_per_px")
    lines = []
    if mm is not None:
        lines.append(f"GSD {mm:.3f} mm/px ({calib.get('method')})")
    if interp:
        c = interp.get("crack_0.3mm") or {}
        b = interp.get("blob_10mm") or {}
        if c.get("pixels_on_target") is not None:
            lines.append(f"裂缝0.3mm {c['pixels_on_target']:.1f}px "
                         f"[{str(c.get('level','')).upper()}]")
        if b.get("pixels_on_target") is not None:
            lines.append(f"块状10mm {b['pixels_on_target']:.1f}px "
                         f"[{str(b.get('level','')).upper()}]")
    if risk.get("level"):
        lines.append(f"风险等级 {risk['level']}")
    if not lines:
        return bgr
    return draw_cn_overlay(bgr, [(t, (0, 255, 255)) for t in lines],
                           origin=(8, 8), size=18)


def draw_hud(bgr: np.ndarray, lines: list[str], size: int = 18,
             origin=(8, 8)) -> np.ndarray:
    """运行时指标 HUD（帧号 / 推理耗时 / 延迟 / 检出数）。

    origin 由调用方给出，便于与信息条错开。
    """
    if not lines:
        return bgr
    texts = [(t, (0, 255, 255)) for t in lines]
    return draw_cn_overlay(bgr, texts, origin=origin, size=size)


# ============================================================================
# 3) 分位统计
# ============================================================================

class LatencyStats:
    """一组延迟样本的分位统计。

    为什么不用 statistics.mean：实时性的成败由长尾决定。一个 P99=800ms 的
    系统，均值可能仍是 120ms 却完全不可用 —— 均值在这里是误导性指标。
    """

    def __init__(self, percentiles=(50, 90, 99)):
        self.percentiles = list(percentiles)
        self.samples: list[float] = []

    def add(self, ms: float) -> None:
        self.samples.append(float(ms))

    def __len__(self) -> int:
        return len(self.samples)

    def summary(self) -> dict:
        if not self.samples:
            return {"n": 0}
        a = np.asarray(self.samples, dtype=float)
        out = {
            "n": int(a.size),
            "mean_ms": round(float(a.mean()), 3),
            "min_ms": round(float(a.min()), 3),
            "max_ms": round(float(a.max()), 3),
            "std_ms": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 3),
        }
        for p in self.percentiles:
            out[f"p{p}_ms"] = round(float(np.percentile(a, p)), 3)
        return out


def fps_of(ms: float) -> float:
    """单帧耗时（ms）→ 等价 FPS。"""
    return 1000.0 / ms if ms > 0 else 0.0


# ============================================================================
# 4) 计时（照搬 bench_pipeline.py 的范式，保持口径一致）
# ============================================================================

def sync_device(device: str) -> None:
    """GPU 计时必须同步，否则测到的是「发起调用」的时间，不是「算完」的时间。"""
    if str(device).lower().startswith("cuda") or str(device) == "0":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass


class Seg:
    """with Seg() as s: ...  →  s.ms 为毫秒耗时（退出时已 sync）。"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.ms = 0.0

    def __enter__(self):
        sync_device(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync_device(self.device)
        self.ms = (time.perf_counter() - self.t0) * 1000.0
        return False


def save_frame(path, img) -> bool:
    """中文路径安全写图。"""
    return imwrite_u(path, img)


__all__ = [
    "ConfigError", "load_config", "resolve_source", "build_args",
    "cn_label", "draw_cn_overlay", "draw_det_labels", "draw_boxes",
    "draw_banner", "draw_hud",
    "CLASS_COLORS", "class_color",
    "LatencyStats", "fps_of", "sync_device", "Seg", "save_frame",
    "CLASSES", "CLASS_CN", "log",
]
