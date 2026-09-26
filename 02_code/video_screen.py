# -*- coding: utf-8 -*-
r"""
video_screen.py —— 视频输入筛查：把「一段外墙视频」跑成「一份筛查结论」

设计目标
--------
现有系统的输入是**单张图**（`pipeline.run_one`）。现场实际更常见的是
**拿手机绕着墙走一圈录一段**。本模块让系统接受视频输入，而不改单图链路。

核心思路：**加一层抽帧 + 时间聚合，复用现有单图管线**
    视频 → 抽帧（按目标间隔）→ 每帧调用 pipeline.run_one
         → 跨帧聚合（同一缺陷多帧合并）→ 输出结构化结论

★ 一条必须写进输出、不许省略的诚实约束
----------------------------------------
**视频不能提高毫米级判读能力。**
判读门槛由 GSD（每个像素代表多少毫米）决定，而 GSD 只取决于
「镜头的物理分辨率 + 拍摄距离」—— 一帧 1080p 的视频帧，和一张 1080p 的照片，
判读能力**完全相同**。

所以本模块在报告里始终标注：
    · 视频的价值 = **覆盖面**（一圈墙 vs 一张照片）与**多视角冗余**（减少漏检）
    · 视频**不改变** 0.3mm 裂缝能不能判读 —— 那仍然由 GSD 决定
这一点如果不说清，就是把「视频」当成营销词，而不是工程事实。

时间聚合为什么必要（实测依据）
------------------------------
见 `logs/_VIDEO_TEMPORAL_PROBE.md`：
    · 同一处缺陷在不同帧上的检测框尺寸必然抖动（压缩噪声、运动模糊、框抖动）
    · 若**随便抽一帧**就报毫米数，等于报一个随机数 ——
      会把系统最核心的「可判读性自检」作废
    · 故本模块对同一缺陷的**跨帧测量取中位数**，并记录帧间离散度；
      离散度过大时，把该缺陷标为「帧间不一致，不建议单独采信」。

用法
----
    # 基本：抽帧 + 检测 + 聚合
    python video_screen.py --video=wall_walk.mp4 --model=best.pt --calib-brick

    # 指定抽帧间隔（秒）与最大帧数
    python video_screen.py --video=wall_walk.mp4 --every=0.5 --max-frames=60

    # 与单图一致的全部标定参数都可用（--calib-object-px / --distance / --rectify）
"""
from __future__ import annotations

import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from common import (
    EVAL_DIR,
    RESULT_DIR,
    argv_flag,
    default_weight,
    dump_json,
    ensure_dirs,
    log,
    resolve_weight,
)

# 视频容器扩展名（供 list_videos / 上层判断复用）
VIDEO_EXTS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}

# 跨帧「同一缺陷」的判定门限（相对画面宽度的比例，避免分辨率敏感）
MATCH_CENTER_TOL_RATIO = 0.06      # 中心点距离 < 画面宽 6%
MATCH_IOU_TOL = 0.20               # 或 IoU > 0.20
# 帧间离散度门限：变异系数超过它 ⇒ 标注「帧间不一致」
CV_INCONSISTENT = 0.35


@dataclass
class FrameResult:
    """单帧的处理结果（薄包装，保留 pipeline.run_one 的原始 dict）。"""
    idx: int                # 抽取序号
    frame_no: int           # 视频里的真实帧号
    t_sec: float            # 时间戳（秒）
    raw: dict               # pipeline.run_one 的返回
    status: str = "ok"


@dataclass
class Track:
    """跨帧聚合出的一个「缺陷轨迹」——代表视频里的一处缺陷。"""
    cls: str
    n_frames: int = 0
    confs: list[float] = field(default_factory=list)
    widths_mm: list[float] = field(default_factory=list)
    lengths_mm: list[float] = field(default_factory=list)
    areas_mm2: list[float] = field(default_factory=list)
    severities: list[str] = field(default_factory=list)
    # 代表 bbox（取置信度最高那一帧的）
    best_bbox: list[int] = field(default_factory=list)
    best_conf: float = 0.0
    best_frame: int = -1

    # ---- 聚合 ----
    def add(self, m: dict, frame_no: int) -> None:
        self.n_frames += 1
        c = float(m.get("conf", 0.0))
        self.confs.append(c)
        self.widths_mm.append(float(m.get("width_mean_mm", 0.0)))
        self.lengths_mm.append(float(m.get("length_mm", 0.0)))
        self.areas_mm2.append(float(m.get("area_mm2", 0.0)))
        if c >= self.best_conf:
            self.best_conf = c
            self.best_bbox = list(m.get("bbox_xyxy", []))
            self.best_frame = frame_no

    def finalize(self, grades_by_bbox: dict | None = None) -> dict:
        """产出聚合结论：中位数 + 离散度 + 一致性判定。"""
        def med(xs: list[float]) -> float:
            xs = [x for x in xs if x == x and abs(x) < 1e12]   # 去 NaN/inf
            return round(statistics.fmean(xs), 3) if xs else 0.0

        def cv(xs: list[float]) -> float:
            xs = [x for x in xs if x == x and abs(x) < 1e12]
            if len(xs) < 2:
                return 0.0
            m = statistics.fmean(xs)
            return round(statistics.pstdev(xs) / m * 100, 1) if m else 0.0

        w_cv = cv(self.widths_mm)
        out = {
            "cls": self.cls,
            "n_frames": self.n_frames,
            "conf_median": med(self.confs),
            "conf_best": round(self.best_conf, 4),
            # ★ 跨帧聚合值（中位数）—— 这是推荐的采信值
            "width_mean_mm_median": med(self.widths_mm),
            "length_mm_median": med(self.lengths_mm),
            "area_mm2_median": med(self.areas_mm2),
            # 离散度（衡量「视频里这处缺陷测得稳不稳」）
            "width_cv_pct": w_cv,
            "width_range_mm": round(
                (max(self.widths_mm) - min(self.widths_mm)), 3
            ) if len(self.widths_mm) >= 2 else 0.0,
            # ⚠️ 量纲必须对齐：w_cv 是**百分数**（如 4.6 表示 4.6%），
            #   而 CV_INCONSISTENT 是**小数**（0.35）。直接比较会把
            #   所有轨迹都误判为「不一致」（4.6 <= 0.35 恒 False）。
            #   本处曾漏乘 100，已由单测发现并修正（2026-09-25）。
            "consistent": bool(w_cv <= CV_INCONSISTENT * 100),
            "representative_bbox": self.best_bbox,
            "representative_frame": self.best_frame,
        }
        if not out["consistent"] and self.n_frames >= 3:
            out["note"] = (
                f"帧间测量离散度 CV={w_cv:.1f}% 超过 {CV_INCONSISTENT*100:.0f}% 阈值"
                f" ⇒ 该缺陷在不同帧上的尺寸差异较大，**不建议单独采信单帧值**，"
                f"已给出逐帧中位数。"
            )
        return out


# --------------------------------------------------------------------------
def list_videos(d) -> list[Path]:
    """列出目录下所有视频文件（供上层与 --dir 模式复用）。"""
    p = Path(d)
    if not p.exists():
        return []
    return sorted(f for f in p.rglob("*")
                  if f.is_file() and f.suffix.lower() in VIDEO_EXTS)


def probe_video(path: Path) -> dict:
    """
    读取视频元信息。**打不开就如实返回 ok=False**，不抛异常。

    为什么单独抽出来：上层要先知道「这段视频能不能处理」，
    才能在抽帧前给出明确错误（而不是抽到一半失败）。
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return {"ok": False, "reason": "OpenCV 无法打开该视频（缺少解码后端或文件损坏）"}
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc_i = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        cc = "".join(chr((fourcc_i >> (8 * i)) & 0xFF) for i in range(4))
    finally:
        cap.release()
    return {
        "ok": bool(fps > 0 and n > 0),
        "fps": round(fps, 3),
        "n_frames": n,
        "width": w, "height": h,
        "fourcc": cc,
        "duration_sec": round(n / fps, 2) if fps else 0.0,
        "reason": "" if (fps > 0 and n > 0) else "元信息异常（fps 或帧数为 0）",
    }


def plan_sampling(meta: dict, every_sec: float, max_frames: int) -> list[tuple[int, float]]:
    """
    规划抽帧点：返回 [(frame_no, t_sec), ...]

    策略：**按时间等间隔**（不是按帧号），因为不同视频帧率不同，
    按帧号抽会让「抽帧密度」随帧率漂移。等时间间隔的语义更稳定：
    「每 every_sec 秒看一帧」。

    末尾保证**至少覆盖到视频结尾附近**，避免整段后半没看。
    """
    fps = float(meta.get("fps") or 0)
    n = int(meta.get("n_frames") or 0)
    if fps <= 0 or n <= 0:
        return []
    dur = n / fps
    step = max(1, int(round(every_sec * fps)))
    pts: list[tuple[int, float]] = []
    f = 0
    while f < n and len(pts) < max_frames:
        pts.append((f, round(f / fps, 3)))
        f += step
    # 补一帧接近结尾的（若与最后一帧间隔较远）
    last_t = pts[-1][1] if pts else 0.0
    if n - 1 > 0 and (dur - last_t) > every_sec * 0.6 and len(pts) < max_frames:
        pts.append((n - 1, round((n - 1) / fps, 3)))
    return pts


def _center(b: list) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _iou(a: list, b: list) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def aggregate_tracks(frames: list[FrameResult], width_px: int) -> list[dict]:
    """
    把逐帧检测结果聚合成「跨帧缺陷轨迹」。

    匹配规则（两条任一满足即认为同一缺陷）：
      · 中心点距离 < 画面宽 × MATCH_CENTER_TOL_RATIO
      · IoU > MATCH_IOU_TOL
      且**类别相同**（跨类不合并——这比「按位置合并」保守，
      避免把紧邻的裂缝与露筋合成一个）。

    ⚠️ 这是**贪心匹配**，不是 SORT/DeepSORT 那类真跟踪器。
    为什么不引入真跟踪器：本项目的目的不是「稳定追踪每个目标」，
    而是「同一处缺陷在多帧上都量一次，看它稳不稳」。贪心匹配对这种
    「慢速平移拍摄」场景够用，且零额外依赖。若将来做高通量视频，
    再换真跟踪器（届时本函数的输入输出契约不变）。
    """
    tracks: list[Track] = []
    tol = max(12.0, width_px * MATCH_CENTER_TOL_RATIO)
    # 先按帧序、再按置信度降序，让高置信度的框优先成为轨迹代表
    for fr in frames:
        if fr.status != "ok":
            continue
        for m in fr.raw.get("measurements", []) or []:
            bb = m.get("bbox_xyxy")
            if not bb or len(bb) < 4:
                continue
            cls = m.get("cls_name", "")
            cx, cy = _center(bb)
            hit = None
            for t in tracks:
                if t.cls != cls:
                    continue
                if not t.best_bbox:
                    continue
                tx, ty = _center(t.best_bbox)
                d = math.hypot(cx - tx, cy - ty)
                if d < tol or _iou(bb, t.best_bbox) > MATCH_IOU_TOL:
                    hit = t
                    break
            if hit is None:
                hit = Track(cls=cls)
                tracks.append(hit)
            hit.add(m, fr.frame_no)
    out = [t.finalize() for t in tracks]
    out.sort(key=lambda x: (-x["n_frames"], -x["conf_best"]))
    return out


def summarize(frames: list[FrameResult], tracks: list[dict],
              meta: dict, logic: dict) -> dict:
    """产生整段视频的筛查结论。"""
    ok_frames = [f for f in frames if f.status == "ok"]
    n_ok = len(ok_frames)

    # ---- 标定稳定性诊断（★ 本次新增，来自实测发现的真问题）----
    #
    # 实测（logs/_test_video/video_screen_out.json）：同一段 1.33 秒、
    # 同一面墙的视频，逐帧独立做「砖缝周期标定」得到的 GSD 是
    #     21.28 / 13.03 / 12.34 / 22.27 mm/px
    # —— 极差达 **80%**。原因是砖缝检测对画面平移/裁剪敏感，
    # 而视频帧间必然存在轻微位移。
    #
    # 后果很严重：GSD 直接决定毫米数。若逐帧各用自己的 GSD，
    # 同一处缺陷在不同帧上会被量成差 80% 的宽度，而「跨帧取中位数」
    # 会把这种系统性偏差平均掉，产生一个**看似稳定、实则无意义**的数字。
    #
    # 正确处理：**一段视频属于同一次拍摄 ⇒ 应当只有一个 GSD**。
    # 故本函数汇总各帧 GSD，给出「主 GSD」，并把逐帧抖动如实报出。
    # ⚠️ 本函数**只做诊断与报告，不改各帧已算出的测量值**（纪律：
    #    不改动已上报的数字）。若要按统一 GSD 重算，需显式重跑。
    gsds = []
    for f in ok_frames:
        c = f.raw.get("calibration") or {}
        v = c.get("mm_per_px")
        if isinstance(v, (int, float)) and v > 0:
            gsds.append((f.frame_no, float(v), c.get("method")))
    calib_stability = {"n_frames_with_gsd": len(gsds)}
    if len(gsds) >= 2:
        vals = [g[1] for g in gsds]
        gm = statistics.fmean(vals)
        spread = (max(vals) - min(vals)) / gm * 100 if gm else 0.0
        calib_stability.update({
            "gsd_values": [round(v, 4) for v in vals],
            "gsd_median": round(statistics.median(vals), 4),
            "gsd_min": round(min(vals), 4),
            "gsd_max": round(max(vals), 4),
            "gsd_rel_spread_pct": round(spread, 1),
            "methods": sorted({g[2] for g in gsds if g[2]}),
            "stable": bool(spread <= 20.0),
        })
        if spread > 20.0:
            calib_stability["warning"] = (
                f"逐帧 GSD 极差达 {spread:.1f}%（超过 20%）⇒ **逐帧标定不稳定**。"
                f"同一段视频本应只有一个 GSD（同一次拍摄、同一距离）。"
                f"此现象说明逐帧独立标定（尤其是砖缝周期法）受帧间位移影响。"
                f"**建议以 gsd_median 作为本段视频的统一尺度重算**，"
                f"或改用「标定物」这种与画面内容无关的标定方式。"
            )
    elif len(gsds) == 1:
        calib_stability.update({"gsd_median": round(gsds[0][1], 4),
                                "stable": None,
                                "note": "仅 1 帧有标定结果，无法评估帧间稳定性"})

    # 选取「代表帧」：检出最多、且达到毫米级报告门槛的那一帧
    best = None
    for f in ok_frames:
        it = (f.raw.get("interpretability") or {})
        can_mm = bool(it.get("can_report_mm"))
        score = (1 if can_mm else 0) * 10000 + int(f.raw.get("n_detections", 0))
        if best is None or score > best[0]:
            best = (score, f)

    # 风险等级：取所有帧里最严重的
    order = {"A": 0, "B": 1, "C": 2, "D": 3, "U": 4}
    risks = [f.raw.get("risk", {}) for f in ok_frames]
    worst = None
    for r in risks:
        lv = r.get("level", "A")
        if worst is None or order.get(lv, 9) > order.get(worst.get("level", "A"), 9):
            worst = r

    # 跨帧「稳定检出」的缺陷：至少在 2 帧出现
    stable = [t for t in tracks if t["n_frames"] >= 2]
    inconsistent = [t for t in stable if not t.get("consistent", True)]

    per_cls: dict[str, int] = {}
    for t in tracks:
        per_cls[t["cls"]] = per_cls.get(t["cls"], 0) + 1

    return {
        "video": meta,
        "sampling": {
            "n_sampled": len(frames),
            "n_ok": n_ok,
            "n_failed": len(frames) - n_ok,
        },
        "logic": logic,
        "calibration_stability": calib_stability,
        "n_defect_tracks": len(tracks),
        "n_tracks_multi_frame": len(stable),
        "defects_per_class": per_cls,
        "worst_frame_risk": worst or {},
        "representative_frame": {
            "frame_no": best[1].frame_no if best else None,
            "t_sec": best[1].t_sec if best else None,
        },
        "tracks": tracks,
        "inconsistent_tracks": [t["representative_bbox"] for t in inconsistent],
        # ★★ 诚实约束：写进 JSON，任何下游都无法忽略
        "capability_note": (
            "**视频不提高毫米级判读能力。** 判读门槛由 GSD 决定，"
            "而 GSD 只取决于镜头物理分辨率与拍摄距离 —— 一帧 1080p 视频帧"
            "与一张 1080p 照片的判读能力相同。视频的价值在于覆盖面"
            "（一圈墙 vs 一张照片）与多视角冗余（减少漏检），"
            "不改变 0.3mm 裂缝能否判读。逐缺陷的毫米值为**跨帧中位数**，"
            "单帧值不单独采信（见 width_cv_pct）。"
            "★ 另需注意：**逐帧独立标定不稳定**（实测极差可达 80%），"
            "因为砖缝周期等标定方式对帧间位移敏感。"
            "同一段视频应只有一个 GSD，见 calibration_stability。"
        ),
    }


# --------------------------------------------------------------------------
def run_video(video: Path, model, args: dict, run_one) -> dict:
    """
    处理单段视频，返回结构化结果。

    参数
    ----
    run_one : 单图处理函数（默认 pipeline.run_one）。**显式传入而非 import**，
        便于单测时替换为假函数，也避免本模块与 pipeline 形成循环依赖。
    """
    meta = probe_video(video)
    meta["file"] = str(video)
    meta["name"] = video.name

    if not meta.get("ok"):
        log(f"  !! 无法处理：{meta.get('reason')}")
        return {"video": meta, "status": "probe_failed",
                "reason": meta.get("reason"), "frames": [], "tracks": []}

    every = float(args.get("every_sec", 1.0))
    max_frames = int(args.get("max_frames", 40))
    pts = plan_sampling(meta, every, max_frames)
    log(f"  抽帧：{len(pts)} 帧（每 {every}s，上限 {max_frames} 帧）"
        f"  视频 {meta['width']}x{meta['height']} {meta['fps']}fps "
        f"{meta['duration_sec']}s")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"video": meta, "status": "open_failed",
                "frames": [], "tracks": []}

    out_dir = Path(args.get("out", str(RESULT_DIR / "vis" / f"video_{video.stem}")))
    ensure_dirs(out_dir)

    frames: list[FrameResult] = []
    try:
        for i, (fno, t) in enumerate(pts, 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ok, fr = cap.read()
            if not ok or fr is None:
                frames.append(FrameResult(i, fno, t, {"status": "read_failed"},
                                          status="read_failed"))
                continue
            # ---- 不再逐帧「写盘 → 读盘」（2026-09-26 改）----
            # 原写法 `cv2.imwrite(str(tmp), fr)` 有两个问题：
            #   ① 纯 I/O 浪费：写一张、run_one 再读回来，只为适配「路径进」契约；
            #   ② **不检查返回值**：实测 cv2.imwrite 对含非 ASCII 的路径返回 False
            #      （静默失败）—— 它之所以能work，只是因为本进程惰性导入过
            #      ultralytics，而 ultralytics 会给 cv2.imwrite 打 unicode 补丁。
            #      ⇒ 依赖副作用，导入顺序一变就每帧静默失败并被记成「帧读不出来」。
            # 改为**内存内**走一次等价的 JPEG 往返：实测 imencode 的字节与 imwrite
            # 逐字节相同、解码像素逐位相同（logs/_jpeg_equiv.txt）
            # ⇒ **输入像素一位不变**，只是不再落盘。
            tmp = out_dir / f"_frame_{fno:06d}.jpg"   # 仅作结果里的逻辑标识名
            ok_e, buf = cv2.imencode(".jpg", fr)
            if not ok_e:
                # 显式失败，不静默：编解码出错不能被误记成「帧读不出来」
                frames.append(FrameResult(
                    i, fno, t,
                    {"status": "encode_failed", "image": str(tmp),
                     "detail": "cv2.imencode('.jpg') 返回 False"},
                    status="encode_failed"))
                continue
            sub = dict(args)
            # 帧图单独落到 out_dir，避免污染主 vis 目录
            sub["out"] = str(out_dir)
            rec = run_one(tmp, model, sub,
                          image=cv2.imdecode(buf, cv2.IMREAD_COLOR))
            frames.append(FrameResult(i, fno, t, rec,
                                      status=rec.get("status", "unknown")))
            if i % 5 == 0 or i == len(pts):
                log(f"    [{i}/{len(pts)}] 帧{fno} 检出"
                    f"{rec.get('n_detections', 0)} 处")
    finally:
        cap.release()

    # 代表帧的宽度用于匹配容差（各帧同源，取元信息里的宽度）
    tracks = aggregate_tracks(frames, meta.get("width") or 0)
    logic = {
        "match_center_tol_ratio": MATCH_CENTER_TOL_RATIO,
        "match_iou_tol": MATCH_IOU_TOL,
        "cv_inconsistent_threshold_pct": CV_INCONSISTENT * 100,
        "aggregation": "median_across_frames",
        "tracking": "greedy_center_or_iou（非 SORT/DeepSORT）",
    }
    summary = summarize(frames, tracks, meta, logic)

    keep = [
        {"idx": f.idx, "frame_no": f.frame_no, "t_sec": f.t_sec,
         "status": f.status,
         "n_detections": f.raw.get("n_detections", 0),
         "calibration": f.raw.get("calibration"),
         "risk": f.raw.get("risk"),
         "interpretability": f.raw.get("interpretability"),
         "annotated": f.raw.get("annotated")}
        for f in frames
    ]

    log(f"  聚合：{len(tracks)} 处缺陷轨迹，其中 {summary['n_tracks_multi_frame']} "
        f"处跨多帧稳定出现")
    if summary["inconsistent_tracks"]:
        log(f"  ⚠️ {len(summary['inconsistent_tracks'])} 处缺陷帧间离散度偏大，"
            f"已在输出中标注为「不建议单独采信单帧值」")
    # 标定稳定性诊断（实测发现逐帧标定会有大抖动，必须显式告警）
    cs = summary.get("calibration_stability") or {}
    if cs.get("gsd_values"):
        log(f"  逐帧 GSD: {cs['gsd_values']}")
        log(f"  中位 GSD={cs.get('gsd_median')} mm/px  "
            f"极差={cs.get('gsd_rel_spread_pct')}%")
        if not cs.get("stable", True):
            log(f"  ⚠️ {cs.get('warning', '')}")

    return {
        "video": meta,
        "status": "ok",
        "args": {k: v for k, v in args.items()
                 if k in ("conf", "imgsz", "every_sec", "max_frames",
                          "distance", "focal", "calib_object_px",
                          "calib_brick", "rectify")},
        "summary": summary,
        "frames": keep,
        "tracks": tracks,
    }


def main() -> None:
    video = argv_flag("video")
    folder = argv_flag("dir")
    save_json = argv_flag("json", "")
    model_path = argv_flag("model", default_weight("v11s640"))

    args = {
        "conf": argv_flag("conf", "0.25"),
        "imgsz": argv_flag("imgsz", "640"),
        "distance": argv_flag("distance", "20"),
        "focal": argv_flag("focal", "24"),
        "calib_object_px": argv_flag("calib-object-px"),
        "calib_object_mm": argv_flag("calib-object-mm", "210"),
        "calib_object_name": argv_flag("calib-object-name", "A4短边210mm"),
        "calib_brick": argv_flag("calib-brick") is not None,
        "brick_pitch_mm": argv_flag("brick-pitch-mm", "250"),
        "env": argv_flag("env", "二类环境（露天/潮湿）"),
        "jgj125": argv_flag("jgj125", ""),
        "every_sec": float(argv_flag("every", "1.0")),
        "max_frames": int(argv_flag("max-frames", "40")),
    }
    # 布尔开关（与 pipeline._flag 同语义）
    args["rectify"] = ("--rectify" in sys.argv[1:]
                       or str(argv_flag("rectify", "")).strip().lower()
                       not in ("", "0", "none", "false"))

    resolved = resolve_weight(model_path)
    log(f"模型: {resolved}")
    if not Path(resolved).exists():
        log("!! 模型不存在。请先训练（train.py）或用 --model 指定权重")
        return

    # 延迟导入，避免本模块被 import 时就去加载 ultralytics
    from ultralytics import YOLO
    from pipeline import run_one

    model = YOLO(resolved)
    # ★ device 显式可配：训练与视频筛查若同时在跑，会争抢同一块 GPU。
    #   传 --device=cpu 可让视频筛查走 CPU（慢但互不干扰）。
    #   run_one 内部调用 model.predict(img, ...) 不传 device，
    #   故此处把 device 挂到模型上（ultralytics 的 YOLO 支持
    #   通过 model.to(device) 或 args 指定；这里用 override 方式最稳）。
    dev = str(argv_flag("device", "") or "").strip()
    if dev:
        try:
            model.overrides["device"] = dev
            log(f"推理设备: {dev}")
        except Exception as e:      # noqa: BLE001
            log(f"!! 设置 device={dev} 失败，沿用默认: {e}")

    if video:
        targets = [Path(video)]
    elif folder:
        targets = list_videos(folder)
        if not targets:
            log(f"目录 {folder} 下没有视频文件（支持 {'/'.join(sorted(VIDEO_EXTS))}）")
            return
    else:
        log("请指定 --video=<文件> 或 --dir=<目录>")
        return

    log(f"待处理 {len(targets)} 段视频")
    records = []
    for i, v in enumerate(targets, 1):
        log(f"[{i}/{len(targets)}] {v.name}")
        rec = run_video(v, model, args, run_one)
        records.append(rec)
        if rec.get("status") == "ok":
            s = rec["summary"]
            log(f"  检出 {s['n_defect_tracks']} 处缺陷"
                f"（{s['n_tracks_multi_frame']} 处跨多帧）")

    out_json = Path(save_json) if save_json else (
        EVAL_DIR / "video_screen_results.json")
    dump_json({"model": resolved, "args": args, "results": records}, out_json)
    log(f"结果已写入 {out_json}")


if __name__ == "__main__":
    main()
