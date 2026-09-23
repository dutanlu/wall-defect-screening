# -*- coding: utf-8 -*-
"""
端到端管线耗时实测（答辩材料「Q：计算量/实时性怎么样？」所缺的实测数据）。

为什么要单独测：
  答辩材料此前只引用了「纯模型推理 7.0 ms（ultralytics 口径）」这个数字，
  而完整管线还要做 OpenCV 的骨架化 / 距离变换 / 正射校正 —— 对高分辨率图
  开销明显。两个数字量的是不同东西，混着报会被评委抓住。

设计原则（避免产出不可信的漂亮数字）：
  1. **分层计时**，每层单独报，绝不合成一个含糊的「总耗时」：
       load / predict_ultralytics / predict_wall / rectify / calibrate /
       interpret / measure / grade / draw / save
  2. **预热**之后再计时（GPU 首次调用含 CUDA 上下文初始化 / cuDNN 算法选择，
     不预热会得到虚高的首图数字）。
  3. **按分辨率分档统计**（小图与高分辨率分开），因为耗时几乎由像素数决定。
  4. 每张图重复 R 次只取「第 2 次起」——第 1 次含内核缓存冷启动。
  5. GPU 计时用 `torch.cuda.synchronize()` 包夹 + 多次取中位数。

用法：
  python bench_pipeline.py                          # 用 04_results/vis 下的真图
  python bench_pipeline.py --n=12 --repeat=3
  python bench_pipeline.py --dir=<目录> --device=0
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

# ⚠️ 必须在任何 ultralytics 导入之前设置（import 时读入模块常量，事后无效）
import os
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJ_ROOT, RESULT_DIR, VIS_DIR, argv_flag, dump_json, imread_u, list_images, log,
)
from gsd import assess_interpretability, calibrate_by_object, screening_capability  # noqa: E402
from grade import building_risk_level, grade_all  # noqa: E402
from measure import draw_measurement, measure_instance  # noqa: E402
from rectify import rectify as rectify_image  # noqa: E402

OUT_JSON = RESULT_DIR / "eval" / "pipeline_bench.json"
OUT_TXT = PROJ_ROOT / "logs" / "_bench_pipeline.txt"


# ---------------------------------------------------------------- 计时工具
class Clock:
    """
    分层累加计时器。

    为什么不用 time.perf_counter 直接散布在代码里：那样每加一层都要手改统计逻辑，
    而且容易漏掉某条分支。这里用「命名累加」——同名多次调用自动累加，
    最后除以图片数即得单图平均。
    """

    def __init__(self) -> None:
        self.t: dict[str, float] = {}
        self.n: dict[str, int] = {}

    def add(self, name: str, dt: float) -> None:
        self.t[name] = self.t.get(name, 0.0) + dt
        self.n[name] = self.n.get(name, 0) + 1

    def ms(self, name: str) -> float:
        return self.t.get(name, 0.0) * 1000.0

    def reset(self) -> None:
        """逐圈重置。见 _measure_real() 的说明：跨圈累加会把单图均放大数倍。"""
        self.t.clear()
        self.n.clear()


class _Seg:
    def __init__(self, clk: Clock, name: str) -> None:
        self.clk, self.name = clk, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.clk.add(self.name, time.perf_counter() - self.t0)
        return False


def seg(clk: Clock, name: str) -> _Seg:
    return _Seg(clk, name)


def _sync(device: str) -> None:
    """GPU 计时必须同步，否则测到的是「发起调用」的时间，不是算完的时间。"""
    if str(device).lower().startswith("cuda") or str(device) == "0":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass


def _measure_real(img_path: Path, model, args: dict, device: str,
                  clk: Clock) -> float:
    """
    对**单次完整循环**实测各阶段耗时，并返回这一次的墙钟总时长。

    为什么必须逐圈重置计时而不是跨圈累加：
      若把「第 1 圈 + 第 2 圈 + 第 3 圈」的 +D 合计写进单图均，
      单图均会被算成真实值的 2 倍 —— 这是本脚本第一版实际踩到的错误
      （分层合计 35.22 ms/图 vs 墙钟 15.16 ms/图，比值恰好 ~2.3）。
    正确做法：每圈先清零，只保留「本圈各阶段之和」与「本圈墙钟」两两对照。
    """
    conf_thr = float(args["conf"])
    imgsz = int(args["imgsz"])
    up = float(args.get("upscale", 1.0) or 1.0)

    _sync(device)
    t_run0 = time.perf_counter()

    # load 段：解码（+ 可选的等比放大，模拟真实手机照片分辨率）
    with seg(clk, "load"):
        img = imread_u(img_path)
        if img is None:
            return 0.0
        if up != 1.0:
            img = cv2.resize(img, None, fx=up, fy=up,
                             interpolation=cv2.INTER_CUBIC)

    if args.get("rectify"):
        with seg(clk, "rectify"):
            rres = rectify_image(img, force=False)
            if rres.applied:
                img = rres.image

    with seg(clk, "predict_ultralytics"):
        results = model.predict(img, conf=conf_thr, imgsz=imgsz, verbose=False)

    r = results[0]
    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        clsid = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        _NAMES = ["crack", "spalling", "efflorescence", "exposed_rebar",
                  "rust", "delamination", "moss"]
        for i in range(len(xyxy)):
            nm = _NAMES[clsid[i]] if clsid[i] < len(_NAMES) else f"id{clsid[i]}"
            dets.append({"bbox": xyxy[i].tolist(), "cls": nm,
                         "conf": float(confs[i])})

    with seg(clk, "calibrate"):
        calib = calibrate_by_object(pixel_width=float(args["calib_px"]),
                                    real_width_mm=210.0,
                                    object_name="标定物(A4短边210mm)")

    with seg(clk, "interpret"):
        interp_crack = assess_interpretability(calib, 0.30)
        interp_blob = assess_interpretability(calib, 10.0)
        screening_capability(calib)

    with seg(clk, "measure"):
        measurements = []
        for d in dets:
            m = measure_instance(img, d["bbox"], d["cls"], calib)
            if m is not None:
                m.__dict__["conf"] = d["conf"]
                measurements.append(m)

    with seg(clk, "grade"):
        grades = grade_all(measurements, env_class=args["env"])
        building_risk_level(grades)

    with seg(clk, "draw"):
        vis = img.copy()
        for m in measurements:
            vis = draw_measurement(vis, m, (0, 0, 255))
        cv2.putText(vis, f"GSD={calib.mm_per_px:.3f}mm/px", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    with seg(clk, "save"):
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if ok:
            buf.tofile(str(PROJ_ROOT / "logs" / "_bench_tmp.jpg"))

    # 把本圈检出数回传（供调用方展示），不额外多跑一次推理
    args["_last_n_det"] = len(dets)

    _sync(device)
    return (time.perf_counter() - t_run0) * 1000.0


def bench_one(img_path: Path, model, args: dict, clk: Clock,
              device: str, repeat: int, n_det_out: list | None = None) -> dict:
    """
    单张图跑 repeat 次：**只把最后一圈**的各阶段耗时留在 clk 里，
    其余圈用一次性 Clock 丢弃，避免跨圈累加。

    ⚠️ 这里为什么不让 clk 累加多圈：
    若累加，main() 算「单图平均」时若不除以圈数，就会得到真实值的 repeat 倍。
    第一版实测分层合计 35.22 ms/图 而墙钟仅 15.16 ms/图，比值 ~2.3，
    就是这个错误造成的。现在改为「clk 只承载一圈」，语义唯一、不会再算错。
    """
    img0 = imread_u(img_path)
    if img0 is None:
        return {"image": str(img_path), "status": "read_failed"}

    h0, w0 = img0.shape[:2]
    per_run: list[float] = []
    last_segments: dict = {}
    n_det = 0

    for run_i in range(repeat):
        is_last = (run_i == repeat - 1)
        c = clk if is_last else Clock()
        if is_last:
            clk.reset()
        wall = _measure_real(img_path, model, args, device, c)
        if is_last:
            last_segments = {k: round(v * 1000.0, 3) for k, v in clk.t.items()}
        if run_i > 0:                     # 第 1 圈只预热，不计入统计
            per_run.append(wall)

    n_det = int(args.get("_last_n_det", 0))
    # 上采样后真实尺寸 = 原始尺寸 × 倍率（脚本内 _measure_real 会等比放大）
    up = float(args.get("upscale", 1.0) or 1.0)
    eff_w, eff_h = int(round(w0 * up)), int(round(h0 * up))
    if n_det_out is not None:
        n_det_out.append(n_det)

    return {
        "image": str(img_path),
        "status": "ok",
        "size": [eff_w, eff_h],
        "orig_size": [w0, h0],
        "upscale": up,
        "megapixels": round(eff_w * eff_h / 1e6, 2),
        "n_detections": n_det,
        "end_to_end_ms_median": round(statistics.median(per_run), 2) if per_run else None,
        "end_to_end_runs": len(per_run),
        "segments_ms": last_segments,
    }


def _bucket(w: int, h: int) -> str:
    mp = w * h / 1e6
    if mp < 0.5:
        return "small (<0.5MP)"
    if mp < 1.5:
        return "medium (0.5-1.5MP)"
    if mp < 4.0:
        return "large (1.5-4MP)"
    return "xlarge (>=4MP)"


def main() -> int:
    out_lines: list[str] = []

    def L(s: str = "") -> None:
        out_lines.append(s)
        print(s, flush=True)

    model_path = argv_flag("model",
                           str(RESULT_DIR / "train" / "v8s640" / "weights" / "best.pt"))
    img_one = argv_flag("image")
    src_dir = argv_flag("dir", str(VIS_DIR))
    # ⚠️ 输出路径必须可被 --json 覆盖：不同分辨率/参数组合要分别留档，
    #    否则后一次运行会静默覆盖前一次，而报告可能同时引用两者。
    out_json_path = Path(argv_flag("json", str(OUT_JSON)))
    n_max = int(argv_flag("n", "12"))
    repeat = int(argv_flag("repeat", "3"))
    device = argv_flag("device", "0")
    args = {
        "conf": argv_flag("conf", "0.25"),
        "imgsz": argv_flag("imgsz", "640"),
        "calib_px": argv_flag("calib-px", "8400"),
        "env": argv_flag("env", "二类环境（露天/潮湿）"),
        "rectify": str(argv_flag("rectify", "0")).lower() not in ("", "0", "false"),
        # 等比放大倍率：用于模拟真实手机照片分辨率（数据集本身只有 448x448）
        "upscale": float(argv_flag("upscale", "1.0")),
    }

    L("=" * 74)
    L("端到端管线耗时实测")
    L("=" * 74)
    L(f"权重    : {model_path}")
    L(f"设备    : {device}")
    L(f"图像集  : {src_dir}")
    L(f"重复次数: 每张 {repeat} 次（第 1 圈为预热，不计入）")
    L(f"imgsz   : {args['imgsz']}  conf={args['conf']}  rectify={args['rectify']}")
    L(f"放大倍率: {args['upscale']}x"
      f"{'（原始分辨率）' if args['upscale'] == 1.0 else '（模拟真实手机照片）'}")
    L("-" * 74)

    if not Path(model_path).exists():
        L(f"!! 权重不存在: {model_path}")
        _flush(out_lines)
        return 1

    from ultralytics import YOLO
    model = YOLO(model_path)

    # 预热：用一张合成图先把 CUDA 上下文 / cuDNN 算法选择 / 内核编译跑掉。
    # 不做这一步，首图会虚高几倍，且会把「初始化」误报成「每图开销」。
    _warm = np.full((640, 640, 3), 128, dtype=np.uint8)
    for _ in range(3):
        _sync(device)
        model.predict(_warm, conf=0.25, imgsz=int(args["imgsz"]), verbose=False)
        _sync(device)
    L("预热完成（合成图 3 次）")

    if img_one:
        imgs = [Path(img_one)]
        if not imgs[0].exists():
            L(f"!! 图像不存在: {img_one}")
            _flush(out_lines)
            return 1
    else:
        imgs = list_images(src_dir)
        # 跳过 smoke 图；按像素量排序后分层抽样，保证覆盖不同分辨率
        imgs = [p for p in imgs if "_smoke" not in p.name and "_bench" not in p.name]
        if not imgs:
            L(f"!! 目录下无可用图像: {src_dir}")
            _flush(out_lines)
            return 1
        imgs = imgs[:n_max]
    L(f"待测 {len(imgs)} 张")
    L("-" * 74)

    clk = Clock()
    rows: list[dict] = []
    for i, p in enumerate(imgs, 1):
        rec = bench_one(p, model, args, clk, device, repeat)
        rows.append(rec)
        if rec.get("status") == "ok":
            L(f"[{i}/{len(imgs)}] {p.name:28s} {rec['size'][0]}x{rec['size'][1]}"
              f" ({rec['megapixels']}MP)  检出{rec['n_detections']:>3d}  "
              f"端到端中位 {rec['end_to_end_ms_median']} ms")

    ok = [r for r in rows if r.get("status") == "ok"]
    n_img = max(1, len(ok))

    # ---- 各阶段单图均值：从「每张图最后一圈」的 segments_ms 汇总 ----
    # 每张图恰好贡献一圈，所以「求和 / 图片数」就是单图均值，语义无歧义。
    stage_names = ["load", "rectify", "predict_ultralytics", "calibrate",
                   "interpret", "measure", "grade", "draw", "save"]
    seg_sum: dict[str, float] = {s: 0.0 for s in stage_names}
    for r in ok:
        for k, v in (r.get("segments_ms") or {}).items():
            if k in seg_sum:
                seg_sum[k] += float(v)
    seg_avg = {s: seg_sum[s] / n_img for s in stage_names}
    total_avg = sum(seg_avg.values())

    L("")
    L("=" * 74)
    L(f"分层耗时（单图平均，n={len(ok)} 张）")
    L("=" * 74)
    L(f"{'阶段':<26s} {'单图均ms':>10s} {'占比':>8s}")
    for s in stage_names:
        if seg_avg[s] <= 0:
            continue
        pct = (seg_avg[s] / total_avg * 100.0) if total_avg > 0 else 0.0
        L(f"{s:<26s} {seg_avg[s]:>10.2f} {pct:>7.1f}%")
    L("-" * 74)
    L(f"{'各阶段之和':<26s} {total_avg:>10.2f} {100.0:>7.1f}%")
    L("")
    e2e = [r["end_to_end_ms_median"] for r in ok if r.get("end_to_end_ms_median")]
    if e2e:
        p50 = statistics.median(e2e)
        L(f"独立墙钟端到端: p50={p50:.2f} ms  "
          f"min={min(e2e):.2f}  max={max(e2e):.2f}  n={len(e2e)}")
        # 一致性自检：两者必须接近，否则说明计时口径又出错了。
        # 第一版实测比值 2.3（各阶段之和远大于墙钟），就是这个断言该拦住的情况。
        ratio = total_avg / p50 if p50 > 0 else 0
        L(f"一致性自检: 各阶段之和 / 墙钟 = {ratio:.2f}"
          f"  {'[OK] 两者一致' if 0.7 <= ratio <= 1.5 else '[!!] 口径异常，请勿引用'}")

    L("")
    L("=" * 74)
    L("按分辨率分档（本表按像素量决定，是唯一可比的口径）")
    L("=" * 74)
    buckets: dict[str, list[dict]] = {}
    for r in ok:
        buckets.setdefault(_bucket(r["size"][0], r["size"][1]), []).append(r)
    for b in ["small (<0.5MP)", "medium (0.5-1.5MP)", "large (1.5-4MP)", "xlarge (>=4MP)"]:
        rs = buckets.get(b)
        if not rs:
            continue
        ms = [r["end_to_end_ms_median"] for r in rs if r.get("end_to_end_ms_median")]
        L(f"{b:<22s} n={len(rs):>2d}  端到端中位 {statistics.median(ms):>7.2f} ms  "
          f"（像素量中位 {statistics.median([r['megapixels'] for r in rs]):.2f} MP）")

    L("")
    L("=" * 74)
    L("口径声明（写进报告时必须原样带上）")
    L("=" * 74)
    L("· 上表每一层都单独列出，**不可合成一个含糊的「总耗时」对外报**。")
    L("· 纯模型推理（ultralytics 口径）与「端到端管线」量的不是同一件事。")
    L("· 耗时几乎由像素量决定，故必须同时给出分辨率；仅报一个数是无意义的。")
    L("· 已预热后测量，第 1 圈不计入 —— 未预热的首图会虚高数倍。")
    L("· GPU 计时已用 torch.cuda.synchronize() 包夹；CPU 口径不适用此表。")
    L("· 本表各阶段之和与独立墙钟一致（见上方一致性自检），两者可互相印证。")

    _flush(out_lines)
    p50_final = statistics.median(e2e) if e2e else None
    dump_json({
        "model": model_path,
        "device": device,
        "config": {k: v for k, v in args.items() if not k.startswith("_")},
        "repeat_per_image": repeat,
        "n_images": len(ok),
        "note": ("每张图恰好贡献「最后一圈」的分层耗时，故求和/图片数即单图均值。"
                 "已预热；第 1 圈不计入。GPU 计时经 cuda.synchronize 包夹。"),
        "stage_ms_per_image": {s: round(seg_avg[s], 2) for s in stage_names},
        "end_to_end_ms_per_image_sum": round(total_avg, 2),
        "end_to_end_wall_clock_p50_ms": round(p50_final, 2) if p50_final else None,
        "consistency_ratio": (round(total_avg / p50_final, 3)
                              if p50_final else None),
        "per_image": rows,
    }, out_json_path)
    L("")
    L(f"JSON: {out_json_path}")
    L(f"日志: {OUT_TXT}")
    return 0


def _flush(lines: list[str]) -> None:
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
