# -*- coding: utf-8 -*-
"""rt_infer.py —— 无人机外墙缺陷「实时识别」主程序。

归属：09_uav_realtime/（独立实验目录）。**不修改任何既有源文件**，
只读复用 02_code/pipeline.py 的 run_one 作为唯一推理链路。

一句话：把视频源（RTSP / USB / 文件 / 摄像头）的帧喂给主链路，逐帧量测
耗时与延迟，叠加检测框+中文标签，并把结果落盘以便复核。

------------------------------------------------------------------ 设计要点
【为什么必须新建模块，而不是改 video_screen.py】
  video_screen.py 是「事后分析型」：按时间等间隔抽帧、跨帧贪心聚合，
  目标是"看完一段视频给一份覆盖面结论"。它没有 FPS / 端到端延迟概念，
  也没有流式读取与重连。本模块是「实时型」：关心每帧多久出结果、
  源断了怎么办。两者目的不同，硬塞在一起会同时污染双方的口径。

【为什么走 run_one(image=frame)】
  run_one 已支持传入已解码帧（pipeline.py 第 157/169 行），可跳过磁盘读取。
  自写一套检测+分级逻辑会让本实验结论无法与主链路对齐 —— 那实验就白做了。

【为什么要跳帧采样】
  无人机常 30fps 而模型跑不到 30fps。若逐帧必推理，要么帧率被拖垮、
  要么缓冲越积越多导致延迟无上限增长（实时系统最致命的失效模式）。
  故按 target_fps 算 stride 主动丢帧。target_fps=0 时退化为逐帧全跑。

【三级指标定义（写进 JSON，报告须用这三个口径）】
  infer_ms    ：单次 run_one 的墙钟耗时（含检测+量化+分级+绘制）
  proc_fps    ：实际完成推理的帧数 / 处理墙钟秒   ← 决定"能不能实时交互"
  e2e_ms      ：从"取到这一帧"到"叠加图产出"的墙钟差  ← 决定"反馈迟不迟"
  另记 decode_ms（取帧耗时）以便定位瓶颈在解码还是在模型。

【边界声明（务必如实写进报告）】
  工程内没有真实实拍外墙视频，logs/_test_video 下的合成视频只能证明
  管线连通，**不得据此声称已完成实拍验证**（见 logs/_VIDEO_SCREEN_VERIFY.md）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
# ★ 兼容两种布局：源目录 02_code/ 与交付包 code/（见 rt_common.py 同名注释）
_CODE_CANDIDATES = (_HERE.parent / "02_code", _HERE.parent / "code")
for _cand in _CODE_CANDIDATES:
    if (_cand / "common.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break
else:
    raise SystemExit("找不到既有源码目录（期望 common.py 位于 %s 之一）"
                     % " 或 ".join(str(c) for c in _CODE_CANDIDATES))

from common import dump_json, log  # noqa: E402
from pipeline import run_one  # noqa: E402
import rt_common as R  # noqa: E402


# ============================================================================
# 源打开
# ============================================================================

def open_source(src: dict, cfg: dict):
    """按 kind 分派打开视频源，返回 (cap, meta)。

    meta 含：kind / uri / fps / width / height / frame_count / is_live
    """
    kind = src["kind"]
    stream_cfg = src.get("stream") or {}
    cap_cfg = src.get("capture") or {}
    is_live = False

    if kind == "file":
        uri = src["uri_resolved"]
        cap = cv2.VideoCapture(uri)
        if not cap.isOpened():
            raise SystemExit(f"[源打开失败] 无法打开视频文件：{uri}")

    elif kind == "device":
        # OpenCV 设备号必须传 int；配置里为便于书写存的是字符串
        try:
            dev = int(str(src["uri"]).strip())
        except ValueError:
            raise SystemExit(f"[源打开失败] device 源的 uri 必须是整数设备号，"
                             f"实际为 '{src['uri']}'")
        # Windows 上 DirectShow 打开更快更稳；失败再退回默认后端
        cap = cv2.VideoCapture(dev, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            raise SystemExit(f"[源打开失败] 无法打开摄像头设备 {dev}（可能被占用）")
        is_live = True

    elif kind == "stream":
        uri = str(src["uri"]).strip()
        if not uri:
            raise SystemExit("[源打开失败] stream 源 uri 为空")
        # 实时流直连；OPENCV_FFMPEG_* 环境变量控制底层超时（毫秒）
        # ★ 必须在 VideoCapture 之前设置，否则不生效
        if stream_cfg.get("open_timeout_sec"):
            ms = int(float(stream_cfg["open_timeout_sec"]) * 1000)
            import os
            os.environ["OPENCV_FFMPEG_OPEN_TIMEOUT"] = str(ms)
        if stream_cfg.get("read_timeout_sec"):
            ms = int(float(stream_cfg["read_timeout_sec"]) * 1000)
            import os
            os.environ["OPENCV_FFMPEG_READ_TIMEOUT"] = str(ms)

        # transport 选择：tcp 更稳，udp 更低延迟
        if str(stream_cfg.get("transport", "tcp")).lower() == "tcp" and \
                uri.startswith("rtsp://") and "transport" not in uri:
            sep = "&" if "?" in uri else "?"
            uri = f"{uri}{sep}transport=tcp"
        cap = cv2.VideoCapture(uri)
        if not cap.isOpened():
            raise SystemExit(f"[源打开失败] 无法打开视频流：{uri}")
        is_live = True

        # 只留最新帧：实时系统绝不能堆积缓冲区，否则延迟无限增长
        bs = stream_cfg.get("buffer_size")
        if bs:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, int(bs))
            except Exception:
                pass
    elif kind == "wifi_fpv":
        # 无人机 WiFi 私有图传（WiFi_CAM 类公版方案：UDP + MJPEG/H.264）。
        # ★ 来源：本机实测说明书为 `WiFi_CAM` App + WIFI_____xxx 热点，
        #   既不是 RTSP 也不是标准 IP 摄像头 ⇒ 需要专门的接收端。
        #   接收端返回的是 cv2.VideoCapture 兼容对象，故主循环无需改动。
        try:
            from uav_receiver import open_uav_stream
        except ImportError as e:
            raise SystemExit(
                f"[源打开失败] 导入 uav_receiver 失败：{e}\n"
                f"  该模块位于本目录，请确认在 09_uav_realtime/ 下运行。")
        cap = open_uav_stream(src, verbose=True)
        if cap is None:
            raise SystemExit(
                f"[源打开失败] 无法在 UDP 端口 {src.get('port')} 上开始监听。\n"
                f"  排查：1) 是否已连上无人机热点 WIFI_____xxx\n"
                f"        2) 端口号是否正确（先用 uav_probe.py 实测）\n"
                f"        3) 端口是否被其它程序占用")
        is_live = True

    else:  # pragma: no cover
        raise SystemExit(f"[源打开失败] 未知 kind={kind}")

    # --- wifi_fpv 源的分辨率/帧率由接收端决定，跳过 OpenCV 属性请求 ---
    if kind == "wifi_fpv":
        meta = {
            "kind": kind, "uri": f"udp://{src.get('drone_ip','')}:{src.get('port')}",
            "fps": float(src.get("assumed_fps", 30.0) or 30.0),
            "width": int(src.get("width", 0) or 0),
            "height": int(src.get("height", 0) or 0),
            "frame_count": 0, "is_live": True,
        }
        log(f"[源] 无人机 WiFi 图传：UDP 端口 {src.get('port')}"
            f"（流类型 {src.get('stream_type','auto')}）")
        return cap, meta

    # --- 分辨率 / 帧率请求（0 表示不强制）---
    req_w = int(cap_cfg.get("width", 0) or 0)
    req_h = int(cap_cfg.get("height", 0) or 0)
    req_fps = float(cap_cfg.get("request_fps", 0) or 0)
    if req_w > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    if req_h > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    if req_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, req_fps)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 实时流常报 fps=0 或离谱值；给一个保守默认，否则 stride 算不出来
    if fps <= 0.5 or fps > 240:
        fps = 30.0
    if w <= 0 or h <= 0:
        # 再试读一帧探真实尺寸（读不到就让后续循环报错，不在此吞掉）
        ok, fr = cap.read()
        if ok and fr is not None:
            h, w = fr.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) if not is_live else None

    meta = {
        "kind": kind, "uri": src.get("uri") or src.get("uri_resolved"),
        "fps": round(fps, 3), "width": w, "height": h,
        "frame_count": n, "is_live": is_live,
    }
    return cap, meta


def compute_stride(src_fps: float, target_fps: float) -> int:
    """按「源帧率 / 目标推理帧率」算跳帧步长。

    target_fps<=0 → 返回 1（逐帧全跑，不跳帧）。
    结果至少为 1，避免 stride=0 导致除零或死循环。
    """
    if not target_fps or target_fps <= 0:
        return 1
    if src_fps <= 0:
        src_fps = 30.0
    return max(1, int(round(src_fps / float(target_fps))))


# ============================================================================
# 主循环
# ============================================================================

def run(cfg: dict, source_key: str | None = None) -> dict:
    """跑完一次实验，返回结果 dict（同时已落盘）。"""
    src = R.resolve_source(cfg, source_key)
    args = R.build_args(cfg)
    out_cfg = cfg.get("output") or {}
    cap_cfg = src.get("capture") or {}
    met_cfg = cfg.get("metrics") or {}
    stream_cfg = src.get("stream") or {}

    device = args["device"]
    target_fps = float(cap_cfg.get("target_fps", 0) or 0)
    max_frames = int(cap_cfg.get("max_frames", 0) or 0)
    duration_sec = float(cap_cfg.get("duration_sec", 0) or 0)
    warmup = int(cap_cfg.get("warmup_frames", 0) or 0)

    # --- 输出目录 ---
    root = _HERE / str(out_cfg.get("root", "."))
    frames_dir = root / str(out_cfg.get("frames_dir", "out_frames"))
    video_dir = root / str(out_cfg.get("video_dir", "out_video"))
    logs_dir = root / str(out_cfg.get("logs_dir", "logs"))
    for d in (frames_dir, video_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"{src['key']}_{stamp}"

    log("=" * 78)
    log(f"无人机实时识别实验  源预设={src['key']}  kind={src['kind']}")
    log(f"  配置：{cfg.get('_config_path')}")
    log(f"  模型：{Path(args['model']).name}   imgsz={args['imgsz']}   conf={args['conf']}")
    log(f"  设备：{device}   环境类别：{args['env']}")
    log("=" * 78)

    # --- 打开源 ---
    cap, meta = open_source(src, cfg)
    src_fps = meta["fps"]
    stride = compute_stride(src_fps, target_fps)
    log(f"[源] {meta['uri']}")
    log(f"     分辨率 {meta['width']}x{meta['height']}  fps={src_fps}  "
        f"总帧={meta['frame_count']}  实时流={meta['is_live']}")
    log(f"[采样] target_fps={target_fps}  ⇒ stride={stride} "
        f"（每 {stride} 帧取 1 帧送推理）")
    log(f"[终止] max_frames={max_frames}  duration_sec={duration_sec}  "
        f"warmup={warmup}（前 {warmup} 帧丢作预热）")

    # --- 载入模型（★ 只载一次。每帧重载会让耗时虚高数十倍）---
    from ultralytics import YOLO
    model = YOLO(args["model"])
    try:
        model.to(device)
    except Exception as e:
        log(f"  [警告] model.to({device}) 失败（{e}），改用默认设备")

    # --- 回放视频 writer（懒创建：要等第一帧才知道尺寸）---
    writer = None
    save_video = bool(out_cfg.get("save_video", False))
    video_path = video_dir / f"{tag}.mp4"

    # --- 统计容器 ---
    infer_stats = R.LatencyStats(met_cfg.get("percentiles", [50, 90, 99]))
    decode_stats = R.LatencyStats(met_cfg.get("percentiles", [50, 90, 99]))
    e2e_stats = R.LatencyStats(met_cfg.get("percentiles", [50, 90, 99]))

    records: list[dict] = []          # 逐帧结构化记录
    n_read = 0                        # 读到的原始帧数
    n_infer = 0                       # 已消费的推理配额（含预热）
    n_warmup_done = 0                 # 已完成的预热帧数
    n_measured = 0                    # 计入统计的正式推理帧数
    n_empty = 0                       # 0 检出的推理帧
    n_fail = 0                        # run_one 返回 read_failed 等异常
    cls_counter: dict[str, int] = {}  # 类别命中计数
    save_every = int(out_cfg.get("save_frame_every", 0) or 0)
    max_saved = int(out_cfg.get("max_saved_frames", 0) or 0)
    n_saved = 0

    # ★ 预热与正式推理是两套预算。
    #   教训（本模块首次冒烟实测）：若把预热帧也算进 max_frames，则在
    #   max_frames < warmup 的短跑里，全部配额都被预热吃掉，正式统计样本为 0，
    #   汇总里所有耗时字段变成 None —— 看起来"跑成功了"，实际什么也没测到。
    if warmup > 0 and max_frames > 0 and warmup >= max_frames:
        log(f"  [提示] warmup={warmup} ≥ max_frames={max_frames}："
            f"预热独立预算，正式统计仍会跑满 {max_frames} 帧。"
            f"短跑时建议把 warmup 调小以获得更贴近该长度的数字。")

    t_start = time.perf_counter()
    last_log = t_start
    reconnect_left = int(stream_cfg.get("max_reconnect", 0) or 0)

    try:
        while True:
            # --- 终止条件 A：时长 ---
            if duration_sec > 0 and (time.perf_counter() - t_start) >= duration_sec:
                log(f"[终止] 已达 duration_sec={duration_sec}")
                break
            # --- 终止条件 B：正式统计帧数（不含预热）---
            if max_frames > 0 and n_measured >= max_frames:
                log(f"[终止] 已达 max_frames={max_frames}")
                break

            # --- 取帧（计时）---
            with R.Seg(device="cpu") as s_dec:
                ok, frame = cap.read()
            if not ok or frame is None:
                if meta["is_live"] and reconnect_left > 0:
                    reconnect_left -= 1
                    wait = float(stream_cfg.get("reconnect_wait_sec", 1.0) or 1.0)
                    log(f"  [流中断] 等待 {wait}s 后重连 "
                        f"（剩余重连次数 {reconnect_left}）")
                    time.sleep(wait)
                    cap.release()
                    try:
                        cap, meta = open_source(src, cfg)
                    except SystemExit as e:
                        log(f"  [重连失败] {e}")
                        break
                    continue
                if meta["is_live"]:
                    log("  [流结束] 实时流不再返回帧（或重连次数用尽）")
                else:
                    log(f"  [文件结束] 共读 {n_read} 帧")
                break

            n_read += 1
            # --- 跳帧节流：不是采样点就丢弃（只读不推理，成本极低）---
            if stride > 1 and (n_read - 1) % stride != 0:
                continue

            # --- 预热：丢弃前 N 帧的推理结果（含 CUDA 上下文/显存分配开销）---
            # 预热帧不写入 records、不进分位统计，但确实跑完整推理以让
            # cudnn 完成 autotune。预热有独立预算，不占 max_frames。
            if n_warmup_done < warmup:
                _ = run_one(Path(f"warmup_{n_read:06d}"), model, args, image=frame)
                n_warmup_done += 1
                n_infer += 1
                log(f"  [预热] {n_warmup_done}/{warmup}")
                continue

            # --- 主推理 + 端到端计时 ---
            t_e2e0 = time.perf_counter()
            with R.Seg(device) as s_inf:
                res = run_one(Path(f"{src['key']}_f{n_read:06d}"), model, args,
                              image=frame)

            if res.get("status") == "read_failed":
                n_fail += 1
                log(f"  [异常] 帧 {n_read} run_one 返回 read_failed")
                continue

            # --- 叠加 ---------------------------------------------------
            # ★ 实测教训（本模块首轮跑通即触发）：
            #   run_one 返回的 `annotated` 是**文件路径字符串**（未开
            #   --save-annotated 时为 None），**不是图像数组**。曾误当数组用，
            #   于是静默回退成原帧、一个框都没画上，且不报错。
            #   ⇒ 正确做法：本层用 measure.draw_measurement 重画框
            #     （与主链路同一函数 ⇒ 样式一致），再叠中文标签与 HUD。
            ms_list = res.get("measurements") or []

            if out_cfg.get("draw_boxes", True):
                vis = R.draw_boxes(frame, ms_list, res.get("grades"))
            else:
                vis = frame.copy()

            # 左上信息条：GSD / 可判读性 / 风险等级
            vis = R.draw_banner(vis, res.get("calibration") or {},
                                res.get("risk") or {},
                                res.get("interpretability"))

            # 中文类别标签（检测框上方）
            if out_cfg.get("draw_labels", True):
                vis = R.draw_det_labels(vis, ms_list)

            # HUD 放在左下角，避免与左上信息条重叠
            if out_cfg.get("draw_metrics_hud", True):
                h = vis.shape[0]
                hud = [
                    f"帧 {n_read}  推理 {s_inf.ms:.1f} ms  "
                    f"延迟 {(time.perf_counter() - t_e2e0) * 1000:.1f} ms",
                    f"检出 {res.get('n_detections', 0)}  "
                    f"风险 {res.get('risk', {}).get('level', '-')}",
                ]
                vis = R.draw_hud(vis, hud, size=18, origin=(8, max(h - 70, 0)))
            e2e_ms = (time.perf_counter() - t_e2e0) * 1000.0

            n_measured += 1
            infer_stats.add(s_inf.ms)
            decode_stats.add(s_dec.ms)
            e2e_stats.add(e2e_ms)

            if not ms_list:
                n_empty += 1
            for m in ms_list:
                c = m.get("cls_name", "?")
                cls_counter[c] = cls_counter.get(c, 0) + 1

            # --- 逐帧结构化记录 ---
            # ★ n_abstained 说明：run_one 的返回里**没有**这个字段。
            #   apply_abstention 内部算出 n_unjudgeable 后只用于改写 grades，
            #   并未传出（pipeline.py:267）。故此处不编造该字段，改为记录
            #   可从 risk 推出的弃权信号：risk.level == "U" 即「全弃权」。
            records.append({
                "frame_no": n_read,
                "infer_index": n_measured,
                "infer_ms": round(s_inf.ms, 3),
                "decode_ms": round(s_dec.ms, 3),
                "e2e_ms": round(e2e_ms, 3),
                "n_detections": res.get("n_detections", 0),
                "risk_level": (res.get("risk") or {}).get("level"),
                "all_abstained": (res.get("risk") or {}).get("level") == "U",
                "gsd_mm_per_px": (res.get("calibration") or {}).get("mm_per_px"),
                "calib_method": (res.get("calibration") or {}).get("method"),
                "can_screen": (res.get("interpretability") or {}).get("can_screen"),
                "classes": sorted({m.get("cls_name") for m in ms_list}),
                "confs": [round(float(m.get("conf", 0) or 0), 4) for m in ms_list],
            })

            # --- 落盘：回放视频 ---
            if save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(
                        *str(out_cfg.get("video_fourcc", "mp4v")).ljust(4)[:4])
                    vfps = float(out_cfg.get("video_fps", 10.0) or 10.0)
                    writer = cv2.VideoWriter(str(video_path), fourcc, vfps, (w, h))
                    if not writer.isOpened():
                        log(f"  [警告] 回放视频 writer 打开失败，关闭视频落盘：{video_path}")
                        writer = None
                        save_video = False
                if writer is not None:
                    writer.write(vis)

            # --- 落盘：抽样截图 ---
            if save_every > 0 and (n_measured % save_every == 0):
                if max_saved <= 0 or n_saved < max_saved:
                    fp = frames_dir / f"{tag}_f{n_read:06d}.jpg"
                    if R.save_frame(fp, vis):
                        n_saved += 1
                    else:
                        log(f"  [警告] 截图落盘失败：{fp}")

            # --- 心跳日志 ---
            now = time.perf_counter()
            if now - last_log >= 5.0:
                el = now - t_start
                log(f"  ... 读 {n_read} 帧 / 统计推理 {n_measured} 帧 / 用时 {el:.1f}s "
                    f"/ 处理 FPS {n_measured / el:.2f} / "
                    f"最近推理 {s_inf.ms:.1f} ms")
                last_log = now

    finally:
        if writer is not None:
            writer.release()
        cap.release()

    wall = time.perf_counter() - t_start

    # ==================================================================== 汇总
    infer_sum = infer_stats.summary()
    decode_sum = decode_stats.summary()
    e2e_sum = e2e_stats.summary()

    # ★★ 吞吐口径（本模块第二轮实测暴露的重要问题，务必看清）★★
    #   最初用「统计帧数 / 墙钟」当处理 FPS，但那会被**素材长度**绑架：
    #   40 帧素材跑完只用 4.9s，算出 2.04 FPS —— 而单帧推理其实只要 61ms
    #   （真实吞吐 16.2 FPS）。原因是素材太短，"读帧→跑完"整个过程没跑满，
    #   分母里混入了模型加载、建 writer、预热等一次性开销。
    #
    #   ⇒ 报告应以 **throughput_fps**（由单帧推理耗时反推）为准，
    #     它是与素材长度无关的稳态吞吐；素材很长时它才与 wall_clock_fps 收敛。
    #   wall_clock_fps 保留，仅作"含全部开销的端到端实测"参考。
    wall_clock_fps = (n_measured / wall) if wall > 0 else 0.0
    throughput_fps = R.fps_of(infer_sum.get("mean_ms", 0.0))
    throughput_fps_p90 = R.fps_of(infer_sum.get("p90_ms", 0.0))
    e2e_fps = R.fps_of(e2e_sum.get("mean_ms", 0.0))

    fps_pass = float(met_cfg.get("realtime_fps_pass", 15.0))
    lat_pass = float(met_cfg.get("e2e_latency_pass_ms", 200.0))
    verdict = {
        "throughput_fps": round(throughput_fps, 3),
        "throughput_fps_p90": round(throughput_fps_p90, 3),
        "e2e_throughput_fps": round(e2e_fps, 3),
        "wall_clock_fps": round(wall_clock_fps, 3),
        "fps_pass_threshold": fps_pass,
        "e2e_p90_ms": e2e_sum.get("p90_ms"),
        "latency_pass_threshold_ms": lat_pass,
        # 判定用吞吐口径（与素材长度无关的那个），不用 wall_clock_fps
        "realtime_ok": bool(throughput_fps >= fps_pass),
        "latency_ok": bool((e2e_sum.get("p90_ms") or 1e9) <= lat_pass),
        "note": ("throughput_fps 由单帧推理均值反推，是稳态吞吐、与素材长度无关；"
                 "wall_clock_fps 含预热/加载等一次性开销，短素材上会显著偏低，"
                 "仅在素材足够长时才与前者收敛。"),
    }

    # 类别计数按 common.CLASSES 顺序输出，便于与主链路口径对齐
    cls_hits = {c: cls_counter.get(c, 0) for c in R.CLASSES}
    cls_hits["_其他"] = sum(v for k, v in cls_counter.items() if k not in R.CLASSES)

    result = {
        "schema": "uav_rt_infer/v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": cfg.get("_config_path"),
        "source": {**meta, "preset": src["key"]},
        "sampling": {"target_fps": target_fps, "stride": stride,
                     "warmup_frames": warmup,
                     "max_frames": max_frames, "duration_sec": duration_sec},
        "infer_args": {k: v for k, v in args.items() if k != "model"},
        "model": Path(args["model"]).name,
        "device": device,
        "counts": {"frames_read": n_read, "warmup_done": n_warmup_done,
                   "frames_measured": n_measured, "frames_inferred_total": n_infer,
                   "measured_empty": n_empty, "measured_failed": n_fail,
                   "frames_saved": n_saved},
        "class_hits": cls_hits,
        "timing_ms": {"infer": infer_sum, "decode": decode_sum, "e2e": e2e_sum},
        "wall_sec": round(wall, 3),
        "verdict": verdict,
        "artifacts": {
            "video": str(video_path) if (save_video and video_path.exists()) else None,
            "frames_dir": str(frames_dir) if n_saved else None,
        },
        "caveat": ("本结果来自实时识别管线，素材可能是合成/录屏/自采视频；"
                   "若 source.kind 非无人机实拍流，不得据此声称已完成实拍验证。"
                   "（工程内无真实实拍外墙视频，见 logs/_VIDEO_SCREEN_VERIFY.md）"),
        "per_frame": records,
    }

    # --- 落盘 JSON / 日志 ---
    json_path = None
    if out_cfg.get("save_json", True):
        json_path = logs_dir / f"rt_metrics_{tag}.json"
        dump_json(result, json_path)

    # ---- 控制台/日志汇总 ----
    log("=" * 78)
    log("实验结果汇总")
    if n_measured == 0:
        # ★ 显式区分「跑完但没测到」与「跑完且测到了」。
        #   静默输出一堆 None 会让人误以为实验成功 —— 这是本工程反复踩过的
        #   「静默失败」模式（训练看退出码不看落盘、copy_paste 静默 no-op 同类）。
        log("  [无效实验] 统计推理帧数为 0：没有任何耗时样本可用。")
        log("  可能原因：源提前结束 / max_frames 或 duration_sec 太小 / 流未打开。")
        log("  ⇒ 本结果不得用于任何性能结论。")
        if json_path is not None:
            log(f"  （诊断记录仍已落盘：{json_path}）")
        log("=" * 78)
        return result
    log(f"  读帧 {n_read} / 预热 {n_warmup_done} / 统计推理 {n_measured}"
        f"（其中 0 检出 {n_empty}，异常 {n_fail}）")
    log(f"  墙钟 {wall:.2f}s")
    log(f"  ★ 吞吐 FPS = {throughput_fps:.2f}（由单帧推理反推，稳态口径）"
        f"  p90 口径 {throughput_fps_p90:.2f}"
        f"  ⇒ 阈值 {fps_pass}：{'通过' if verdict['realtime_ok'] else '未通过'}")
    log(f"    端到端口径吞吐 {e2e_fps:.2f} FPS；"
        f"墙钟口径 {wall_clock_fps:.2f} FPS（含加载/预热等一次性开销，"
        f"短素材会显著偏低）")
    log(f"  单帧推理 ms: mean={infer_sum.get('mean_ms')} "
        f"p50={infer_sum.get('p50_ms')} p90={infer_sum.get('p90_ms')} "
        f"p99={infer_sum.get('p99_ms')} max={infer_sum.get('max_ms')}")
    log(f"  端到端延迟 ms: mean={e2e_sum.get('mean_ms')} "
        f"p90={e2e_sum.get('p90_ms')} p99={e2e_sum.get('p99_ms')} "
        f"（阈值 {lat_pass} ⇒ {'通过' if verdict['latency_ok'] else '未通过'}）")
    log(f"  取帧耗时 ms: mean={decode_sum.get('mean_ms')}  "
        f"⇒ 瓶颈定位：{'解码' if (decode_sum.get('mean_ms') or 0) > (infer_sum.get('mean_ms') or 0) else '模型推理'}")
    if any(v for k, v in cls_hits.items() if k != "_其他"):
        log(f"  类别命中：{ {k: v for k, v in cls_hits.items() if v} }")
    else:
        log("  类别命中：本段素材未检出任何外墙缺陷"
            "（若素材是录屏/无关场景，0 检出是正确行为）")
    if n_saved:
        log(f"  截图 {n_saved} 张 → {frames_dir}")
    if save_video and video_path.exists():
        log(f"  回放视频 → {video_path}")
    if json_path is not None:
        log(f"  结构化记录 → {json_path}")
    log("=" * 78)
    log(f"边界声明：{result['caveat']}")

    return result


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="无人机外墙缺陷实时识别实验（独立模块，不改动既有源文件）")
    ap.add_argument("--config", default="rt_config.yaml",
                    help="配置文件路径（默认 rt_config.yaml，相对本目录）")
    ap.add_argument("--source", default=None,
                    help="视频源预设名，覆盖配置里的 source.preset"
                         "（synthetic / reclip / rtsp / camera）")
    # 三个最常临时调的覆盖项。其余一律走配置文件，避免参数漂移到主链路之外。
    ap.add_argument("--device", default=None, help="覆盖 infer.device，如 cpu / cuda:0")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="覆盖 capture.max_frames")
    ap.add_argument("--target-fps", type=float, default=None,
                    help="覆盖 capture.target_fps（0=逐帧全跑）")
    a = ap.parse_args()

    cfg = R.load_config(a.config)
    if a.device is not None:
        cfg.setdefault("infer", {})["device"] = a.device
    if a.max_frames is not None:
        cfg.setdefault("capture", {})["max_frames"] = a.max_frames
    if a.target_fps is not None:
        cfg.setdefault("capture", {})["target_fps"] = a.target_fps

    run(cfg, source_key=a.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
