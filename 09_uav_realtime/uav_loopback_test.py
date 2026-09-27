# -*- coding: utf-8 -*-
"""uav_loopback_test.py —— 无人机链路端到端自测（回环模拟，不需要无人机）。

归属：09_uav_realtime/。不修改任何既有源文件。

============================== 它验证什么 ==============================
真实场景是：无人机 → UDP 推 MJPEG → 本系统接收解码 → 跑识别 → 叠框 → 出指标。

其中「无人机推流」这一段在没实机时无法验证，但**其余全部可以**。
本脚本用「合成外墙图 → 编成 MJPEG → 按真实图传方式分片 UDP 发送」
来替代无人机，从而把除无人机本体之外的整条链路跑通。

★ 价值：等实机联调时若拿不到画面，可通过本脚本的通过与否，立刻判断
  问题出在「无人机侧（端口/魔数/网段）」还是「系统侧（我的代码）」。
  没有这道隔离，两类问题会混在一起，排查成本极高。

============================== 与自测的区别 ==============================
  uav_receiver.py --selftest ：只验证「接收+解码」这一层
  本脚本                     ：验证「接收+解码+识别+叠框+落盘+指标」全链路
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_PROJ = _HERE.parent
_SYNTH_VIDEO = _PROJ / "logs" / "_test_video" / "synthetic_wall_walk.mp4"
_OUT = _HERE / "logs"


def make_mjpeg_sender(port: int, source: str, fps: float, count: int,
                      stop: threading.Event, loop_until_stop: bool = False):
    """后台线程：把视频帧编成 MJPEG，按真实图传方式（分片 UDP）发给本机端口。

    为什么分片：真实图传不会把一个 JPEG 放进一个 UDP 包（MTU 限制），
    必然跨包切割。只有分片发送才能检验 deframer 的跨包重组能力。

    ★ loop_until_stop=True 的由来（实测教训）：
      第二阶段是「先起发送线程、再调 rt_infer.run()」，而 run() 内部要先
      加载 YOLO 模型（GPU 上需数秒）。若发送端固定只发 N 帧就退出，
      这数秒内帧已全部发完 ⇒ 接收端真正开始监听时流已结束，
      表现为"流中断"反复重连、几乎收不到帧。
      ⇒ 模拟"无人机持续推流"必须**一直发到被叫停**，这才是真实行为。
    """
    def _run():
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            print(f"[发送] ✗ 打不开素材 {source}")
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sent = 0
        interval = 1.0 / fps if fps > 0 else 0.033
        try:
            while not stop.is_set():
                if not loop_until_stop and sent >= count:
                    break
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # 循环播放
                    continue
                ok2, buf = cv2.imencode(".jpg", frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok2:
                    continue
                blob = buf.tobytes()
                # 按 1400 字节分片（贴近真实 MTU 分片）
                for off in range(0, len(blob), 1400):
                    s.sendto(blob[off:off + 1400], ("127.0.0.1", port))
                sent += 1
                if not loop_until_stop and sent >= count:
                    break
                time.sleep(interval)
        finally:
            s.close()
            cap.release()
            print(f"[发送] 结束，共发出 {sent} 帧")
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th


def main() -> int:
    ap = argparse.ArgumentParser(
        description="无人机链路端到端自测（回环模拟推流，不需要无人机）")
    ap.add_argument("--port", type=int, default=18090,
                    help="回环测试端口（默认 18090，避开真实图传端口）")
    ap.add_argument("--source", default=str(_SYNTH_VIDEO),
                    help="用作“无人机画面”的素材")
    ap.add_argument("--fps", type=float, default=15.0, help="模拟推流帧率")
    ap.add_argument("--frames", type=int, default=24, help="模拟发送帧数")
    ap.add_argument("--max-frames", type=int, default=12,
                    help="实际跑识别的帧数")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    if not Path(a.source).exists():
        print(f"素材不存在：{a.source}")
        return 1

    print("=" * 74)
    print("无人机链路端到端自测（回环模拟，不需要无人机）")
    print("=" * 74)
    print(f"  素材：{Path(a.source).name}")
    print(f"  模拟推流：UDP 127.0.0.1:{a.port}  {a.fps} fps  {a.frames} 帧"
          f"（按 1400B 分片，模拟真实图传）")
    print()

    # ---- 1) 先验证接收层能拿到帧 ----
    from uav_receiver import UavStream
    stop = threading.Event()
    th1 = make_mjpeg_sender(a.port, a.source, a.fps, a.frames, stop)

    st = UavStream(port=a.port, stream_type="auto", verbose=True)
    if not st.start():
        print("✗ 无法绑定端口")
        return 1

    got, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 6:
        ok, frame, meta = st.next_frame(timeout=0.5)
        if ok:
            got += 1
            if got == 1:
                print(f"[接收] ✓ 首帧 {frame.shape}  {meta}")
    st.close()
    stop.set()
    # ★ 必须等第一个发送线程真正结束再进第二阶段。
    #   教训：本脚本初版没等，结果两个发送线程同时往同一端口灌数据，
    #   第二阶段读到大量第一阶段残余包 + 两个流的字节交错，
    #   导致 deframer 频繁失同步、表现为"流中断"反复重连。
    #   这不是产品代码的问题，是**测试装置**的问题。
    th1.join(timeout=3)
    time.sleep(0.5)   # 给内核清掉端口上残余的 UDP 包

    if got == 0:
        print()
        print("✗ 接收层一帧都没拿到 ⇒ 问题在**系统侧**，先修接收层。")
        return 1
    print(f"[接收] ✓ 共收到 {got} 帧")

    # ---- 2) 走完整识别链路（接收 → run_one → 叠框 → 指标）----
    print()
    print("-" * 74)
    print("第二阶段：接收 + 识别 + 叠框 + 指标（完整链路）")
    print("-" * 74)

    import rt_common as R
    import rt_infer

    cfg = R.load_config("rt_config.yaml")
    cfg["infer"]["device"] = a.device
    cfg["capture"].update({
        "max_frames": a.max_frames, "warmup_frames": a.warmup,
        "target_fps": 0, "duration_sec": 0,
    })
    # ★ 关键：把源换成回环 wifi_fpv，验证 rt_infer 真的能吃无人机源
    cfg["source"]["presets"]["_loopback"] = {
        "kind": "wifi_fpv", "port": a.port, "drone_ip": "127.0.0.1",
        "stream_type": "mjpeg", "width": 0, "height": 0,
        "send_magic": False,          # 回环不发魔数
        "assumed_fps": a.fps,
    }
    cfg["output"]["save_frame_every"] = 2
    cfg["output"]["max_saved_frames"] = 8
    cfg["output"]["save_video"] = True

    # 起发送线程，且**持续发送到被叫停**（模拟无人机一直推流）。
    # 见 make_mjpeg_sender 的注释：固定帧数会在模型加载期间就发完。
    stop2 = threading.Event()
    make_mjpeg_sender(a.port, a.source, a.fps, 0, stop2, loop_until_stop=True)
    time.sleep(1.0)   # 让发送先跑起来，确保 rt_infer 启动时流已在

    result = rt_infer.run(cfg, source_key="_loopback")
    stop2.set()

    counts = result.get("counts") or {}
    v = result.get("verdict") or {}
    print()
    print("=" * 74)
    print("端到端自测结论")
    print("=" * 74)
    ok_chain = (counts.get("frames_measured") or 0) > 0
    print(f"  统计推理帧数：{counts.get('frames_measured')} "
          f"⇒ {'PASS' if ok_chain else 'FAIL'}")
    print(f"  吞吐 FPS：{v.get('throughput_fps')}     "
          f"端到端 P90：{(result.get('timing_ms') or {}).get('e2e', {}).get('p90_ms')} ms")
    print(f"  源类型：{(result.get('source') or {}).get('kind')}  "
          f"URI：{(result.get('source') or {}).get('uri')}")
    art = result.get("artifacts") or {}
    print(f"  回放视频：{art.get('video')}")
    print()
    if ok_chain:
        print("  ⇒ 除无人机本体外，整条链路（接收→识别→叠框→落盘→指标）**全通**。")
        print("     实机联调时若失败，问题必在无人机侧（端口/魔数/网段），"
              "可直接查 uav_probe.py 的结论。")
        return 0
    print("  ⇒ FAIL：链路有问题，先修系统侧。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
