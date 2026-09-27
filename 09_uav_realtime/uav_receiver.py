# -*- coding: utf-8 -*-
"""uav_receiver.py —— 无人机 WiFi 图传接收端（UDP + 自适应解码）。

归属：09_uav_realtime/ 独立实验目录。不修改任何既有源文件。

================================= 它解决什么 =================================
无人机（WiFi_CAM 类公版方案）把画面以 UDP 推给手机 App。要在电脑上拿到帧，
必须自己监听 UDP 端口、把分片的包重组成完整帧、再解码成图像。

================================= 关键设计 =================================
【1】不硬编码端口与编码，全部从探测结果 / 配置读取。
     探测脚本 uav_probe.py 会实测出端口与流类型，本模块据此工作。

【2】自适应解码 —— 因为实测前不知道是 MJPEG 还是 H.264：
      - MJPEG：包里直接含 JPEG 帧（找 FF D8 ... FF D9），逐帧 imdecode 即可
      - H.264：裸流，需交给解码器（本机 FFmpeg 可用，cv2 有 avcodec 61.19）
     先按「包里是否出现 JPEG SOI/EOI」判定；H.264 走内存管道喂 ffmpeg。

【3】帧重组要抗乱序/丢包（UDP 天然会乱序丢包）：
     MJPEG 的容错策略 = 以 FF D8 为起点、FF D9 为终点就地切帧，
     不依赖包序号；只要收到完整一帧的字节就能出图，丢包只损失那一帧。
     这是最鲁棒的做法，因为很多公版图传的包头格式并无公开规范。

【4】统一输出接口 next_frame() -> (ok, frame_bgr, meta)
     与 rt_infer.py 的循环对接，meta 里带接收耗时，便于分层计时。

================================== 安全 ==================================
  ★ 只接收、只做模式切换魔数（协议要求），不发送任何飞行控制指令。
  ★ 模式魔数只在明确连上无人机热点时发送（由 --send-magic 显式开启）。
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent

# 协议魔数（公开资料：天衡 WiFi_CAM / 同族机型逆向经验）
# 上电默认 2.4GHz 遥控器模式，不发这个就不推视频流。
MAGIC_ENTER_APP_MODE = bytes([0x42, 0x76])
MAGIC_EXIT_APP_MODE = bytes([0x42, 0x77])

# 流类型
STREAM_UNKNOWN = "unknown"
STREAM_MJPEG = "mjpeg"
STREAM_H264 = "h264"


# ============================================================================
# MJPEG 帧切分器：不依赖包边界，从字节流里切出完整 JPEG
# ============================================================================
class MjpegDeframer:
    """从连续字节流中切出完整 JPEG 帧。

    做法：维护一个缓冲；找 FF D8（SOI）作起点、FF D9（EOI）作终点。
    为什么不用包序号重组：公开资料显示这类图传包头无统一规范，
    按序号重组容易因为丢包而永久卡死；按 JPEG 标记切分则天然容错 ——
    丢一包只损失一帧，下一帧的 SOI 会立刻重新同步。
    """

    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"

    def __init__(self, max_buf: int = 8 * 1024 * 1024):
        self.buf = bytearray()
        self.max_buf = max_buf
        self.dropped_bytes = 0     # 因找不到 SOI 而丢弃的字节数
        self.frames_out = 0

    def feed(self, data: bytes) -> list[bytes]:
        """喂入原始字节，返回本次切出的完整 JPEG 列表。"""
        self.buf.extend(data)
        out: list[bytes] = []

        while True:
            s = self.buf.find(self.SOI)
            if s < 0:
                # 没有 SOI：只保留最后 1 字节（可能是被切断的 FF）
                if len(self.buf) > 1:
                    self.dropped_bytes += len(self.buf) - 1
                    del self.buf[:-1]
                break
            if s > 0:
                # SOI 之前是上个帧的残尾，丢掉
                self.dropped_bytes += s
                del self.buf[:s]
            e = self.buf.find(self.EOI, 2)
            if e < 0:
                # 帧还没收完；防止缓冲无限增长
                if len(self.buf) > self.max_buf:
                    self.dropped_bytes += len(self.buf)
                    self.buf.clear()
                break
            end = e + 2
            out.append(bytes(self.buf[:end]))
            self.frames_out += 1
            del self.buf[:end]

        # 整体上限保护
        if len(self.buf) > self.max_buf:
            self.dropped_bytes += len(self.buf)
            self.buf.clear()
        return out


def sniff_stream_type(data: bytes) -> str:
    """根据首个包的字节判断流类型。"""
    if MjpegDeframer.SOI in data[:64] or MjpegDeframer.SOI in data:
        return STREAM_MJPEG
    # H.264 起始码 00 00 00 01 或 00 00 01
    if data[:4] == b"\x00\x00\x00\x01" or data[:3] == b"\x00\x00\x01":
        return STREAM_H264
    # RTP 封装：版本号在头 2 bit，通常是 0x80
    if len(data) >= 12 and (data[0] >> 6) == 2:
        return "rtp?"
    return STREAM_UNKNOWN


# ============================================================================
# H.264 解码：把裸流喂给 ffmpeg，从 stdout 读回 BGR 帧
# ============================================================================
class H264Decoder:
    """通过 ffmpeg 子进程解码 H.264 裸流。

    选用 ffmpeg 而非 cv2.VideoCapture：cv2 无法直接吃"内存里的裸流片段"，
    而 ffmpeg 支持 stdin 管道输入，天然适配流式场景。
    ffmpeg 二进制来源优先 imageio-ffmpeg（自带完整 libx264/解码器），
    退化到 PATH 里的 ffmpeg。
    """

    def __init__(self, width: int = 0, height: int = 0):
        self.proc = None
        self.width, self.height = width, height
        self._ff = self._find_ffmpeg()

    @staticmethod
    def _find_ffmpeg() -> str:
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"

    def start(self, width: int, height: int) -> bool:
        self.width, self.height = int(width), int(height)
        cmd = [
            self._ff, "-hide_banner", "-loglevel", "error",
            "-f", "h264", "-i", "pipe:0",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=10 ** 7)
            return True
        except Exception:
            self.proc = None
            return False

    def feed_and_read(self, data: bytes) -> list[np.ndarray]:
        """喂入裸流并尽量读回已解码的帧（非阻塞式，读尽即返回）。"""
        if self.proc is None or self.proc.stdin is None:
            return []
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except Exception:
            return []
        return self._drain()

    def _drain(self) -> list[np.ndarray]:
        """非阻塞地把 stdout 里已就绪的完整帧读出来。"""
        frames: list[np.ndarray] = []
        if self.proc is None or self.proc.stdout is None:
            return frames
        need = self.width * self.height * 3
        if need <= 0:
            return frames
        import os
        import select
        # Windows 上 select 不支持管道，改用「有数据就非阻塞读」
        while True:
            try:
                # 用 os.read 做一次性读取，避免 read() 阻塞等满 need 字节
                r, _, _ = ([], [], [])
                try:
                    import msvcrt  # noqa: F401
                    avail = self.proc.stdout.peek(1)  # 触发缓冲
                    if not avail:
                        break
                except Exception:
                    pass
                chunk = self.proc.stdout.read1(need) if hasattr(
                    self.proc.stdout, "read1") else None
                if not chunk:
                    break
                if len(chunk) < need:
                    # 不足一帧，先缓存（下次补）
                    self._pending = getattr(self, "_pending", b"") + chunk
                    if len(self._pending) < need:
                        break
                    chunk = self._pending[:need]
                    self._pending = self._pending[need:]
                img = np.frombuffer(chunk[:need], dtype=np.uint8)
                img = img.reshape((self.height, self.width, 3))
                frames.append(img.copy())
            except Exception:
                break
        return frames

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None


# ============================================================================
# UDP 接收线程：把包塞进队列，主线程消费（避免主线程被 recv 阻塞）
# ============================================================================
class UdpReceiver:
    def __init__(self, port: int, bind_ip: str = "", queue_size: int = 512):
        self.port = port
        self.bind_ip = bind_ip
        self.q: deque = deque(maxlen=queue_size)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.pkts = 0
        self.bytes = 0
        self.pkts_dropped_queue = 0
        self.first_packet_time = 0.0
        self.last_packet_time = 0.0

    def start(self) -> bool:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 大缓冲，降低高码率下的丢包
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except Exception:
            pass
        try:
            self._sock.bind((self.bind_ip, self.port))
        except OSError:
            return False
        self._sock.settimeout(0.3)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            self.pkts += 1
            self.bytes += len(data)
            now = time.perf_counter()
            if not self.first_packet_time:
                self.first_packet_time = now
            self.last_packet_time = now
            if len(self.q) >= self.q.maxlen:
                self.pkts_dropped_queue += 1
            self.q.append(data)

    def get(self, timeout: float = 0.1) -> bytes | None:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self.q:
                return self.q.popleft()
            time.sleep(0.002)
        return None

    def empty(self) -> bool:
        return not self.q

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass


# ============================================================================
# 统一入口：把上面的零件组装成 next_frame()
# ============================================================================
class UavStream:
    """无人机图传接收器：UDP 收包 → 自适应解码 → 产出 BGR 帧。

    用法：
        st = UavStream(port=8080, drone_ip="192.168.4.1", send_magic=True)
        st.start()
        while True:
            ok, frame, meta = st.next_frame()
            ...
        st.close()
    """

    def __init__(self, port: int, drone_ip: str = "", bind_ip: str = "",
                 send_magic: bool = False, stream_type: str = "auto",
                 width: int = 640, height: int = 480, verbose: bool = True):
        self.port = port
        self.drone_ip = drone_ip
        self.bind_ip = bind_ip
        self.send_magic = send_magic
        self.force_type = stream_type
        self.width, self.height = width, height
        self.verbose = verbose

        self.rx: UdpReceiver | None = None
        self.deframer = MjpegDeframer()
        self.h264: H264Decoder | None = None
        self.stream_type = (stream_type if stream_type != "auto" else STREAM_UNKNOWN)
        self.frames_decoded = 0
        self.decode_fail = 0
        self._type_probe_buf = bytearray()
        # 未识别流类型的出现次数，用于日志限频（避免垃圾流刷屏）
        self._unknown_seen = 0

    # ---------------- 生命周期 ----------------
    def start(self) -> bool:
        self.rx = UdpReceiver(self.port, self.bind_ip)
        if not self.rx.start():
            if self.verbose:
                print(f"[接收] ✗ 无法绑定 UDP {self.bind_ip or '0.0.0.0'}:{self.port}"
                      f"（端口被占用？换个端口或用管理员权限）")
            return False
        if self.verbose:
            print(f"[接收] ✓ 已在 UDP 端口 {self.port} 上监听")

        if self.send_magic and self.drone_ip:
            # ★ 仅发送 2 字节模式魔数，用于触发推流；不发任何飞行指令
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                for p in {self.port, 8090, 8080}:
                    try:
                        s.sendto(MAGIC_ENTER_APP_MODE, (self.drone_ip, p))
                    except Exception:
                        pass
                s.close()
                if self.verbose:
                    print(f"[模式] 已向 {self.drone_ip} 发送 App 模式魔数 "
                          f"{MAGIC_ENTER_APP_MODE.hex()}（仅此 2 字节，"
                          f"不含任何飞行指令）")
            except Exception as e:
                if self.verbose:
                    print(f"[模式] 魔数发送失败：{e}")
        return True

    def _auto_detect(self, data: bytes) -> None:
        """累积若干包后判定流类型。"""
        self._type_probe_buf.extend(data[:4096])
        if len(self._type_probe_buf) < 4096 and self.rx and self.rx.pkts < 8:
            return
        t = sniff_stream_type(bytes(self._type_probe_buf))
        if t == STREAM_MJPEG:
            self.stream_type = STREAM_MJPEG
            if self.verbose:
                print(f"[识别] 流类型 = MJPEG（检测到 JPEG SOI/EOI 标记）")
        elif t == STREAM_H264:
            self.stream_type = STREAM_H264
            if self.verbose:
                print(f"[识别] 流类型 = H.264 裸流（检测到起始码）"
                      f"，将经 ffmpeg 解码（需要正确分辨率 {self.width}x{self.height}）")
            self.h264 = H264Decoder()
            self.h264.start(self.width, self.height)
        else:
            # ★ 限频打印：垃圾流/未知流下 probe 会被调用几十次，
            #   不限频会把日志刷爆（实测 60 包刷了 60 行，同类提示无信息增量）。
            self._unknown_seen += 1
            if self.verbose and (self._unknown_seen <= 3 or
                                 self._unknown_seen % 50 == 0):
                extra = "" if self._unknown_seen <= 3 else \
                    f"（同类型已第 {self._unknown_seen} 次）"
                print(f"[识别] 流类型暂未确定{extra}"
                      f"（首包 {data[:16].hex()}），继续观察…")

    # ---------------- 取帧 ----------------
    def next_frame(self, timeout: float = 2.0):
        """返回 (ok, frame_bgr, meta)。

        meta 含 recv_ms（收包+重组+解码耗时），供上层分层计时。
        """
        t0 = time.perf_counter()
        if self.rx is None:
            return False, None, {"error": "not_started"}

        while time.perf_counter() - t0 < timeout:
            data = self.rx.get(timeout=0.05)
            if data is None:
                continue

            # 首次自动识别流类型
            if self.stream_type == STREAM_UNKNOWN:
                self._auto_detect(data)
                if self.stream_type == STREAM_UNKNOWN:
                    continue

            if self.stream_type == STREAM_MJPEG:
                frames = self.deframer.feed(data)
                if not frames:
                    continue
                # 只取最新的一帧（实时场景丢弃积压，避免延迟增长）
                jpg = frames[-1]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                recv_ms = (time.perf_counter() - t0) * 1000.0
                if img is None:
                    self.decode_fail += 1
                    # 退化重试：剥掉可能的私有包头再试
                    i = jpg.find(b"\xff\xd8")
                    if i > 0:
                        img = cv2.imdecode(
                            np.frombuffer(jpg[i:], np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                self.frames_decoded += 1
                self.width, self.height = img.shape[1], img.shape[0]
                return True, img, {
                    "stream_type": STREAM_MJPEG, "recv_ms": recv_ms,
                    "jpg_bytes": len(jpg),
                    "deframer_dropped": self.deframer.dropped_bytes,
                }

            elif self.stream_type == STREAM_H264:
                if self.h264 is None:
                    self.h264 = H264Decoder()
                    self.h264.start(self.width, self.height)
                imgs = self.h264.feed_and_read(data)
                if not imgs:
                    continue
                img = imgs[-1]
                recv_ms = (time.perf_counter() - t0) * 1000.0
                self.frames_decoded += 1
                return True, img, {
                    "stream_type": STREAM_H264, "recv_ms": recv_ms,
                    "h264_frames_in_batch": len(imgs),
                }
        return False, None, {"error": "timeout", "rx_pkts": self.rx.pkts}

    def stats(self) -> dict:
        r = self.rx
        return {
            "port": self.port, "stream_type": self.stream_type,
            "rx_packets": r.pkts if r else 0,
            "rx_bytes": r.bytes if r else 0,
            "queue_dropped": r.pkts_dropped_queue if r else 0,
            "frames_decoded": self.frames_decoded,
            "decode_fail": self.decode_fail,
            "deframer_dropped_bytes": self.deframer.dropped_bytes,
        }

    def close(self) -> None:
        if self.send_magic and self.drone_ip:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.sendto(MAGIC_EXIT_APP_MODE, (self.drone_ip, self.port))
                s.close()
                if self.verbose:
                    print("[模式] 已发送还原魔数（切回遥控器模式）")
            except Exception:
                pass
        if self.h264:
            self.h264.close()
        if self.rx:
            self.rx.close()


# ============================================================================
# VideoCapture 兼容适配器
# ============================================================================
class UavCapture:
    """把 UavStream 包装成 `cv2.VideoCapture` 的接口子集。

    ★ 为什么要这层适配（重要设计决策）：
      rt_infer.py 的主循环是按 cv2.VideoCapture 的契约写的（cap.read() 返回
      (ok, frame)、cap.release()、cap.get(CAP_PROP_*) 取元信息）。
      如果为无人机源单独写一套取帧分支，主循环里就会出现两条并行路径 ——
      这正是本项目反复出问题的地方（同一件事两处实现 ⇒ 口径漂移）。
      ⇒ 只在**源打开**这一层做适配，主循环保持单一实现，零改动复用。
        这是"新增源"与"新增逻辑"的区别：前者是替换，后者是分叉。
    """

    def __init__(self, stream: UavStream):
        self.stream = stream
        self._last_ok = False
        self._frames = 0

    # ---- cv2.VideoCapture 接口子集 ----
    def read(self):
        ok, frame, meta = self.stream.next_frame(timeout=2.0)
        self._last_ok = ok
        if ok:
            self._frames += 1
        return ok, frame

    def isOpened(self) -> bool:
        return self.stream.rx is not None and not self.stream.rx.empty() \
            or self._frames > 0 or self.stream.rx is not None

    def release(self) -> None:
        self.stream.close()

    def get(self, prop: int):
        """返回元信息。无人机流的真实分辨率要到解出第一帧才知道。"""
        import cv2 as _cv2
        if prop == _cv2.CAP_PROP_FPS:
            return 30.0                       # 保守值；真实值靠实测统计
        if prop == _cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.stream.width)
        if prop == _cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.stream.height)
        if prop == _cv2.CAP_PROP_FRAME_COUNT:
            return 0.0                        # 实时流无总帧数
        return 0.0

    def set(self, prop: int, value):
        return False                          # 实时源不支持设置

    # ---- 额外的取流统计（供 rt_infer 记录）----
    def stream_stats(self) -> dict:
        return self.stream.stats()


def open_uav_stream(cfg_src: dict, verbose: bool = True) -> UavCapture | None:
    """按配置打开无人机 WiFi 图传源，返回 UavCapture（失败返回 None）。"""
    port = int(cfg_src.get("port", 8080))
    drone_ip = str(cfg_src.get("drone_ip", "") or "")
    stream_type = str(cfg_src.get("stream_type", "auto"))
    width = int(cfg_src.get("width", 640) or 640)
    height = int(cfg_src.get("height", 480) or 480)
    send_magic = bool(cfg_src.get("send_magic", False))

    st = UavStream(port=port, drone_ip=drone_ip, send_magic=send_magic,
                   stream_type=stream_type, width=width, height=height,
                   verbose=verbose)
    if not st.start():
        return None
    return UavCapture(st)


# ============================================================================
# CLI：独立冒烟测试（不需要无人机 —— 用本机回环造 MJPEG 流验证解码链路）
# ============================================================================
def _selftest(port: int, seconds: float) -> int:
    """自测：在本机回环上发一段 MJPEG 到目标端口，验证接收+重组+解码。

    意义：把"接收端能不能正确解出帧"与"无人机能不能推流"这两件事**解耦**。
    前者可以用回环 100% 验证，后者才需要实机。这样实机联调时若失败，
    能立刻判断是接收端 bug 还是无人机侧问题。
    """
    print("=" * 74)
    print("接收端自测（本机回环，不需要无人机）")
    print("=" * 74)

    # 造测试帧：纯色 + 序号文字
    frames_jpg = []
    for i in range(12):
        img = np.zeros((240, 320, 3), np.uint8)
        img[:] = (30 + i * 15, 60, 120)
        cv2.putText(img, f"FRAME {i}", (40, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            frames_jpg.append(buf.tobytes())
    print(f"  造了 {len(frames_jpg)} 个测试 JPEG 帧")

    st = UavStream(port=port, send_magic=False, stream_type="auto", verbose=True)
    if not st.start():
        return 1

    # 后台发送（模拟无人机：分片发送，制造跨包切帧的场景）
    def _sender():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        blob = b"".join(frames_jpg)
        # 故意按 1400 字节分片 —— 这样 JPEG 会被切到多个 UDP 包里，
        # 正是真实图传的情形，能检验 deframer 的跨包重组能力
        for off in range(0, len(blob), 1400):
            s.sendto(blob[off:off + 1400], ("127.0.0.1", port))
            time.sleep(0.005)
        s.close()

    th = threading.Thread(target=_sender, daemon=True)
    th.start()

    got, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, frame, meta = st.next_frame(timeout=0.5)
        if not ok:
            continue
        got += 1
        if got <= 3:
            print(f"  收到帧 #{got}: {frame.shape}  {meta}")

    st.close()
    print()
    print(f"  统计：{st.stats()}")
    ok_all = got >= len(frames_jpg) - 2   # 容忍边界丢 1~2 帧（分片切割）
    print(f"  期望收到约 {len(frames_jpg)} 帧，实际 {got} 帧 "
          f"⇒ {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="无人机 WiFi 图传接收端（UDP + 自适应 MJPEG/H.264 解码）")
    ap.add_argument("--selftest", action="store_true",
                    help="本机回环自测（不需要无人机），验证解码链路")
    ap.add_argument("--selftest-port", type=int, default=18080,
                    help="自测用的回环端口（默认 18080）")
    ap.add_argument("--selftest-sec", type=float, default=6.0)
    ap.add_argument("--port", type=int, default=8080,
                    help="监听端口（无人机推流端口，由 uav_probe 实测得出）")
    ap.add_argument("--drone-ip", default="", help="无人机 IP，如 192.168.4.1")
    ap.add_argument("--send-magic", action="store_true",
                    help="发送 App 模式魔数触发推流（仅 2 字节，无飞行指令）")
    ap.add_argument("--stream-type", default="auto",
                    choices=["auto", "mjpeg", "h264"])
    ap.add_argument("--width", type=int, default=640,
                    help="H.264 解码需要正确分辨率；MJPEG 可忽略")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--show", action="store_true", help="弹出窗口预览（按 q 退出）")
    a = ap.parse_args()

    if a.selftest:
        return _selftest(a.selftest_port, a.selftest_sec)

    st = UavStream(port=a.port, drone_ip=a.drone_ip, send_magic=a.send_magic,
                   stream_type=a.stream_type, width=a.width, height=a.height)
    if not st.start():
        return 1
    print("按 Ctrl+C 停止。")
    try:
        n, t0 = 0, time.perf_counter()
        while True:
            ok, frame, meta = st.next_frame(timeout=1.0)
            if not ok:
                print(f"  等待帧…  {meta}")
                continue
            n += 1
            if n % 10 == 1:
                el = time.perf_counter() - t0
                print(f"  帧 {n}  {frame.shape}  接收 {meta.get('recv_ms', 0):.1f}ms"
                      f"  平均 {n/el:.2f} FPS  {meta.get('stream_type')}")
            if a.show:
                cv2.imshow("UAV stream (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，停止。")
    finally:
        st.close()
        if a.show:
            cv2.destroyAllWindows()
        print(f"最终统计：{st.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
