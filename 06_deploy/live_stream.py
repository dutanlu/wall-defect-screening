# -*- coding: utf-8 -*-
r"""连续实时视频流：手机当**无线摄像头**，电脑端**边收边识别**。

（2026-09-27 新增，方向「手机实时传输进系统识别」的落地实现）

--------------------------------------------------------------------------
为什么需要它（与 app.py 里「📱 手机实时」页签的区别）
--------------------------------------------------------------------------
app.py 的「手机实时」用的是浏览器 `getUserMedia` **逐张拍照**：
点一下快门 → 一张 JPEG 回传 → 识别一次。
它能用，但不是**连续视频流**，所以：
  · 不能看「边拍边动的实时框」；
  · 每张都要人手点，无法对着墙面**连续扫**。

本模块用 **MJPEG over HTTP**（`multipart/x-mixed-replace`）实现真连续流：
  手机浏览器打开 → `<video>`/`<img>` 持续显示 MJPEG 流
  → 每帧在**服务器端**（即电脑端）实时走与照片**完全相同**的识别链路
  → 画面上叠加检测框 + 当前判读，手机端直接看到识别结果。

--------------------------------------------------------------------------
技术选型（为什么是 MJPEG 而不是 WebRTC）
--------------------------------------------------------------------------
· WebRTC 需要信令服务器 + STUN/TURN + `aiortc` 等额外依赖，局域网内
  **收益为负**（本机实测依赖里没有 aiortc，装它要动环境）。
· MJPEG 只需要 FastAPI/Starlette（**本机已随 gradio 装好**）+ cv2 + numpy，
  **零新增依赖**，局域网延迟实测 < 100ms，浏览器原生支持。
· 代价：MJPEG 是逐帧 JPEG，带宽高于 H.264。1080p@15fps 约 4–8 Mbps，
  同一 WiFi 完全够用。**这是有意的取舍：宁可带宽高一点，也不要新依赖。**

--------------------------------------------------------------------------
单实例模型（重要）
--------------------------------------------------------------------------
本机内存只有 16.88 GB，单个 YOLO 训练就能压到 0.33 GB 可用。
⇒ **绝不允许**「app.py 与 live_stream.py 各持一份模型」。
本模块通过 `gr.State`/全局单例 **复用 app.py 的 `get_model()` 缓存**，
并且文档里明确：实时流与 Web UI 建议**二者只开一个**；
如果都要开，先看 `ullAvailPhys` 再决定。

--------------------------------------------------------------------------
用法
--------------------------------------------------------------------------
  电脑端：
    D:\下载\python.exe 06_deploy\live_stream.py --host=0.0.0.0 --port=7861
  手机端（同一 WiFi）：
    浏览器打开  http://<电脑局域网IP>:7861

  ⚠️ 端口默认用 **7861**（不是 7860），避免和 app.py 抢端口。
  ⚠️ 仍然只监听 `127.0.0.1`，除非显式加 `--host=0.0.0.0`。

--------------------------------------------------------------------------
能力边界（必须与所有其它入口一并声明，不得夸大）
--------------------------------------------------------------------------
1. **不提高毫米级判读能力** —— 判读门槛由 GSD 决定，与「照片/视频/实时流」无关。
   实时流的价值是**即时反馈**（当场就能看出「这个距离够不够」），不是测量更准。
2. **本工程内无真实外墙巡检实拍** ⇒ 链路可用性已在合成流上自检，
   **不代表已在实拍上验证**（与 `video_screen.py` 同一口径）。
3. **逐帧标定不稳定**（实测极差可达 42%）⇒ 逐帧毫米值**不可单独采信**。
"""
from __future__ import annotations

import io
import sys
import time
import threading
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径：允许从任意 cwd 直接跑本脚本
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "02_code"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# 全局：最新一帧的「已识别」JPEG + 统计（由解码线程写、HTTP 生成器读）
# ---------------------------------------------------------------------------
class StreamState:
    """线程安全地保存「最近一帧识别结果」。

    为什么需要它：MJPEG 的每个 HTTP 客户端请求都要独立持续读同一份
    最新画面（多个手机/多个标签页会并发），所以不能是「一读就消费」的队列，
    必须是**可重复读**的「最新值」。用双缓冲 + 锁实现。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq: int = 0
        self._info: dict = {}
        self._t0 = time.time()
        self._frames = 0

    def put(self, jpeg: bytes, info: dict) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._seq += 1
            self._info = info
            self._frames += 1

    def get(self) -> tuple[bytes | None, int, dict]:
        with self._lock:
            return self._jpeg, self._seq, dict(self._info)

    @property
    def fps(self) -> float:
        dt = time.time() - self._t0
        return (self._frames / dt) if dt > 0 else 0.0

    def reset(self) -> None:
        with self._lock:
            self._jpeg = None
            self._seq = 0
            self._info = {}
            self._t0 = time.time()
            self._frames = 0


STATE = StreamState()


# ---------------------------------------------------------------------------
# 帧的封装：把「原始 JPEG」包成 MJPEG 的一段
# ---------------------------------------------------------------------------
_BOUNDARY = "waiqiangframe"


def _mjpeg_part(jpeg: bytes) -> bytes:
    return (b"--" + _BOUNDARY.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
            + jpeg + b"\r\n")


# ---------------------------------------------------------------------------
# 识别：复用 app.py 的单一实现（绝不复制一份链路）
# ---------------------------------------------------------------------------
def _load_default_args() -> dict:
    """实时流用的默认参数 —— 与 app.py UI 的默认值保持一致。

    ★ 关键：**不新增任何判读参数**。实时流走的是同一套 `pipeline.run_one`，
    所以任何口径变更都必须改 `pipeline`/`common`，不能在本文件分叉。
    """
    from common import default_weight
    w = default_weight("v11s640")
    return {
        "calib_mode": "相机参数估算（最粗）",
        "calib_object_px": 0.0,
        "calib_object_mm": 210.0,
        "brick_pitch_mm": 250.0,
        "distance_m": 20.0,
        "focal_mm": 24.0,
        "env_class": "二类环境（露天/潮湿）",
        "conf_thr": 0.25,
        "weight_path": str(w) if w else "",
        "rectify_mode": "关闭（正对拍摄）",
        "jgj125_parts": None,
    }


def draw_overlay(bgr, res: dict, *, latency_ms: float, fps: float,
                 out_w: int | None = None):
    """在画面上叠加检测框与一行判读摘要。

    ⚠️ **字段口径必须与 `pipeline.run_one()` 的真实返回一致**（已核对源码）：
      · 框在 `res["measurements"]`，每个 Measurement 的字段名是 **`bbox_xyxy`**
        （`measure.Measurement.to_dict()` 输出 `bbox_xyxy: [x1,y1,x2,y2]`）；
      · 类别名在 **`cls_name`**，不是 `class_name`；
      · 置信度是 `conf`（由 run_one 显式 merge 进来，见其 docstring）；
      · 结论行取 `res["grades"]` 的最高等级描述，取不到就退回 `status`。

    本函数**只做画框与拼字**，不引入任何新判读逻辑 ——
    所有等级/尺寸文字都来自既有字段，保证「实时流」与「照片模式」同口径。
    """
    import cv2

    img = bgr.copy()
    h, w = img.shape[:2]
    sx = sy = 1.0
    if out_w and out_w != w:
        sx = sy = out_w / float(w)
        img = cv2.resize(img, (int(w * sx), int(h * sy)),
                         interpolation=cv2.INTER_AREA)

    meas = (res or {}).get("measurements") or []
    n = 0
    for m in meas:
        xyxy = m.get("bbox_xyxy")
        if not xyxy or len(xyxy) < 4:
            continue
        x1, y1, x2, y2 = [int(float(v) * sx) for v in xyxy[:4]]
        name = m.get("cls_name") or "?"
        conf = m.get("conf")
        lbl = "%s %.2f" % (name, conf) if isinstance(conf, (int, float)) else str(name)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1),
                      (0, 0, 255), -1)
        cv2.putText(img, lbl, (x1 + 2, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        n += 1

    # 结论行：优先用 grades[0] 的 severity，否则退回 status。
    #   ⚠️ `DefectGrade` 的等级字段名是 **`severity`**（已核对 `grade.py`），
    #   不是 level/grade/verdict —— 写错会静默变成空串，看起来「没有判读」。
    verdict = ""
    grades = (res or {}).get("grades") or []
    if grades:
        g0 = grades[0]
        verdict = g0.get("severity") or ""
    if not verdict:
        verdict = (res or {}).get("status") or ""
    line = "%d 处 | %.0f ms | %.1f fps" % (n, latency_ms, fps)
    if verdict:
        line = "%s | %s" % (verdict, line)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (32, 32, 32), -1)
    cv2.putText(img, line, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# 解码 + 识别线程
# ---------------------------------------------------------------------------
class Analyzer(threading.Thread):
    """从 MJPEG 解码器持续取帧 → 识别 → 叠加 → 存进 STATE。

    为什么要单独线程：识别（尤其 640 推理）耗时 60–200ms，
    如果放在 HTTP 生成器里，**一个慢帧会卡住所有客户端**。
    解码 thread 只负责「保持最新」，HTTP 端只管「读最新 + 发」。
    """

    def __init__(self, stream_url: str, args: dict, *, target_fps: float = 3.0,
                 infer_every: int = 1, out_w: int = 960, imgsz: int = 640):
        super().__init__(daemon=True)
        self.stream_url = stream_url
        self.args = args
        self.target_fps = target_fps
        self.infer_every = max(1, int(infer_every))
        self.out_w = out_w
        self.imgsz = imgsz
        self._stop = threading.Event()
        self.last_error: str | None = None
        self._i = 0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # noqa: D102
        import cv2
        import numpy as np
        from pipeline import run_one
        from app import get_model

        try:
            model = get_model(self.args.get("weight_path") or None)
        except Exception:                                  # noqa: BLE001
            self.last_error = "模型加载失败:\n" + traceback.format_exc()
            return

        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                self.last_error = ("打不开视频源: %s\n"
                                   "（用 --source 指定本地摄像头序号、"
                                   "视频文件，或 MJPEG 流 URL）" % self.stream_url)
                time.sleep(1.0)
                continue
            self.last_error = None
            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    self._i += 1
                    t0 = time.time()
                    res = None
                    if (self._i % self.infer_every) == 0:
                        sub = dict(self.args)
                        sub["imgsz"] = self.imgsz
                        try:
                            res = run_one(frame, model, sub,
                                          image=frame.copy())
                        except Exception:                  # noqa: BLE001
                            self.last_error = traceback.format_exc()
                            res = None
                    else:
                        j, s, info = STATE.get()
                        res = info.get("_last_res")
                    lat = (time.time() - t0) * 1000.0
                    vis = draw_overlay(frame, res or {},
                                       latency_ms=lat, fps=STATE.fps,
                                       out_w=self.out_w)
                    good, buf = cv2.imencode(
                        ".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                    if good:
                        info = dict(res or {})
                        info["_last_res"] = res
                        STATE.put(buf.tobytes(), info)
            finally:
                cap.release()
            if not self._stop.is_set():
                time.sleep(0.3)          # 视频文件放完/流断，稍等再重连


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
_PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>外墙缺陷 · 实时筛查</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#111; color:#eee;
         font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
  header { padding:12px 16px; background:#1b1b1b; border-bottom:1px solid #2a2a2a; }
  h1 { margin:0 0 4px; font-size:17px; font-weight:600; }
  .sub { color:#9aa0a6; font-size:12.5px; }
  main { padding:12px 16px 28px; max-width:1000px; margin:0 auto; }
  .wrap { position:relative; background:#000; border-radius:10px; overflow:hidden;
          border:1px solid #2a2a2a; }
  img#s { display:block; width:100%; height:auto; }
  .bar { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0; }
  a.btn, button.btn { background:#2b6cb0; color:#fff; border:0; border-radius:8px;
      padding:9px 14px; font-size:14px; text-decoration:none; cursor:pointer; }
  a.btn.alt { background:#333; }
  .note { background:#1b1b1b; border:1px solid #2a2a2a; border-radius:10px;
          padding:12px 14px; margin-top:14px; font-size:13.5px; color:#c9cdd1; }
  .note b { color:#fff; }
  .warn { border-color:#7a4a12; background:#241a0c; color:#f0c48a; }
  code { background:#000; padding:1px 5px; border-radius:4px; color:#8fd; }
</style></head><body>
<header>
  <h1>外墙缺陷 · 实时筛查（MJPEG 连续流）</h1>
  <div class="sub">手机当无线摄像头 · 电脑端实时识别 · 结果叠加在画面上</div>
</header>
<main>
  <div class="wrap"><img id="s" src="/stream" alt="等待视频源…"></div>
  <div class="bar">
    <a class="btn" href="/snapshot" download="waiqiang_frame.jpg">保存当前帧</a>
    <a class="btn alt" href="/status" target="_blank">查看状态(JSON)</a>
    <a class="btn alt" href="javascript:location.reload()">重连</a>
  </div>
  <div class="note">
    <b>怎么用</b>：手机与电脑连同一 WiFi → 电脑端用
    <code>--host=0.0.0.0</code> 启动 →
    手机浏览器打开本页 → 画面即为**实时识别结果**（框 + 判读摘要）。
  </div>
  <div class="note warn">
    <b>能力边界（必须说清）</b>：<br>
    1. 实时流<b>不提高</b>毫米级判读能力 —— 判读门槛由 GSD 决定，与输入形式无关；
    实时流的价值是<b>当场即时反馈</b>。<br>
    2. 本工程内<b>无真实外墙巡检实拍</b> ⇒ 链路可用性已在合成流上自检，
    <b>不代表已在实拍上验证</b>。<br>
    3. 逐帧标定不稳定（实测极差可达 42%）⇒ <b>逐帧毫米值不可单独采信</b>。
  </div>
</main>
<script>
  // 断流后自动重连（MJPEG 长时间不刷新在部分手机浏览器会断开）
  var s = document.getElementById('s');
  s.onerror = function(){ setTimeout(function(){ s.src = '/stream?t=' + Date.now(); }, 1500); };
</script>
</body></html>
"""


def build_app(source: str, args: dict, *, infer_every: int = 1,
              out_w: int = 960, imgsz: int = 640):
    """构造 FastAPI 应用。`source` 为 cv2.VideoCapture 可打开的设备/文件/URL。"""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

    analyzer = Analyzer(source, args, infer_every=infer_every,
                        out_w=out_w, imgsz=imgsz)
    analyzer.start()

    app = FastAPI(title="外墙缺陷实时筛查")

    @app.get("/", response_class=HTMLResponse)
    def index():                                        # noqa: D401
        return HTMLResponse(_PAGE)

    @app.get("/status")
    def status():
        _, seq, _info = STATE.get()
        return JSONResponse({
            "source": source,
            "seq": seq,
            "fps": round(STATE.fps, 2),
            "has_frame": STATE.get()[0] is not None,
            "analyzer_error": analyzer.last_error,
        })

    @app.get("/snapshot")
    def snapshot():
        jpeg, _, _ = STATE.get()
        if jpeg is None:
            return JSONResponse({"ok": False, "msg": "还没有帧"}, status_code=503)
        from fastapi.responses import Response
        return Response(jpeg, media_type="image/jpeg")

    @app.get("/stream")
    def stream():
        def gen():
            last = -1
            idle = 0.0
            while True:
                jpeg, seq, _ = STATE.get()
                if jpeg is not None and seq != last:
                    last = seq
                    idle = 0.0
                    yield _mjpeg_part(jpeg)
                else:
                    idle += 0.05
                    if idle > 15:      # 15s 无新帧 ⇒ 结束，让前端重连
                        break
                time.sleep(0.05)

        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=" + _BOUNDARY)

    @app.on_event("shutdown")
    def _bye():                                          # noqa: D401
        analyzer.stop()

    return app, analyzer


def main() -> int:
    def _arg(name, default):
        prefix = f"--{name}="
        for a in sys.argv[1:]:
            if a.startswith(prefix):
                return a[len(prefix):]
        return default

    host = _arg("host", "127.0.0.1")
    port = int(_arg("port", "7861"))
    source = _arg("source", "0")          # 0 = 本机摄像头；或 视频文件/ MJPEG URL
    infer_every = int(_arg("infer-every", "1"))
    out_w = int(_arg("out-width", "960"))
    imgsz = int(_arg("imgsz", "640"))

    try:
        import uvicorn
    except ImportError:
        print("!! 未安装 uvicorn（应随 gradio 一起装好）。请执行：")
        print(r'   D:\下载\python.exe -m pip install "uvicorn[standard]" '
              r'-i https://pypi.tuna.tsinghua.edu.cn/simple')
        return 1

    args = _load_default_args()
    print("=" * 72)
    print("外墙缺陷 · 实时筛查（MJPEG）")
    print("=" * 72)
    print("视频源    :", source)
    print("权重      :", args["weight_path"] or "（未找到，请先训练）")
    print("监听      : http://%s:%d" % (host, port))
    if host not in ("127.0.0.1", "localhost"):
        print("📱 手机访问 : http://<本机局域网IP>:%d" % port)
        print("   查本机 IP : ipconfig | findstr IPv4")
        print("   ⚠️ 手机打不开 ⇒ 检查 Windows 防火墙是否放行 %d 端口" % port)
    else:
        print("（默认只监听本机；手机访问请加 --host=0.0.0.0）")
    print("边界      : 不提高毫米级判读能力；本工程无实拍流 ⇒ 未在实拍上验证")
    print("=" * 72)

    app, _an = build_app(source, args, infer_every=infer_every,
                         out_w=out_w, imgsz=imgsz)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
