# -*- coding: utf-8 -*-
r"""手机当摄像头 → 电脑实时识别 → 结果回传手机（真·连续实时，无需装 App）。

（2026-09-27 新增）

--------------------------------------------------------------------------
为什么需要本模块（与另两个入口的区别）
--------------------------------------------------------------------------
| 入口 | 手机摄像头 | 连续性 | 需装 App |
|---|---|---|---|
| app.py「📱 手机实时」 | 用（getUserMedia） | **逐张**（点一次快门一张） | 否 |
| live_stream.py | **不用**（读电脑端视频源） | 连续 MJPEG | 否 |
| **本模块 phone_live.py** | **用** | **连续**（自动持续推帧） | **否** |

**动机**：本机实测**没有摄像头**（`cv2.VideoCapture(0..3)` 全打不开，
且本机 OpenCV 的 ffmpeg 未编入 libavdevice）⇒ `live_stream.py` 的默认源
在这台机器上跑不起来。而手机摄像头是现成的、高清的、且正好在工地现场。
⇒ 让**手机浏览器**当采集端，把帧 POST 给电脑，电脑识别后叠加框，
再用 MJPEG 把结果画面发回手机。

--------------------------------------------------------------------------
数据流（三段，都在局域网内）
--------------------------------------------------------------------------
  ① 手机 getUserMedia 拿到摄像头画面
     ↓ 每 N 毫秒 canvas.toBlob('image/jpeg', q) → POST /push
  ② 电脑 FastAPI 收帧 → 后台线程识别（复用 pipeline.run_one）→ 叠加框
     ↓ 存入可重复读的最新帧（StreamState）
  ③ 手机 <img src="/stream"> 持续接收 MJPEG（multipart/x-mixed-replace）
     ⇒ 手机上看到的是**电脑识别后的画面**（框 + 判读摘要）

--------------------------------------------------------------------------
为什么是「推帧 + MJPEG 拉流」而不是 WebRTC
--------------------------------------------------------------------------
· WebRTC 需信令 + STUN/TURN + aiortc（本机未装，装它要动环境）。
· 本方案只依赖**已随 gradio 装好的** fastapi/uvicorn + cv2 + numpy，
  **零新增依赖**。
· 代价：画面是「采集→识别→回传」的往返，延迟约 0.3~1s，
  **不是**毫秒级体感。对「对着墙慢慢扫」这个使用场景完全够用。
  ⇒ 这是**刻意的取舍**：宁可延迟大一点，也不要新依赖。

--------------------------------------------------------------------------
能力边界（与所有其它入口一并声明，不得夸大）
--------------------------------------------------------------------------
1. **不提高毫米级判读能力** —— 判读门槛由 GSD 决定，与输入形式无关。
   手机离得远，GSD 必然差 ⇒ 实时流的价值是「**当场看出够不够格**」。
2. **逐帧标定不稳定**（实测极差可达 42%）⇒ 逐帧毫米值**不可单独采信**。
3. **本工程内无真实外墙巡检实拍** ⇒ 链路已在**合成帧**上自检，
   **不代表已在实拍上验证**。

--------------------------------------------------------------------------
★ 手机必须走 HTTPS（否则摄像头打不开）—— 2026-09-27 实测补
--------------------------------------------------------------------------
手机浏览器有硬性安全策略：`getUserMedia()`（调摄像头）**只在安全上下文**
可用 —— 即 `https://` 或 `localhost`。
用 `http://<局域网IP>:7862` 打开时，`navigator.mediaDevices` 直接是
`undefined` ⇒ 页面报「浏览器不支持摄像头（需 https 或 localhost）」。

⇒ 手机访问**必须**加 `--ssl`（默认走 7863）：
    D:\下载\python.exe phone_live.py --host=0.0.0.0 --port=7863 --ssl
证书先由 `logs/_make_selfsigned_cert.py` 生成（**必须把局域网 IP 放进 SAN**，
现代浏览器只看 SAN 不看 CN）。
自签名证书浏览器会提示「不安全」⇒ 点「高级 → 继续访问」即可，
这是**预期行为**，不是故障。

--------------------------------------------------------------------------
用法
--------------------------------------------------------------------------
  电脑端（手机用这个）：
    D:\下载\python.exe 06_deploy\phone_live.py --host=0.0.0.0 --port=7863 --ssl
  手机浏览器（同一 WiFi）打开  https://<电脑局域网IP>:7863
  点「开始」→ 授权摄像头（首次需点「高级 → 继续访问」）→ 对着墙慢慢扫。

  仅本机自测（不需要 https，因为 localhost 算安全上下文）：
    D:\下载\python.exe 06_deploy\phone_live.py --port=7862

  可选：
    --ssl             启用 HTTPS（手机会用；默认端口 7863）
    --cert-dir=<dir>  证书目录（默认 06_deploy/_cert）
    --interval=300    手机每 300ms 推一帧（默认；越小越流畅、越吃 CPU）
    --quality=0.6     手机 JPEG 质量（默认）
    --imgsz=640       推理分辨率
"""
from __future__ import annotations

import io
import sys
import time
import threading
import traceback
from pathlib import Path

# ★ 关键：`Request` 必须**在模块级导入**（不能只在 build_app() 里 import）。
#   原因（2026-09-27 实测定位）：本文件有 `from __future__ import annotations`，
#   所有注解变成**字符串**；FastAPI 解析 `async def push(request: Request)`
#   时要在**模块 globals** 里查 `"Request"`。若只在函数内 import，
#   闭包作用域查不到 ⇒ FastAPI 把 `request` 当成**查询参数**
#   ⇒ 收到原始 JPEG body 直接返回 **422 Unprocessable Content**。
#   实测对照：闭包内 import ⇒ query_params=['request']；模块级 import ⇒ []。
from fastapi import Request as _FastAPIRequest  # noqa: E402
from fastapi.responses import JSONResponse as _JSONResponse  # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "02_code"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 复用 live_stream 里**已验证**的 MJPEG 封装与状态容器（单一实现，不复制）
from live_stream import STATE, _mjpeg_part, _BOUNDARY, draw_overlay  # noqa: E402


# ---------------------------------------------------------------------------
# 识别线程：从「最新收到的原始帧」取帧 → 识别 → 叠加 → 回写 STATE
# ---------------------------------------------------------------------------
class PhoneAnalyzer(threading.Thread):
    """持续消费 `INBOX` 里最新的一帧原始 JPEG，识别后写进 `STATE`。

    为什么要**独立**于 HTTP 请求：识别耗时 60–200ms，
    若放在 `/push` 处理器里，**手机必须等识别完才能拿到响应**，
    推帧节奏会被识别速度绑死，而且一次慢帧会阻塞后续所有请求。
    ⇒ `/push` 只做一件事：把字节丢进 INBOX（微秒级返回）；
       识别在后台线程按自己的节奏跑。
    """

    def __init__(self, args: dict, *, out_w: int = 960, imgsz: int = 640):
        super().__init__(daemon=True)
        self.args = args
        self.out_w = out_w
        self.imgsz = imgsz
        self._lock = threading.Lock()
        self._pending: bytes | None = None
        self._seq_seen = 0
        self._stop = threading.Event()
        self.last_error: str | None = None
        self.n_received = 0
        self.n_inferred = 0

    # --- 供 HTTP 处理器调用：丢一帧进来（微秒级）---
    def submit(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._pending = jpeg_bytes
            self.n_received += 1

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

        last_res = None
        while not self._stop.is_set():
            with self._lock:
                buf = self._pending
                self._pending = None
            if buf is None:
                time.sleep(0.01)
                continue

            t0 = time.time()
            arr = np.frombuffer(buf, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.last_error = "收到无法解码的帧（不是合法 JPEG？）"
                continue

            sub = dict(self.args)
            sub["imgsz"] = self.imgsz
            try:
                res = run_one(Path("<phone>"), model, sub, image=frame)
                last_res = res
                self.n_inferred += 1
                self.last_error = None
            except Exception:                              # noqa: BLE001
                self.last_error = traceback.format_exc()
                res = last_res

            lat = (time.time() - t0) * 1000.0
            vis = draw_overlay(frame, res or {}, latency_ms=lat,
                               fps=STATE.fps, out_w=self.out_w)
            good, out = cv2.imencode(
                ".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if good:
                info = dict(res or {})
                info["_last_res"] = res
                STATE.put(out.tobytes(), info)


# ---------------------------------------------------------------------------
# 手机端页面（自带摄像头采集 + 推帧）
# ---------------------------------------------------------------------------
_PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>外墙缺陷 · 手机实时筛查</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  body { margin:0; background:#111; color:#eee;
         font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
  header { padding:12px 16px; background:#1b1b1b; border-bottom:1px solid #2a2a2a; }
  h1 { margin:0 0 4px; font-size:17px; font-weight:600; }
  .sub { color:#9aa0a6; font-size:12.5px; }
  main { padding:12px 16px 30px; max-width:1000px; margin:0 auto; }
  .wrap { position:relative; background:#000; border-radius:10px; overflow:hidden;
          border:1px solid #2a2a2a; min-height:200px; }
  img#s { display:block; width:100%; height:auto; }
  .placeholder { color:#666; padding:60px 16px; text-align:center; font-size:14px; }
  .bar { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }
  button.btn, a.btn { background:#2b6cb0; color:#fff; border:0; border-radius:8px;
      padding:12px 18px; font-size:15px; text-decoration:none; cursor:pointer;
      flex: 1 1 auto; text-align:center; }
  button.btn.stop { background:#a33; }
  a.btn.alt { background:#333; flex: 0 0 auto; }
  .note { background:#1b1b1b; border:1px solid #2a2a2a; border-radius:10px;
          padding:12px 14px; margin-top:14px; font-size:13.5px; color:#c9cdd1; }
  .note b { color:#fff; }
  .warn { border-color:#7a4a12; background:#241a0c; color:#f0c48a; }
  .stat { font-size:12.5px; color:#8a9096; margin-top:8px; }
  code { background:#000; padding:1px 5px; border-radius:4px; color:#8fd; }
</style></head><body>
<header>
  <h1>外墙缺陷 · 手机实时筛查</h1>
  <div class="sub">手机当摄像头 · 电脑实时识别 · 结果回传本页</div>
</header>
<main>
  <div class="wrap">
    <img id="s" alt="点「开始」后这里显示识别结果">
    <div class="placeholder" id="ph">点下面的「开始」并允许使用摄像头</div>
  </div>
  <div class="bar">
    <button class="btn" id="go">▶ 开始</button>
    <button class="btn stop" id="stop" style="display:none">⏸ 停止</button>
    <a class="btn alt" href="/snapshot" download="waiqiang_frame.jpg">存当前帧</a>
  </div>
  <div class="stat" id="stat">未开始</div>

  <div class="note">
    <b>怎么用</b>：点「开始」→ 允许摄像头 → <b>对着墙面慢慢扫</b>。
    画面上方那行字就是实时判读结果（检测框 + 耗时 + 帧率）。
  </div>
  <div class="note warn">
    <b>能力边界（必须说清）</b><br>
    1. 实时流<b>不提高</b>毫米级判读能力 —— 门槛由 GSD 决定，
    手机离得远时 GSD 必然差 ⇒ 实时流的价值是「<b>当场看出这张够不够格</b>」。<br>
    2. 逐帧标定不稳定 ⇒ <b>逐帧毫米值不可单独采信</b>。<br>
    3. 本工程内<b>无真实外墙巡检实拍</b> ⇒ 链路已在合成帧上自检，
    <b>不代表已在实拍上验证</b>。
  </div>
</main>
<script>
var video, canvas, ctx, timer=null, pushing=false, sent=0, t0=0;
var s = document.getElementById('s');
var ph = document.getElementById('ph');
var stat = document.getElementById('stat');
var go = document.getElementById('go');
var stopBtn = document.getElementById('stop');
var INTERVAL = __INTERVAL__;     // 推帧间隔 ms（由服务端注入）
var QUALITY  = __QUALITY__;      // JPEG 质量

function logStat(){
  var dt = (Date.now()-t0)/1000;
  stat.textContent = '已推送 ' + sent + ' 帧 · ' + (dt>0?(sent/dt).toFixed(1):'0') + ' 帧/秒';
}

function pickVideo(){
  // 优先环境/后置摄像头（对着墙拍，后置通常更清楚）
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return Promise.reject(new Error('浏览器不支持摄像头（需 https 或 localhost）'));
  }
  var tries = [
    { video: { facingMode: { ideal: 'environment' }, width: {ideal:1280} }, audio:false },
    { video: true, audio:false }
  ];
  return tries.reduce(function(p, c){
    return p.catch(function(){ return navigator.mediaDevices.getUserMedia(c); });
  }, Promise.reject());
}

go.onclick = function(){
  go.disabled = true;
  go.textContent = '启动中…';
  pickVideo().then(function(stream){
    video = document.createElement('video');
    video.srcObject = stream;
    video.setAttribute('playsinline','');   // iOS 必须，否则不内联播放
    video.muted = true;
    return video.play().then(function(){ return stream; });
  }).then(function(){
    canvas = document.createElement('canvas');
    ctx = canvas.getContext('2d');
    sent = 0; t0 = Date.now();
    ph.style.display='none';
    s.src = '/stream?t=' + Date.now();      // 开 MJPEG 结果流
    go.style.display='none'; stopBtn.style.display='';
    timer = setInterval(pushOne, INTERVAL);
    logStat();
  }).catch(function(e){
    ph.style.display='';
    ph.textContent = '摄像头打不开：' + e.message +
      '（请确认用 http 局域网地址打开、并已授权摄像头）';
    go.disabled = false; go.textContent = '▶ 重试';
  });
};

stopBtn.onclick = function(){
  if (timer) { clearInterval(timer); timer=null; }
  if (video && video.srcObject) {
    video.srcObject.getTracks().forEach(function(t){ t.stop(); });
  }
  s.src = '';
  stopBtn.style.display='none';
  go.style.display=''; go.disabled=false; go.textContent='▶ 开始';
  stat.textContent = '已停止（共推送 ' + sent + ' 帧）';
};

function pushOne(){
  if (pushing || !video || video.videoWidth === 0) return;
  pushing = true;
  // 限制长边到 960，减少推帧体积（识别端还会再缩）
  var w = video.videoWidth, h = video.videoHeight;
  var maxw = 960;
  if (w > maxw) { h = Math.round(h * maxw / w); w = maxw; }
  canvas.width = w; canvas.height = h;
  ctx.drawImage(video, 0, 0, w, h);
  canvas.toBlob(function(blob){
    if (!blob) { pushing=false; return; }
    fetch('/push', { method:'POST', body: blob,
                     headers: {'Content-Type':'image/jpeg'} })
      .then(function(){ sent++; logStat(); })
      .catch(function(){})
      .then(function(){ pushing=false; });
  }, 'image/jpeg', QUALITY);
}

// MJPEG 断流自动重连
s.onerror = function(){
  if (timer) setTimeout(function(){ s.src='/stream?t='+Date.now(); }, 1500);
};
</script>
</body></html>
"""


def build_app(args: dict, *, interval_ms: int = 300, quality: float = 0.6,
              out_w: int = 960, imgsz: int = 640):
    """构造「手机推帧」版 FastAPI 应用。"""
    # ⚠️ 这里**不要**再 `from fastapi import Request` —— 见文件顶部说明，
    #    局部导入会导致注解解析失败、/push 返 422。
    from fastapi import FastAPI
    from fastapi.responses import (HTMLResponse, StreamingResponse,
                                   JSONResponse, Response)

    analyzer = PhoneAnalyzer(args, out_w=out_w, imgsz=imgsz)
    analyzer.start()

    app = FastAPI(title="外墙缺陷 · 手机实时筛查")

    @app.get("/", response_class=HTMLResponse)
    def index():                                        # noqa: D401
        html = (_PAGE.replace("__INTERVAL__", str(int(interval_ms)))
                     .replace("__QUALITY__", str(float(quality))))
        return HTMLResponse(html)

    @app.post("/push")
    async def push(request: _FastAPIRequest):
        """只做一件事：把手机推来的 JPEG 丢进识别队列（微秒级返回）。

        ⚠️ 形参注解写法**不能随意改**：
        · 必须用**模块级**导入的 `_FastAPIRequest`（见文件顶部说明），
          写成函数内 import 的 `Request` 会让 FastAPI 把它当查询参数 ⇒ 422；
        · 必须用 `await request.body()` 自己读字节，
          **不要**改成 `body: bytes = Body(...)`（同样会被注解字符串化坑）。
        """
        body = await request.body()
        if not body:
            return JSONResponse({"ok": False, "msg": "空帧"}, status_code=400)
        analyzer.submit(body)
        return JSONResponse({"ok": True, "n": analyzer.n_received})

    @app.get("/status")
    def status():
        _, seq, _info = STATE.get()
        return JSONResponse({
            "mode": "phone-push",
            "seq": seq,
            "fps": round(STATE.fps, 2),
            "has_frame": STATE.get()[0] is not None,
            "n_received": analyzer.n_received,
            "n_inferred": analyzer.n_inferred,
            "analyzer_error": analyzer.last_error,
        })

    @app.get("/snapshot")
    def snapshot():
        jpeg, _, _ = STATE.get()
        if jpeg is None:
            return JSONResponse({"ok": False, "msg": "还没有帧"}, status_code=503)
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
                    if idle > 20:      # 20s 无新帧 ⇒ 结束，让前端重连
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
    port = int(_arg("port", "7862"))
    interval_ms = int(_arg("interval", "300"))
    quality = float(_arg("quality", "0.6"))
    imgsz = int(_arg("imgsz", "640"))
    out_w = int(_arg("out-width", "960"))
    # ★ HTTPS：手机浏览器**只在安全上下文**（https 或 localhost）
    #   才允许 getUserMedia 调摄像头；http://<局域网IP> 会被直接拦掉。
    #   ⇒ 手机访问**必须**开 https（默认端口 7863，避免与 http 版混淆）。
    use_ssl = ("--ssl" in sys.argv[1:]
               or _arg("ssl", "").lower() in ("1", "true", "yes", "on"))
    cert_dir = Path(_arg("cert-dir", str(_HERE / "_cert")))
    ssl_cert = Path(_arg("ssl-cert", str(cert_dir / "localhost.pem")))
    ssl_key = Path(_arg("ssl-key", str(cert_dir / "localhost-key.pem")))
    if "--ssl" in sys.argv[1:] and "--port=" not in " ".join(sys.argv):
        port = 7863                      # 开 ssl 且未显式指定端口 ⇒ 默认 7863

    try:
        import uvicorn
    except ImportError:
        print("!! 未安装 uvicorn（应随 gradio 一起装好）。")
        return 1

    if use_ssl:
        if not (ssl_cert.exists() and ssl_key.exists()):
            print("!! 找不到证书:")
            print("   证书:", ssl_cert)
            print("   私钥:", ssl_key)
            print("   请先生成：D:\\下载\\python.exe logs\\_make_selfsigned_cert.py")
            return 1
        # 预检：证书能否被加载（避免起了进程才发现证书坏）
        try:
            import ssl as _ssl
            ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile=str(ssl_cert), keyfile=str(ssl_key))
        except Exception as e:                          # noqa: BLE001
            print("!! 证书加载失败:", e)
            return 1

    from common import default_weight
    w = default_weight("v11s640")
    base = _HERE.parent
    args = {
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

    scheme = "https" if use_ssl else "http"
    print("=" * 72)
    print("外墙缺陷 · 手机实时筛查（手机推帧 + MJPEG 回传）")
    print("=" * 72)
    print("权重      :", args["weight_path"] or "（未找到，请先训练）")
    print("监听      : %s://%s:%d" % (scheme, host, port))
    print("推帧间隔  : %d ms | JPEG 质量: %.2f | imgsz: %d"
          % (interval_ms, quality, imgsz))
    if use_ssl:
        print("HTTPS     : 已启用（证书 %s）" % ssl_cert.name)
        print("   ⚠️ 自签名证书 ⇒ 手机首次打开会提示「不安全」，"
              "点「高级 → 继续访问」即可")
    if host not in ("127.0.0.1", "localhost"):
        print("📱 手机打开 : %s://<本机局域网IP>:%d" % (scheme, port))
        print("   查本机 IP : ipconfig | findstr IPv4")
        if not use_ssl:
            print("   ⚠️ **手机上不给摄像头的根因就是这里**："
                  "http 明文页面无法调摄像头。")
            print("      ⇒ 手机访问请改用 HTTPS："
                  "加 `--ssl`（默认端口 %d）" % 7863)
        print("   ⚠️ 打不开：① 先确认绑定（netstat -ano | findstr :%d 应为 0.0.0.0）"
              % port)
        print("             ② 再查 Windows 防火墙是否放行 %d 端口" % port)
    else:
        print("（默认只监听本机；手机访问请加 --host=0.0.0.0）")
    print("边界      : 不提高毫米级判读能力；无实拍流 ⇒ 未在实拍上验证")
    print("=" * 72)

    app, _an = build_app(args, interval_ms=interval_ms, quality=quality,
                         out_w=out_w, imgsz=imgsz)
    if use_ssl:
        uvicorn.run(app, host=host, port=port, log_level="warning",
                    ssl_certfile=str(ssl_cert), ssl_keyfile=str(ssl_key))
    else:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
