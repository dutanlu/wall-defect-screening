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
     电脑端打开 /mirror 也读**同一路**的 STATE ⇒ 两端画面逐帧一致
     （多机时**每台手机一路**：`?client=<id>`，见 PhoneAnalyzer 的 per-client 槽）

--------------------------------------------------------------------------
手机端 UI：全屏相机式（2026-09-27 追加）
--------------------------------------------------------------------------
手机页面做成**铺满视口的取景框**（`position:fixed; inset:0` + `object-fit:contain`），
而不是「页头 + 卡片 + 说明文」的文档式布局。要点：

- `viewport-fit=cover` + `env(safe-area-inset-*)` ⇒ 刘海/圆角/手势条不挡画面；
- 控件**叠在画面之上**（HUD + 底部圆钮），点画面即**沉浸模式**（隐藏全部控件）；
- 底部四个圆钮：切换前/后摄像头、停止、存当前帧、全屏；
- `object-fit:contain` 而不是 `cover` —— **判读靠像素，绝不能裁掉边缘**
  （裁掉的那部分可能正是一条裂缝）。

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
    --interval=50     手机每 50ms 推一帧（默认；越小越流畅、越吃 CPU）
                      实测 50ms ⇒ 端到端 19.8 fps，且服务端收推比 ≈1.0
                      （每一帧都真推理了）。**天花板是摄像头源帧率
                      （实测 20.09 fps）**，再降不会更快。
                      弱手机/WiFi 若「丢 N」增长 ⇒ 用 --interval=66 回退。
    --quality=0.6     手机 JPEG 质量（默认）
    --imgsz=640       推理分辨率
    --push-maxw=720   手机推帧长边（默认 720；同时决定服务端叠加图长边）
    --out-width=720   服务端叠加图长边（默认随 --push-maxw 推导，一般不用改）
"""
from __future__ import annotations

import io
import sys
import time
import threading
import traceback
from pathlib import Path
# ★ 必须**模块级**：本文件有 `from __future__ import annotations`，注解是字符串，
#   FastAPI 在**模块 globals** 里 eval 它们（同 `_FastAPIRequest` 那条教训）。
from typing import Optional

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
# ★★ 本文件有两套**不同的目录布局**，路径常量必须同时兼容（2026-09-27 踩到）：
#   源目录（开发时）  : `<root>/06_deploy/phone_live.py`  → 代码在 `<root>/02_code/`
#   交付包（给评委）  : `<pack>/deploy/phone_live.py`     → 代码在 `<pack>/code/`
#   （交付包把 `01_data`→`dataset`、`02_code`→`code`、`06_deploy`→`deploy` 等
#     做了重命名，见 `logs/_copy_delivery_pack.py`。）
#   ⇒ 只写 `_ROOT / "02_code"` 时**源目录能跑、包内 ModuleNotFoundError**，
#     而且**一致性核对（`_pack_vs_src.py`）看不见它** —— 那是「同名文件 md5 比对」，
#     路径解析错不改字节内容 ⇒ 永远 PASS。
#     这正是本项目 P0 纪律「交付包必须在**包自身目录内**跑一次入口」的由来。
#   ⇒ 逐个候选目录探测，谁存在用谁；都不存在时保留原名（让报错信息更直白）。
_CAND_CODE_DIRS = ("02_code", "code")
_CODE_DIR = None
for _n in _CAND_CODE_DIRS:
    if (_ROOT / _n).is_dir():
        _CODE_DIR = _ROOT / _n
        break
if _CODE_DIR is None:
    _CODE_DIR = _ROOT / _CAND_CODE_DIRS[0]
for _p in (str(_CODE_DIR), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 复用 live_stream 里**已验证**的 MJPEG 封装与状态容器（单一实现，不复制）
# ★ 多路改造：还要 `REG`（状态注册表）。`STATE` 保留 —— 它是**默认路**的代理，
#   旧代码与旧 URL 靠它继续工作。
from live_stream import (REG, STATE, _mjpeg_part,  # noqa: E402
                         _BOUNDARY, draw_overlay)


# ---------------------------------------------------------------------------
# ★ 推帧分辨率默认值：**单一真源**（2026-09-27 第二版，「还能再顺畅一点」）
# ---------------------------------------------------------------------------
# 血泪教训复用（同 DEFAULT_INTERVAL_MS）：默认值**禁止写两处**。
# 这里涉及**三个**必须同步的量，任何一个漏改都会让现象变得无法解释：
#
#   ① `_PAGE` 里 JS 的 `MAXW`（手机端 canvas 缩到多小再编码）
#   ② `build_app(out_w=...)` 的**默认值**（服务端叠加图再缩到多小）
#   ③ `main()` 从命令行读默认时的回退值
#
# ⇒ ① 由服务端注入 `__MAXW__`，不写字面量；②③ 都引用下面这个常量。
#
# 取值依据（logs/_sweep_push_resolution.py 实测，本机 RTX 5070 Ti + v11s640，
# 测试集 01_data/dataset/images/test 抽 6 张，每档 3 轮取中位）；
# 判据**不是**「越快越好」，而是「**检出数不掉**」—— 本项目主指标是
# 「有没有漏检」，帧率只是体感。实测表（长边 / 推理p50 / 上限fps / 检出数）：
#
#   长边    推理p50    p95     上限fps   检出总数   均置信度   逐类(crack/moss)
#   960    27.4     73.1     36.54     71       0.572     33 / 4     ← 旧默认
#   800    20.5     30.3     48.79     70       0.587     33 / 4
#   720    17.8     24.7     56.05     72       0.569     33 / 4     ← 采用
#   640    15.4     21.0     65.06     71       0.573     33 / 3     ← moss 掉 1
#   544    14.3     19.5     69.71     74       0.557     33 / 4
#
# 相对 960 基线：720 档 **推理时间 −34.8%、检出 +1.4%（72 vs 71）、
#   逐类一个都没少**；640 档虽再快 9 个百分点，但 `moss` 从 4 → 3
#   （测试集仅 4 个样本，掉 1 个就是 −25%）⇒ **保守起见取 720**。
# p95 也一并改善（73.1 → 24.7 ms，−66%）⇒ 卡顿尖刺同步消失，
# 这比 p50 更重要：体感差往往来自 p95 而不是均值。
DEFAULT_PUSH_MAXW = 720


def _derive_out_w(push_maxw: int = DEFAULT_PUSH_MAXW) -> int:
    """由推帧长边推导「服务端叠加图」的输出长边。

    ★ 为什么不用手填：`out_w` 若**大于**收到帧的长边，`draw_overlay` 会
      把图**放大**（`cv2.resize` 到 out_w）⇒ 白白多花一次重采样、
      推给手机的体积还更大，纯亏。二者本就是同一件事的两个环节，
      必须**联动**，所以由 `push_maxw` 推导而不是再开一个命令行参数。
    """
    return int(push_maxw)


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

    def __init__(self, args: dict, *, out_w: int = DEFAULT_PUSH_MAXW,
                 imgsz: int = 640):
        super().__init__(daemon=True)
        self.args = args
        self.out_w = out_w
        self.imgsz = imgsz
        self._lock = threading.Lock()
        # ★ 2026-09-27 多路改造：**单槽 → per-client 槽**。
        #   原实现 `self._pending: bytes | None = None` 只有**一个**槽，
        #   两台手机同时推流时 A 的帧会被 B 的帧**直接覆盖** ⇒ 画面互相插帧。
        #   现在每个 client 一个槽，且**每路只留最新 1 帧**（新覆盖旧）
        #   ⇒ 内存**有界**：慢的 client 不会把内存堆爆。
        self._pending: dict[str, bytes] = {}
        # 每路的上次接收时刻（限流用）与计数
        self._last_t: dict[str, float] = {}
        self._per: dict[str, dict] = {}
        self.n_throttled = 0
        self._seq_seen = 0
        self._stop = threading.Event()
        self.last_error: str | None = None
        self.n_received = 0
        self.n_inferred = 0

    # ★ 同一路两次推帧的**最小间隔**（秒）。40ms ⇒ 25 fps 封顶。
    #   取值理由：单路实测约 20 fps 是上限，25 fps 已留余量；
    #   超过它的推帧一律**丢弃并计数**，避免某台手机狂推把 GPU 占满、
    #   把别人的路饿死。**丢弃不是静默的** —— `n_throttled` 暴露在 /status。
    MIN_GAP_S = 0.040

    @staticmethod
    def norm_client(client_id) -> str:
        """归一 client_id：None / '' / 纯空白 ⇒ 默认路（与 live_stream 同口径）。"""
        return (str(client_id).strip() if client_id else "") or REG.DEFAULT

    def _slot(self, cid: str) -> dict:
        """取（必要时建）某一路的计数块。"""
        d = self._per.get(cid)
        if d is None:
            d = {"n_received": 0, "n_inferred": 0, "n_throttled": 0,
                 "last_error": None}
            self._per[cid] = d
        return d

    def per_client(self) -> dict:
        """逐路计数（供 /status 暴露）。键与 live_stream.REG.clients() 对齐。"""
        with self._lock:
            items = {k: dict(v) for k, v in self._per.items()}
        # 默认路统一显示为 '<default>'，与 live_stream.per_client() 一致
        return {("<default>" if k == REG.DEFAULT else k): v
                for k, v in items.items()}

    # --- 供 HTTP 处理器调用：丢一帧进来（微秒级）---
    def submit(self, jpeg_bytes: bytes, client_id=None) -> None:
        cid = self.norm_client(client_id)
        now = time.time()
        with self._lock:
            d = self._slot(cid)
            last = self._last_t.get(cid)
            # ★ 限流：太密就**丢这一帧**并计数（不静默）
            if last is not None and (now - last) < self.MIN_GAP_S:
                d["n_throttled"] += 1
                self.n_throttled += 1
                return False
            self._last_t[cid] = now
            self._pending[cid] = jpeg_bytes
            d["n_received"] += 1
            self.n_received += 1
            return True

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

        last_res_by: dict[str, dict | None] = {}
        rr = 0
        while not self._stop.is_set():
            # ---- ★ 轮转取帧：**单线程**串行用 GPU，各路公平轮流 ----
            #   不一次取全部（那会争抢 GPU），也不固定顺序（会饿死后面的路）。
            with self._lock:
                ready = [k for k, v in self._pending.items() if v is not None]
                if not ready:
                    cid, buf = None, None
                else:
                    cid = ready[rr % len(ready)]
                    rr += 1
                    buf = self._pending[cid]
                    self._pending[cid] = None
            if buf is None:
                time.sleep(0.01)
                continue

            t0 = time.time()
            arr = np.frombuffer(buf, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                msg = "收到无法解码的帧（不是合法 JPEG？）"
                self.last_error = msg
                with self._lock:
                    self._slot(cid)["last_error"] = msg
                continue

            sub = dict(self.args)
            sub["imgsz"] = self.imgsz
            last_res = last_res_by.get(cid)
            try:
                res = run_one(Path("<phone>"), model, sub, image=frame)
                last_res = res
                last_res_by[cid] = res
                with self._lock:
                    self._slot(cid)["n_inferred"] += 1
                    self._slot(cid)["last_error"] = None
                self.n_inferred += 1
                self.last_error = None
            except Exception:                              # noqa: BLE001
                self.last_error = traceback.format_exc()
                with self._lock:
                    self._slot(cid)["last_error"] = self.last_error
                res = last_res

            lat = (time.time() - t0) * 1000.0
            # ★ 写进**这一路自己**的状态槽（原来固定写 STATE ⇒ 就是插帧的根因）
            st = REG.get_state(cid)
            vis = draw_overlay(frame, res or {}, latency_ms=lat,
                               fps=st.fps, out_w=self.out_w)
            good, out = cv2.imencode(
                ".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if good:
                info = dict(res or {})
                info["_last_res"] = res
                info["_client"] = cid or "<default>"
                st.put(out.tobytes(), info)


# ---------------------------------------------------------------------------
# 手机端页面（自带摄像头采集 + 推帧）
# ---------------------------------------------------------------------------
_PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<!-- ★ 全屏相机式：viewport 必须带 viewport-fit=cover，否则刘海/圆角会留黑边 -->
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1,
      user-scalable=no, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#000000">
<title>外墙缺陷 · 手机实时筛查</title>
<style>
  :root {
    color-scheme: dark;
    --safe-t: env(safe-area-inset-top, 0px);
    --safe-b: env(safe-area-inset-bottom, 0px);
    --safe-l: env(safe-area-inset-left, 0px);
    --safe-r: env(safe-area-inset-right, 0px);
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html, body { height:100%; margin:0; overflow:hidden; background:#000; }
  body { color:#eee; -webkit-user-select:none; user-select:none;
         font:15px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }

  /* ---------- 全屏取景框：铺满整个视口 ---------- */
  #stage { position:fixed; inset:0; background:#000; overflow:hidden;
           touch-action:none; }
  #s { position:absolute; inset:0; width:100%; height:100%;
       object-fit:contain; display:block; }

  /* 启动前的占位（同时也是「开始」大按钮） */
  #splash { position:absolute; inset:0; display:flex; flex-direction:column;
            align-items:center; justify-content:center; gap:18px; text-align:center;
            padding:calc(var(--safe-t) + 24px) 24px calc(var(--safe-b) + 24px); }
  #splash h1 { margin:0; font-size:19px; font-weight:600; }
  #splash p { margin:0; color:#9aa0a6; font-size:13.5px; max-width:420px; }
  #bigstart { width:96px; height:96px; border-radius:50%; border:0;
      background:#2b6cb0; color:#fff; font-size:34px; line-height:1;
      box-shadow:0 6px 24px rgba(43,108,176,.45); cursor:pointer;
      display:flex; align-items:center; justify-content:center; }
  #bigstart:active { transform:scale(.94); }
  #splash .hint { color:#6b7076; font-size:12.5px; }
  #err { color:#f0a0a0; font-size:13px; max-width:440px; white-space:pre-wrap; }

  /* ---------- 顶部 HUD（叠在画面上） ---------- */
  #hud { position:absolute; left:0; right:0; top:0;
         padding:calc(var(--safe-t) + 8px) calc(var(--safe-r) + 12px) 14px
                 calc(var(--safe-l) + 12px);
         background:linear-gradient(180deg, rgba(0,0,0,.72), rgba(0,0,0,0));
         pointer-events:none; transition:opacity .25s;
         /* ★ 默认隐藏：HUD 讲的是「识别中」的运行时状态，
            在还没开始的启动页上显示「正在启动…」会与 splash 打架（实测截图可见）。
            开始推帧后由 JS 加 .on 显示。 */
         opacity:0; }
  #hud.on { opacity:1; }
  #hud .row { display:flex; align-items:center; gap:8px; }
  #dot { width:8px; height:8px; border-radius:50%; background:#e33;
         box-shadow:0 0 8px #e33; animation:blink 1.4s infinite; flex:0 0 auto; }
  @keyframes blink { 50% { opacity:.25; } }
  #hud .t { font-size:13px; color:#dfe3e6; }
  #hud .meta { margin-top:4px; font-size:11.5px; color:#9aa0a6;
               font-family:ui-monospace,Menlo,Consolas,monospace; }

  /* ---------- 底部控制条 ---------- */
  #bar { position:absolute; left:0; right:0; bottom:0;
         padding:16px calc(var(--safe-r) + 16px) calc(var(--safe-b) + 18px)
                 calc(var(--safe-l) + 16px);
         display:flex; align-items:center; justify-content:space-around; gap:10px;
         background:linear-gradient(0deg, rgba(0,0,0,.78), rgba(0,0,0,0));
         transition:opacity .25s; }
  .cbtn { width:52px; height:52px; border-radius:50%; border:1px solid #444;
      background:rgba(30,30,30,.82); color:#eee; font-size:20px; cursor:pointer;
      display:flex; align-items:center; justify-content:center; flex:0 0 auto; }
  .cbtn:active { transform:scale(.92); }
  #stopBtn { background:#a33; border-color:#a33; }
  a.cbtn { text-decoration:none; }
  #stat2 { position:absolute; bottom:calc(var(--safe-b) + 78px); left:0; right:0;
           text-align:center; font-size:11.5px; color:#8a9096;
           font-family:ui-monospace,Menlo,Consolas,monospace;
           pointer-events:none; transition:opacity .25s; }

  /* 沉浸模式：点一下画面就隐藏所有控件（像相机那样）。
     ★ 用「更高特异性 + !important 兜底」而不是只写 body.immersive：
       因为 #hud.on / #bar 自身也设了 opacity，特异性打架时会漏隐藏（实测踩过）。 */
  body.immersive #hud, body.immersive #bar, body.immersive #stat2,
  body.immersive #hud.on {
    opacity:0 !important; pointer-events:none !important;
    visibility:hidden !important;
  }

  /* 横屏时底部条可以更靠边一点 */
  @media (orientation:landscape) {
    #bar { padding-bottom:calc(var(--safe-b) + 10px); }
    #stat2 { bottom:calc(var(--safe-b) + 66px); }
  }

  /* 电脑镜像页的入口（只在开始后显示） */
  #mirrorTip { position:absolute; top:calc(var(--safe-t) + 52px);
      left:calc(var(--safe-l) + 12px); right:calc(var(--safe-r) + 12px);
      font-size:11.5px; color:#8fd; font-family:ui-monospace,Menlo,Consolas,monospace;
      pointer-events:auto; }
  #mirrorTip a { color:#8fd; }
</style></head><body>

<div id="stage">
  <img id="s" alt="识别结果">
  <div id="hud">
    <div class="row"><span id="dot"></span>
      <span class="t" id="title">正在启动…</span></div>
    <div class="meta" id="meta"></div>
    <div id="mirrorTip"></div>
  </div>
  <div id="splash">
    <h1>外墙缺陷 · 手机实时筛查</h1>
    <p>手机当摄像头 · 电脑实时识别 · 结果回传本机</p>
    <button id="bigstart" title="开始">▶</button>
    <p class="hint">点「▶」并允许使用摄像头</p>
    <div id="err"></div>
  </div>
  <div id="stat2"></div>
  <div id="bar" style="display:none">
    <button class="cbtn" id="flipBtn" title="切换前后摄像头">🔄</button>
    <button class="cbtn" id="stopBtn" title="停止">⏸</button>
    <a class="cbtn" id="saveBtn" href="/snapshot" download="waiqiang_frame.jpg"
       title="存当前帧">💾</a>
    <button class="cbtn" id="fsBtn" title="全屏">⛶</button>
  </div>
</div>

<script>
var video, canvas, ctx, timer=null, pushing=false, sent=0, t0=0;
var dropped = 0;                      // 本拍超时（来不及）的累计次数
var _lsN = 0, _lsT = 0;               // logStat 的滑动窗口基线（见 logStat 注释）
var facing = 'environment';          // 默认后置（拍墙更清楚）
var s        = document.getElementById('s');
var splash   = document.getElementById('splash');
var bigstart = document.getElementById('bigstart');
var errBox   = document.getElementById('err');
var title    = document.getElementById('title');
var meta     = document.getElementById('meta');
var stat2    = document.getElementById('stat2');
var bar      = document.getElementById('bar');
var hud      = document.getElementById('hud');
var stopBtn  = document.getElementById('stopBtn');
var flipBtn  = document.getElementById('flipBtn');
var fsBtn    = document.getElementById('fsBtn');
var mirrorTip= document.getElementById('mirrorTip');
var INTERVAL = __INTERVAL__;     // 推帧间隔 ms（由服务端注入）
var QUALITY  = __QUALITY__;      // JPEG 质量
var MIRROR   = "__MIRROR__";     // 电脑镜像页地址（由服务端注入）

// ★★ 每台手机 / 每个标签页一个 clientId —— 服务端据此**分路**。
//   没有它的话，多台手机会全落到「默认路」⇒ 画面再次互相插帧（等于没改造）。
//   放 sessionStorage：同一标签页刷新仍是同一路（不会每次都新建一路）；
//   换标签页 / 换手机 ⇒ 新 id ⇒ 天然隔离。
//   取不到 sessionStorage（隐私模式）时退回内存随机值，**仍然可用**。
var CLIENT = (function(){
  try {
    var k = 'wq_client_id', v = sessionStorage.getItem(k);
    if (!v) { v = 'c' + Math.random().toString(36).slice(2, 10);
              sessionStorage.setItem(k, v); }
    return v;
  } catch (e) {
    return 'c' + Math.random().toString(36).slice(2, 10);
  }
})();
function CQ(){ return '?client=' + encodeURIComponent(CLIENT); }
function CQAMP(){ return 'client=' + encodeURIComponent(CLIENT) + '&'; }
// ★ 推帧长边（由服务端注入，与 Python 的 DEFAULT_PUSH_MAXW **同一真源**）
//   取 720 的依据（logs/_sweep_push_resolution.py 实测，详见 phone_live.py 注释）：
//   推理时间 27.4 → 17.8 ms（−34.8%）、p95 73.1 → 24.7 ms（−66%），
//   而检出总数 71 → 72、逐类一个都没少 ⇒ 「更快且没变差」。
var MAXW     = __MAXW__;

// ★ 实时帧率（滑动窗口），不是累计平均（2026-09-27 修）
//   旧实现：sent/(now-t0)，t0 只在 begin() 设一次 ⇒ 是**从启动至今的累计平均**，
//   一旦中途卡过就永远爬不回来，屏幕上的数字会**锁死**、看不出当前状态。
//   实测证据：真机录屏里它锁在「3.1 帧/秒」不动（见 logs/_手机端实测帧率分析.md）。
//   ⇒ 改成「自上次采样以来的增量」，反映的是**当前**状态。
function logStat(){
  var now = Date.now();
  var dn = sent - _lsN;
  var dt = (now - _lsT) / 1000;
  var fps = (dt > 0 && _lsT > 0) ? (dn / dt) : 0;
  var parts = sent + ' 帧 · ' + fps.toFixed(1) + ' 帧/秒';
  if (dropped > 0) parts += ' · 丢 ' + dropped;   // ★ 让「来不及」可见
  stat2.textContent = parts;
  _lsN = sent; _lsT = now;
}

function pickVideo(mode){
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return Promise.reject(new Error('浏览器不支持摄像头（需 https 或 localhost）'));
  }
  var tries = [
    { video: { facingMode: { ideal: mode }, width: {ideal:1280},
               height: {ideal:720} }, audio:false },
    { video: { facingMode: mode }, audio:false },
    { video: true, audio:false }
  ];
  return tries.reduce(function(p, c){
    return p.catch(function(){ return navigator.mediaDevices.getUserMedia(c); });
  }, Promise.reject());
}

function startStream(stream){
  video = document.createElement('video');
  video.srcObject = stream;
  video.setAttribute('playsinline','');   // iOS 必须，否则不内联播放
  video.setAttribute('webkit-playsinline','');
  video.muted = true;
  return video.play().then(function(){ return stream; });
}

function begin(){
  bigstart.disabled = true;
  errBox.textContent = '';
  pickVideo(facing).then(startStream).then(function(){
    if (!canvas) {
      canvas = document.createElement('canvas');
      ctx = canvas.getContext('2d');
    }
    s.src = '/stream?' + CQAMP() + 't=' + Date.now();   // 开 MJPEG 结果流
    // ★ 带 client ⇒ 只收**自己这一路**的画面
    splash.style.display = 'none';
    bar.style.display = 'flex';
    hud.classList.add('on');                // HUD 只在运行时显示
    sent = 0; t0 = Date.now();
    dropped = 0; _lsN = 0; _lsT = 0;        // 重置推帧/采样基线
    logStat();                              // 先建立基线，避免首帧算成「除以 0」
    title.textContent = '正在识别（对着墙面慢慢扫）';
    if (MIRROR) {
      var murl = MIRROR + CQ();          // ★ 带上 client ⇒ 大屏正好看这一路
      mirrorTip.innerHTML = '电脑上看这一路的实时画面：<a href="' + murl +
                            '" target="_blank" rel="noopener">' + murl + '</a>';
    }
    if (timer) clearTimeout(timer);
    timer = setTimeout(loopPush, 0);        // 自适应循环（见 loopPush 注释）
  }).catch(function(e){
    splash.style.display = 'flex';
    bar.style.display = 'none';
    errBox.textContent = '摄像头打不开：' + e.message +
      '\\n（请确认用 https 地址打开、并已授权摄像头）';
    bigstart.disabled = false;
  });
}

bigstart.onclick = begin;

stopBtn.onclick = function(){
  if (timer) { clearTimeout(timer); timer = null; }
  if (video && video.srcObject) {
    video.srcObject.getTracks().forEach(function(t){ t.stop(); });
  }
  video = null;
  s.src = '';
  bar.style.display = 'none';
  hud.classList.remove('on');
  splash.style.display = 'flex';
  bigstart.disabled = false;
  title.textContent = '已停止';
  stat2.textContent = '共推送 ' + sent + ' 帧';
};

flipBtn.onclick = function(){
  facing = (facing === 'environment') ? 'user' : 'environment';
  if (video && video.srcObject) {
    video.srcObject.getTracks().forEach(function(t){ t.stop(); });
  }
  if (timer) { clearTimeout(timer); timer = null; }
  pickVideo(facing).then(startStream).then(function(){
    canvas.width = 0; canvas.height = 0;    // 强制按新尺寸重建
    timer = setTimeout(loopPush, 0);        // 自适应循环
  }).catch(function(e){ errBox.textContent = '切换摄像头失败：' + e.message; });
};

fsBtn.onclick = function(){
  var el = document.documentElement;
  if (document.fullscreenElement || document.webkitFullscreenElement) {
    (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  } else {
    (el.requestFullscreen || el.webkitRequestFullscreen).call(el);
  }
};

// ★ 点画面 = 沉浸模式（像相机那样隐藏所有控件）；再点一下恢复
document.getElementById('stage').addEventListener('click', function(ev){
  if (ev.target.closest('button, a')) return;      // 别抢按钮的点击
  if (splash.style.display !== 'none') return;      // 未开始时不切换
  document.body.classList.toggle('immersive');
});

// 状态轮询（拿服务端的真实数字，比前端自报更可信）
setInterval(function(){
  fetch('/status' + CQ()).then(function(r){ return r.json(); }).then(function(j){
    if (!j.has_frame) return;
    title.textContent = '正在识别 · 已出 ' + j.n_inferred + ' 帧结果';
    meta.textContent = 'seq ' + j.seq + ' · 服务端 ' + j.fps +
      ' 帧/秒 · 收到 ' + j.n_received + ' / 已推理 ' + j.n_inferred;
    if (j.analyzer_error) meta.textContent += ' · 错误: ' + j.analyzer_error;
  }).catch(function(){});
}, 1500);

function pushOne(){ return pushOneQ(); }

// ★ 自适应推帧循环（2026-09-27 改，替代 setInterval）
//   为什么必须改：`setInterval(pushOne, INTERVAL)` 是**固定节奏**——
//   若某帧的网络往返/编码耗时超过 INTERVAL，下一次回调会**立刻**触发，
//   而 `pushing` 正好为真 ⇒ 那一拍被**静默丢弃**（无计数、无法区分
//   「没发」与「发了被丢」）。300ms 档实测只达成 67%，正是这个原因。
//   ⇒ 改成「推完一帧（含 fetch 往返）再排下一帧」：
//     ① 队列里**永远只有 1 帧在途**，不会堆积、不会雪崩；
//     ② 节奏自适应实际链路速度，慢的时候自动降频而不是丢帧；
//     ③ 记 dropped，把「来不及」变成**可观测数字**（声明在文件顶部）。
function loopPush(){
  if (!timer) return;                       // 已停止
  var t0p = (performance && performance.now) ? performance.now() : Date.now();
  pushOneQ(function(){
    var spent = ((performance && performance.now) ? performance.now() : Date.now()) - t0p;
    var wait = INTERVAL - spent;
    if (wait < 0) { dropped++; wait = 0; }  // 本拍超时：记账，不等负数延迟
    timer = setTimeout(loopPush, wait);
  });
}

function pushOneQ(done){
  if (pushing || !video || video.videoWidth === 0) { if (done) done(); return; }
  pushing = true;
  // 限制长边到 MAXW，减少推帧体积（推理端 imgsz 仍是 640，两者独立）
  var w = video.videoWidth, h = video.videoHeight;
  var maxw = MAXW;
  if (w > maxw) { h = Math.round(h * maxw / w); w = maxw; }
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  ctx.drawImage(video, 0, 0, w, h);
  canvas.toBlob(function(blob){
    if (!blob) { pushing=false; if (done) done(); return; }
    fetch('/push' + CQ(), { method:'POST', body: blob,
                     headers: {'Content-Type':'image/jpeg'},
                     keepalive:false })
    // ★ 带 client ⇒ 服务端把这一帧记到**我这台手机**的槽里
      .then(function(){ sent++; logStat(); })
      .catch(function(){})
      .then(function(){ pushing=false; if (done) done(); });
  }, 'image/jpeg', QUALITY);
}

// MJPEG 断流自动重连
s.onerror = function(){
  if (timer) setTimeout(function(){ s.src='/stream?'+CQAMP()+'t='+Date.now(); },
                       1500);
};
</script>
</body></html>
"""


# ---------------------------------------------------------------------------
# 电脑端镜像页：把手机推上来的「识别后画面」原样显示在大屏上
# ---------------------------------------------------------------------------
_MIRROR_PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>电脑镜像 · 手机实时画面</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { height:100%; margin:0; overflow:hidden; background:#0b0b0c; }
  body { color:#e8eaed;
         font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
  header { position:fixed; left:0; right:0; top:0; z-index:5;
           padding:10px 18px; display:flex; align-items:center; gap:14px;
           background:linear-gradient(180deg, rgba(0,0,0,.75), rgba(0,0,0,0));
           pointer-events:none; }
  header h1 { margin:0; font-size:16px; font-weight:600; }
  header .live { width:9px; height:9px; border-radius:50%; background:#666;
                 flex:0 0 auto; box-shadow:0 0 8px currentColor; }
  header .live.on { background:#e33; box-shadow:0 0 10px #e33;
                    animation:blink 1.4s infinite; }
  @keyframes blink { 50% { opacity:.3; } }
  header .meta { margin-left:auto; font-size:12.5px; color:#9aa0a6;
                 font-family:ui-monospace,Menlo,Consolas,monospace; }
  #stage { position:fixed; inset:0; display:flex; align-items:center;
           justify-content:center; }
  #s { max-width:100%; max-height:100%; display:block; object-fit:contain; }
  #idle { text-align:center; color:#7a7f85; padding:0 24px; }
  #idle b { color:#cfd3d7; display:block; margin-bottom:10px; font-size:17px; }
  #idle code { background:#000; color:#8fd; padding:2px 7px; border-radius:5px;
               font-size:13px; }
  #foot { position:fixed; left:0; right:0; bottom:0; z-index:5;
          padding:12px 18px; font-size:12px; color:#7a7f85;
          background:linear-gradient(0deg, rgba(0,0,0,.7), rgba(0,0,0,0)); }
</style></head><body>
<header>
  <span class="live" id="dot"></span>
  <h1>电脑镜像 · 手机实时画面</h1>
  <span class="meta" id="meta"></span>
</header>
<div id="stage">
  <!-- ★ 必须初始 display:none —— 否则在还没有帧时浏览器会渲染「破图图标 + alt 文字」，
       正好压在下面的等待提示上（实测截图可见），看起来像页面坏了。
       有帧后由 JS 置为 block。 -->
  <img id="s" alt="手机实时画面" style="display:none">
  <div id="idle">
    <b>等待手机推帧…</b>
    请用手机打开 <code id="phoneUrl">https://&lt;本机IP&gt;:7863</code>，
    点「▶」开始推帧后，这里会实时显示与手机上完全相同的识别画面。
  </div>
</div>
<div id="foot">
  显示的是<b>识别后</b>的画面（检测框 / 分级标注与手机上一致）。
  边界：实时流不提高毫米级判读能力，门槛由 GSD 决定；逐帧毫米值不可单独采信。
</div>
<script>
var s = document.getElementById('s');
var dot = document.getElementById('dot');
var metaEl = document.getElementById('meta');
var idle = document.getElementById('idle');
var phoneUrl = document.getElementById('phoneUrl');
var lastSeq = -1, online = false;
// ★ 由服务端注入（`/mirror?client=<id>`）；空串 ⇒ 看默认路。
var CLIENT = "__CLIENT__";
function CQ(){ return CLIENT ? ('?client=' + encodeURIComponent(CLIENT)) : ''; }
function CQAMP(){ return CLIENT ? ('client=' + encodeURIComponent(CLIENT) + '&') : ''; }

// 用当前页面的 host 推出手机地址（换 IP 时自动跟随，不用改代码）
if (phoneUrl) { phoneUrl.textContent = location.origin; }

setInterval(function(){
  fetch('/status' + CQ()).then(function(r){ return r.json(); }).then(function(j){
    var has = !!j.has_frame && j.seq >= 0;
    var shown = s.style.display !== 'none';
    if (has && !shown) {
      s.style.display = 'block';
      idle.style.display = 'none';
      s.src = '/stream?' + CQAMP() + 't=' + Date.now();
    } else if (!has && shown) {
      // 手机还没开始（或已停），回到等待态
      s.style.display = 'none';
      s.src = '';
      idle.style.display = '';
      dot.classList.remove('on');
      online = false;
    }
    if (has) {
      if (j.seq !== lastSeq) {
        lastSeq = j.seq;
        dot.classList.add('on');
        online = true;
      }
      metaEl.textContent = 'seq ' + j.seq + ' · ' + j.fps + ' 帧/秒 · 收到 ' +
        j.n_received + ' / 已推理 ' + j.n_inferred;
      if (j.analyzer_error) { metaEl.textContent += ' · 错误: ' + j.analyzer_error; }
    }
  }).catch(function(){});
}, 1200);

// MJPEG 断流自动重连（手机端停掉后 /stream 会结束）
s.onerror = function(){ if (online) setTimeout(function(){
  s.src = '/stream?' + CQAMP() + 't=' + Date.now(); }, 1200); };
</script>
</body></html>
"""


# ---------------------------------------------------------------------------
# ★ 推帧间隔默认值：**单一真源**（2026-09-27 修默认值分叉；同日晚按实测再降）
# ---------------------------------------------------------------------------
# 血泪教训：默认值曾在两处各写一份 —— `main()` 里 `_arg("interval", X)`
# 与 `build_app(interval_ms=Y)`。两者一旦不一致，**命令行入口**和
# **任何别的调用者**（测试脚本、别的模块 import）就会拿到不同的值，
# 而且现象非常隐蔽：改了一处、验证脚本仍跑旧值 ⇒ 误判成「改动无效」。
# （本次实测就是：main 已改 100，而 build_app 还是 300，
#   验证脚本调 build_app 不传参 ⇒ 实测仍是 300ms ⇒ 白折腾一轮。）
# ⇒ 现在两处都引用下面这个常量，禁止再写字面量。
#
# ★★ 取值依据（**两轮实测**，本机 RTX 5070 Ti + v11s640）：
#
# 第一轮 `logs/_bench_interval_sweep.py`（自建 HTTP 客户端，**无摄像头**）：
#   INTERVAL  目标fps   实测端到端   达成率
#     300      3.33      2.24       67.3%   ← 唯一不达标档（与识别周期节拍共振）
#     200      5.00      4.98       99.6%
#     100     10.00     10.06      100.6%   ← 当时以为到顶了
#      66     15.15     15.15      100.0%
#   ⇒ 当时（错误地）结论：p50 串行 41ms 是瓶颈，100ms 已留足余量。
#
# 第二轮 `logs/_sweep_e2e_ceiling.py` + `logs/_sweep_interval_efficiency.py`
# （Playwright 假摄像头，**完整真机链路**）推翻了这个结论：
#
#   INTERVAL  推帧fps   真推理fps   收推比   浪费率
#     200      4.82      4.82     1.000    0.0%
#     150      6.65      6.49     1.026    2.5%
#     100      9.66      9.66     1.000    0.0%
#      80     12.16     12.00     1.014    1.4%
#      66     14.98     14.81     1.011    1.1%
#      50     19.83     19.66     1.008    0.8%   ← 采用
#
# ⇒ **端到端 fps 到 50ms 仍在单调上涨，且「收推比」始终 ≈ 1.0**
#   （即服务端**每一帧都真的推理了**，没有积压、没有静默丢弃）。
#
# 为什么第一轮的串行模型是错的（这是本次最有价值的认知）：
#   它默认「服务端识别耗时叠加在推帧节奏上」。实际上 `PhoneAnalyzer`
#   是**独立线程**，按自己的节奏**轮转**消费各 client 槽里**最新的一帧**
#   （2026-09-27 多路改造前是「一个槽」，多机会互相覆盖 ⇒ 已改为 per-client 槽）；
#   而客户端自适应循环只用 `performance.now()` 量**自己这一拍**的耗时。
#   两者**解耦** ⇒ 服务端慢只会让它跳过中间帧，**不会拖慢推帧节奏**。
#   （第一轮那个「66ms ⇒ 15.15 fps = 100% 达成率」其实就是这个证据 ——
#     66 目标 15.15 理论，若真串行叠加 41ms，66ms 档最多只能到 9.3 fps。
#     当时把 100% 达成读成「刚好够用」，其实是「链路根本没有串行瓶颈」。）
#
# 为什么敢用 50ms（安全余量校核）：
#   `logs/_measure_real_stage_breakdown.py` 测得**有检出时**单帧
#   服务端串行 21.2 ms（720 输入）⇒ 上限 47 fps。
#   50ms 档 19.66 fps 只占用 **42%** 的服务端吞吐 ⇒ 约 2.4 倍余量。
#   ⚠️ 该 21.2 ms 是在**真实测试图**（含几十个检出框）上测的，
#      而假摄像头是彩色噪声、检出≈0 ⇒ 假摄像头那一档偏乐观。
#      因此**真机余量比 2.4 倍小**，取 50ms（不是 33ms）正是为了留这份余量。
#   ⚠️ 上限还有一条：**摄像头源帧率**（实测 20.09 fps）。
#      19.83 fps 已用掉源帧率的 98.7% ⇒ 再降 INTERVAL 不会更快，
#      只会白推重复帧（`drawImage` 拿到同一帧）。**这才是真正的天花板。**
#
# ⇒ 若真机上看到「丢 N」持续增长或 `收到 / 已推理` 比值 > 1.2，
#   说明这台手机的源帧率或 WiFi 更弱 ⇒ 用 `--interval=66` 回退一档。
DEFAULT_INTERVAL_MS = 50


def build_app(args: dict, *, interval_ms: int = DEFAULT_INTERVAL_MS,
              quality: float = 0.6, out_w: int = DEFAULT_PUSH_MAXW,
              imgsz: int = 640, push_maxw: int = DEFAULT_PUSH_MAXW):
    """构造「手机推帧」版 FastAPI 应用。

    ★ `out_w` 与 `push_maxw` 的默认值**都**引用 `DEFAULT_PUSH_MAXW`：
      `push_maxw` 是手机端 canvas 的长边，`out_w` 是服务端叠加图的长边，
      二者本该一致（详见 `DEFAULT_PUSH_MAXW` 的注释）。
      别再手填第二个字面量 —— 那正是上一轮踩过的「默认值分叉」。
    """
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
        # 镜像页地址：手机页里给一条可点的链接，方便直接在大屏上打开。
        # 用相对路径 `/mirror` ⇒ 换 IP / 换端口都不用改代码。
        html = (_PAGE.replace("__INTERVAL__", str(int(interval_ms)))
                     .replace("__QUALITY__", str(float(quality)))
                     .replace("__MAXW__", str(int(push_maxw)))
                     .replace("__MIRROR__", "/mirror"))
        return HTMLResponse(html)

    @app.get("/mirror", response_class=HTMLResponse)
    def mirror(client: Optional[str] = None):           # noqa: D401
        """电脑端镜像页：把手机那一路的识别后画面原样显示在大屏上。

        ★ 与手机页读**同一路**的 STATE ⇒ 两端画面逐帧一致，
          不会出现「手机和电脑各跑一套、结果不同」的口径分叉。
        ★ 多机时用 `/mirror?client=<id>` 指定看哪一路（缺省 = 默认路）。
        """
        cid = analyzer.norm_client(client)
        # 把路由注入页面（与 `__INTERVAL__` 等同一套占位符替换手法）
        return HTMLResponse(
            _MIRROR_PAGE.replace("__CLIENT__", "" if cid == REG.DEFAULT else cid))

    @app.post("/push")
    async def push(request: _FastAPIRequest,
                   client: Optional[str] = None):
        """只做一件事：把手机推来的 JPEG 丢进识别队列（微秒级返回）。

        ⚠️ 形参注解写法**不能随意改**：
        · 必须用**模块级**导入的 `_FastAPIRequest`（见文件顶部说明），
          写成函数内 import 的 `Request` 会让 FastAPI 把它当查询参数 ⇒ 422；
        · 必须用 `await request.body()` 自己读字节，
          **不要**改成 `body: bytes = Body(...)`（同样会被注解字符串化坑）。
        """
        body = await request.body()
        # 2026-09-29 安全加固（#7）：帧大小上限（DoS 防护）。JPEG 帧远小于 2MB。
        if len(body) > 2 * 1024 * 1024:
            return JSONResponse({"ok": False, "msg": "帧过大"}, status_code=413)
        if not body:
            return JSONResponse({"ok": False, "msg": "空帧"}, status_code=400)
        cid = analyzer.norm_client(client)
        kept = analyzer.submit(body, cid)
        # ★ `kept=False` 表示被限流丢弃 —— 明确回传，不假装成功。
        return JSONResponse({"ok": True, "n": analyzer.n_received,
                             "client": (cid or "<default>"),
                             "kept": bool(kept),
                             "n_throttled": analyzer.n_throttled})

    def _status_impl(client: Optional[str] = None):
        """★ 只加不删：原 7 个字段语义**完全不变**（全局计数），
        新增字段一律**追加** ⇒ 旧前端（只读已知键）不受影响。"""
        cid = analyzer.norm_client(client)
        st = REG.get_state(cid)
        _, seq, _info = st.get()
        return JSONResponse({
            "mode": "phone-push",
            "seq": seq,
            "fps": round(st.fps, 2),
            "has_frame": st.get()[0] is not None,
            "n_received": analyzer.n_received,
            "n_inferred": analyzer.n_inferred,
            "analyzer_error": analyzer.last_error,
            # ---- 以下为 2026-09-27 多路改造**新增** ----
            "client": (cid or "<default>"),
            "per_client": analyzer.per_client(),
            "clients": REG.clients(),
            "n_throttled": analyzer.n_throttled,
            "n_degraded": REG.n_degraded,
            "max_clients": REG.max_clients,
        })

    app.get("/status")(_status_impl)      # ★ 注册（保留原路径不变）

    @app.get("/snapshot")
    def snapshot(client: Optional[str] = None):
        # ★ 与 /stream 同样接 ?client=，否则会出现
        #   「/stream?client=A 是 A 的画面、/snapshot 却是默认路」的不一致。
        jpeg, _, _ = REG.get_state(client).get()
        if jpeg is None:
            return JSONResponse({"ok": False, "msg": "还没有帧"}, status_code=503)
        return Response(jpeg, media_type="image/jpeg")

    @app.get("/stream")
    def stream(client: Optional[str] = None):
        # ★ 每路读**自己**的槽（缺省 = 默认路，旧 URL 照常可用）。
        _st = REG.get_state(client)

        def gen():
            last = -1
            idle = 0.0
            while True:
                jpeg, seq, _ = _st.get()
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
        from common import argv_flag   # 2026-09-28：裸 flag 统一走 common.argv_flag（单一真源）
        return argv_flag(name, default)

    host = _arg("host", "127.0.0.1")
    port = int(_arg("port", "7862"))
    # 间隔默认值见模块级 DEFAULT_INTERVAL_MS 的注释（含实测表与选取依据）。
    interval_ms = int(_arg("interval", str(DEFAULT_INTERVAL_MS)))
    quality = float(_arg("quality", "0.6"))
    imgsz = int(_arg("imgsz", "640"))
    # 推帧长边默认值见模块级 DEFAULT_PUSH_MAXW 的注释（含实测表与选取依据）。
    # ★ `--out-width` 未显式指定时**由 push_maxw 推导**，不写字面量
    #   （`out_w` 若大于收到帧的长边，draw_overlay 会凭空放大 —— 纯亏）。
    push_maxw = int(_arg("push-maxw", str(DEFAULT_PUSH_MAXW)))
    out_w = int(_arg("out-width", str(_derive_out_w(push_maxw))))
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
    from deploy_args import build_run_one_args
    # ★ 2026-09-29（#4）：args 键名统一走 deploy_args.build_run_one_args（修 11 键错 7）。
    args = build_run_one_args(
        calib_mode="相机参数估算（最粗）",
        calib_object_px=0.0, calib_object_mm=210.0,
        brick_pitch_mm=250.0,
        distance_m=20.0, focal_mm=24.0,
        env_class="二类环境（露天/潮湿）",
        conf_thr=0.25,
        rectify_mode="关闭（正对拍摄）",
        jgj125_parts=None,
    )
    args["weight_path"] = str(w) if w else ""

    scheme = "https" if use_ssl else "http"
    print("=" * 72)
    print("外墙缺陷 · 手机实时筛查（手机推帧 + MJPEG 回传）")
    print("=" * 72)
    print("权重      :", args["weight_path"] or "（未找到，请先训练）")
    print("监听      : %s://%s:%d" % (scheme, host, port))
    print("推帧间隔  : %d ms | JPEG 质量: %.2f | imgsz: %d"
          % (interval_ms, quality, imgsz))
    print("推帧长边  : %d px（手机缩到此尺寸再编码）| 叠加图长边: %d px"
          % (push_maxw, out_w))
    if use_ssl:
        print("HTTPS     : 已启用（证书 %s）" % ssl_cert.name)
        print("   ⚠️ 自签名证书 ⇒ 手机首次打开会提示「不安全」，"
              "点「高级 → 继续访问」即可")
    if host not in ("127.0.0.1", "localhost"):
        print("📱 手机打开 : %s://<本机局域网IP>:%d" % (scheme, port))
        print("   查本机 IP : ipconfig | findstr IPv4")
        print("🖥  电脑镜像 : %s://<本机局域网IP>:%d/mirror" % (scheme, port))
        print("   ⇒ 手机点「▶」开始后，镜像页会实时显示同一路识别画面")
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
                         out_w=out_w, imgsz=imgsz, push_maxw=push_maxw)
    if use_ssl:
        uvicorn.run(app, host=host, port=port, log_level="warning",
                    ssl_certfile=str(ssl_cert), ssl_keyfile=str(ssl_key))
    else:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
