# 09_uav_realtime —— 无人机外墙缺陷实时识别实验

> **本目录是独立实验目录。** 不修改、不覆盖、不重命名 `02_code/`、`06_deploy/`、
> `03_weights/` 下的任何既有文件；只**只读复用** `02_code/pipeline.py::run_one`
> 作为唯一推理链路。

---

## 1. 这个目录解决什么问题

现有 `02_code/video_screen.py` 是**事后分析型**：按时间等间隔抽帧、跨帧贪心聚合，
目标是把一段视频看完给一份覆盖面结论。它**没有** FPS / 端到端延迟概念，也没有
流式读取与断线重连。

本目录是**实时型**：关心"每帧多久出结果、源断了怎么办"，用于验证新到手的
小型无人机能否支撑外墙巡查的实时识别。两者目的不同，硬塞在一起会互相污染口径，
故独立成目录。

---

## 2. 文件清单与职责

| 文件 | 职责 | 是否可改 |
|---|---|---|
| `rt_config.yaml` | **唯一配置面**。视频源（4 种预设）、分辨率、采样帧率、置信度、imgsz、设备、标定参数、落盘开关、判定阈值 | 按需改（注释齐全） |
| `rt_common.py` | 公共层：配置加载与校验、**args 键名归一化**（YAML 友好名 → `run_one` 旧口径名）、中文标签/HUD 绘制、分位统计、计时 | 谨慎改 |
| `rt_infer.py` | **主程序**。源分派打开 → 跳帧节流 → 调 `run_one(image=frame)` → 叠框 → 分层计时 → 落盘 | 谨慎改 |
| `rt_verify.py` | **自检脚本**。不需要无人机，断言管线跑通且产物真的有效 | 谨慎改 |
| `logs/` | 运行日志 + `rt_metrics_*.json`（逐帧记录与分位统计） | 自动生成 |
| `out_frames/` | 抽样标注帧截图（jpg） | 自动生成 |
| `out_video/` | 带框回放 mp4 | 自动生成 |

---

## 3. 依赖安装

**无需安装任何新第三方包。** 所需 `opencv-python` / `numpy` / `PyYAML` /
`ultralytics` / `torch` / `Pillow` 全部已在 `requirements.txt` 内。

确认环境：

```bash
D:\下载\python.exe -c "import cv2, yaml, ultralytics, torch, PIL; print(cv2.__version__, ultralytics.__version__)"
```

万一缺 PyYAML（本机实测不缺）：

```bash
# ★ 本机 pip 全局指向清华源、对 pip 完全不可用，必须显式覆盖为官方源
D:\下载\python.exe -m pip install "PyYAML>=6.0" -i https://pypi.org/simple
```

> 注意：跑本项目一律用 `D:\下载\python.exe`，不要用裸 `python`。

---

## 4. 验证步骤（三步递进）

```bash
cd "D:/pythonstudy 备份/创新题/外墙缺陷筛查/09_uav_realtime"

# ① 自检（CPU，最稳最快暴露逻辑问题；33 项断言，失败即非零退出码）
D:\下载\python.exe -X utf8 rt_verify.py --device=cpu --max-frames=8 --warmup=2

# ② 基准（GPU，合成视频逐帧全跑；拿到正式性能数字）
D:\下载\python.exe -X utf8 rt_infer.py --source=synthetic --device=cuda:0 --max-frames=80 --target-fps=0

# ③ 真实（按你的实际情况选预设）
D:\下载\python.exe -X utf8 rt_infer.py --source=rtsp     --device=cuda:0   # 无人机 RTSP 实时流
D:\下载\python.exe -X utf8 rt_infer.py --source=reclip   --device=cuda:0   # 无人机录回的视频文件
D:\下载\python.exe -X utf8 rt_infer.py --source=camera   --device=cuda:0   # USB 采集卡 / 摄像头
```

**判定标准**：`rt_verify.py` 退出码为 0 且末行打印「自检结果：全部通过」。

---

## 5. 四种视频源预设

在 `rt_config.yaml` 的 `source.presets` 下切换，或用 `--source=` 覆盖：

| 预设名 | kind | 用途 | 需填 |
|---|---|---|---|
| `synthetic` | file | 工程内合成视频，仅验证管线连通 | 已配好 |
| `reclip` | file | 无人机录回的视频文件 | **`uri` 必须填**（为空会显式报错） |
| `rtsp` | stream | 无人机 RTSP 实时流 | `uri` 改成实际图传地址 |
| `camera` | device | USB 采集卡 / 本机摄像头 | `uri` = 设备号（如 `"0"`） |

---

## 6. 三级性能指标口径（报告请用这三个）

| 指标 | 定义 | 决定什么 |
|---|---|---|
| `infer_ms` | 单次 `run_one` 墙钟耗时（检测+量化+分级+绘制） | 模型本身快不快 |
| `proc_fps` | 正式统计帧数 / 处理墙钟秒 | **能不能实时交互** |
| `e2e_ms` | 取到帧 → 叠加图产出 的墙钟差 | **反馈迟不迟** |
| `decode_ms` | 单次取帧耗时 | 瓶颈在解码还是在模型 |

`rt_metrics_*.json` 里给的是 **P50 / P90 / P99 / max**，不只有均值 ——
实时性的成败由长尾决定，一个 P99=800ms 的系统均值可能仍只有 120ms 却完全不可用。

**预热与统计是两套独立预算**：`warmup_frames` 不占 `max_frames`，且预热帧不进统计
（首帧含 CUDA 上下文与 cudnn autotune，属一次性开销，计入会把吞吐压低数倍）。

---

## 7. 已知边界（**必须如实写进报告**）

1. **工程内没有真实实拍外墙视频。** `07_report/` 下那两段 mp4 是界面/PPT **录屏**，
   不是外墙实拍，在其上 0 检出是**正确行为**，不是模型失效。
   （依据：`logs/_VIDEO_SCREEN_VERIFY.md`）
2. 因此用 `--source=synthetic` 得到的结果**只能证明管线与指标可用**，
   **不得据此声称已完成实拍验证**。真实结论须等无人机素材（`reclip` 或 `rtsp`）。
3. `rt_metrics_*.json` 的 `caveat` 字段会自动带上这条声明，`rt_verify.py`
   也会断言该字段存在。

---

## 8. 设计上刻意规避的历史坑

这些是本工程长期踩过、本模块**主动规避**的问题（都写在代码注释里）：

| 坑 | 规避方式 |
|---|---|
| 参数键名不匹配 → **静默**回落默认值 | `build_args()` 集中做键名映射；非法 `env_class` 直接报错 |
| 标定链兜底 `distance<=0` 会**在跑一半时崩** | 配置给正数默认 + `build_args()` 启动期预检 |
| 预热吃光帧数 → 统计为空却"显示成功" | 预热独立预算 + 零样本时打印「无效实验，不得用于性能结论」 |
| `cv2.putText` 不支持中文 → 标签变 `?` | 中文层走 PIL（微软雅黑/黑体），cv2 只画 ASCII |
| GPU 计时未同步 → 测到的是"发起调用"时间 | `Seg` 进出都 `torch.cuda.synchronize()` |
| 逐帧写盘再读盘传路径 → 纯 I/O 浪费 | 走 `run_one(image=frame)` 内存帧入口 |
| 检测框样式各写一套 → 与主链路不一致 | 直接用 `run_one` 返回的 `annotated`，本层只追加中文标签与 HUD |
| 同类缺陷颜色静默回退 → 配图误标 | `CLASS_COLORS` 与 `pipeline.py` 逐项一致，含全部 7 类 |
