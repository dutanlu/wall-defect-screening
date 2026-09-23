# -*- coding: utf-8 -*-
"""
step6_quantize.py —— 模型轻量化：FP32 → ONNX → INT8（赛道二「部署效率」）

===== 为什么需要这一步 =====

赛道二明确要求「模型体积 < 10MB、推理 < 100ms」。FP32 的 yolov8s 训练完是
85 MB、yolov8n 是 6 MB。达标路径有三条，本脚本把前两条做成可复现实验：

  路径 A：换更小的模型规格（yolov8n）—— train.py 里的 v8n640 配方
  路径 B：INT8 量化（本脚本）—— 精度几乎不掉，体积直接砍到 1/4
  路径 C：结构化剪枝 —— 需要 fine-tune，收益/成本比最差，本项目不做

本脚本做**两条量化路线并给出对照**，因为「只报一个数字」在技审时会被追问
「你凭什么说 INT8 不掉精度」。两条路线：

  路线 1（主推）：ONNX Runtime 静态量化（PTQ，QDQ 格式）
      - 需要校准集（用 val 集的前 N 张，代表真实分布）
      - 产出 .onnx，跨平台可部署（ORT / OpenVINO / TensorRT 都能吃）
      - 这是工业界部署 YOLO 的标准做法
  路线 2（对照）：PyTorch FX 动态量化（仅权重 INT8）
      - 不需要校准集，但激活仍是 FP32，加速有限
      - 用它来说明「为什么我们选静态量化而不是动态量化」

===== 关键工程坑（本机实测，勿重犯）=====

坑 1：ultralytics 的 `model.export(format="onnx", int8=True)` 内部走的是
      partial quantize，**质量不可控且不产出可复现的校准缓存**。
      因此本脚本自己走 ORT 的 quantize_static，校准集自己喂，全程可控。

坑 2：ONNX 导出必须显式指定 `opset` 与 `dynamic=False`（固定 batch=1）。
      动态 batch 会让 ORT 量化在 Resize 节点上失败（形状推导拿不到固定 H/W）。

坑 3：YOLO 的 ONNX 输出是 (1, 4+nc, N) 的单张量，**没有内置 NMS**。
      量化后的模型若不再走 ultralytics 的 `NMS` 节点，
      需要在部署侧自己做 NMS —— 这对推理延迟有影响，必须计入。

坑 4：ORT 静态量化前必须跑 `quant_pre_process` 做形状推断，
      否则部分算子的量化会静默跳过（体积看起来降了，实际没量化到主干）。

坑 5（**本机实测踩到，最坑的一个**）：ORT 的 `quantize_static` 在**中文路径**下
      必定抛 `FileNotFoundError(2)`。已用最小复现定位：

        onnx.load("<中文路径>/m.onnx")                    -> 成功
        quantize_static("<中文路径>/m.onnx", ...)         -> FileNotFoundError

      真实原因：`quant_pre_process` 会在源文件旁写一个 `xxx-inferred.onnx`，
      随后 ORT 内部经 C++ 层再把这个路径 round-trip 一次，中文被按 GBK/UTF-8
      错误编解码成乱码（实测报错路径为
      `D:/pythonstudy 澶囦唤/鍒涘妞/...-inferred.onnx`），于是找不到文件。
      注意：**这与「输出文件名是否英文」无关** —— 是**整条路径**任一段含中文都不行。

      修复方式（本脚本已实现）：所有量化相关的中间产物与输出，
      先 stage 到系统临时目录（`tempfile.gettempdir()`，纯 ASCII），
      量化完成后再把最终 INT8 模型复制回项目的 `04_results/quant/`。
      实测：同样一个 FP32 ONNX，中文路径失败，ASCII 临时目录成功
      （11.70MB FP32 -> 3.23MB INT8）。

用法：
  python step6_quantize.py --weights=03_weights/v11s640_best.pt   # 主力
  python step6_quantize.py --all          # 对所有已训练权重批量量化
  python step6_quantize.py --runs=v11s640,v8n640  # 只量化指定的 run（推荐）
  python step6_quantize.py --weights=... --calib-n=200
  python step6_quantize.py --weights=... --calib-method=MinMax
  python step6_quantize.py --weights=... --no-eval   # 只量化，不评估精度

输出：
  04_results/quant/<name>_fp32.onnx          FP32 ONNX（量化基线）
  04_results/quant/<name>_int8.onnx          INT8 ONNX（主交付物）
  04_results/quant/<name>_preprocessed.onnx  预处理后的中间件（调试用，可删）
 04_results/quant/quant_report.json          体积/延迟/精度对照表
"""

from __future__ import annotations

import os

# 为什么必须关掉自动安装（2026-09-20 实测定案，这是本机一次真实事故的根因）：
#   本脚本第 [5b] 步要用 ultralytics 加载 **FP32 / INT8 ONNX** 做端到端检出对比。
#   若没显式指定 device，ultralytics 会认为要跑 CUDA，于是去找 `onnxruntime-gpu`；
#   找不到就**自动执行 `pip install --no-cache-dir onnxruntime-gpu`**。
#   而 onnxruntime-gpu 与本机已装的 CPU 版 `onnxruntime` **共用同一个
#   `onnxruntime/` 包目录** —— pip 会直接**覆盖**其中的 .py 与 .dll。
#   实测后果：Python 层被换成 1.30.0、dist-info 仍是 1.29.0、
#   文件于 12:38~12:42 被替换，随后 `session.run()` 直接抛
#   `AttributeError: ... has no attribute 'is_webgpu_graph_capture_enabled'`
#   —— 本机 ORT 整条推理链路报废，量化连续两轮无法复现。
#   （时间点与 12:23–12:44 那轮量化跑到 [5b] 完全吻合，元凶即此处。）
#
#   所以这里**强制**关掉自动安装：脚本不允许改写共用的 Python 环境。
#   需要 GPU 版请**人工**、显式地安装，而不是让一个推理调用顺手改环境。
#
# --------------------------------------------------------------------------
# 两个开关，第二道才是**真正可靠的**那道（2026-09-20 读 ultralytics 8.4.138 源码确认）
# --------------------------------------------------------------------------
# 触发链（已逐行核实，非推测）：
#   ① `YOLO("x.onnx")` **不会**触发依赖检查 —— `engine/model.py:253-257` 对非 .pt 文件
#      只把 `self.model` 设成**路径字符串**，此时还没有 ORT session。
#   ② 真正的触发点是**第一次 predict/val**：`engine/model.py:512`
#      `args = {**self.overrides, **custom, **kwargs}` → `:520` 把 args 作为
#      `overrides` 传给 predictor → `engine/predictor.py:429`
#      `AutoBackend(device=select_device(self.args.device), ...)`
#      （val 路径同理：`engine/validator.py:183-186`）。
#   ③ `nn/backends/onnx.py:63` 判定 cuda：
#      `cuda = isinstance(self.device, torch.device) and torch.cuda.is_available()
#              and self.device.type != "cpu"`
#      → 本机有 CUDA 且 device 未指定 → cuda=True
#   ④ `nn/backends/onnx.py:76`
#      `check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))`
#      → `onnxruntime-gpu` 缺失 → 进 AutoUpdate 分支 → pip 安装。
#   ⑤ `utils/checks.py:595` `@Retry(times=2, delay=1)` —— 这就是实测里
#      **连续杀到 3 个 pip 进程**（首次 + 2 次重试）的原因。
#
# 【开关一】YOLO_AUTOINSTALL=false
#   `utils/checks.py:625` `if install and AUTOINSTALL:` 是安装动作的总闸，
#   但 `AUTOINSTALL = env_bool("YOLO_AUTOINSTALL", True)` 定义在
#   `utils/__init__.py:72`，是**模块级常量** —— import 时读一次，**之后再设无效**。
#   所以这一行必须在任何 ultralytics 导入之前执行。
#
# 【开关二】ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1  ← 这道更可靠
#   `utils/checks.py:561` `if env_bool("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"): return True`
#   —— 它在 `check_requirements()` **函数体内**求值，也就是**每次调用时**都重新读，
#   不依赖导入顺序；而且它在**包解析之前**就 return，连「哪些包缺失」都不算，
#   因此能从根上杜绝 pip 被调起。这是 ultralytics 专为「别动我的环境」设计的开关。
#   两个都设：开关二保证行为正确，开关一作为兜底（覆盖不经 check_requirements 的路径）。
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"

import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np

from common import (
    DATASET_DIR,
    DATASET_YAML_NAME,
    TRAIN_DIR,
    WEIGHTS_DIR,
    dump_json,
    ensure_dirs,
    imread_u,
    list_images,
    log,
    resolve_weight,
)

# --------------------------------------------------------------------------
# 路径常量
# --------------------------------------------------------------------------
from common import PROJ_ROOT

QUANT_DIR: Path = PROJ_ROOT / "04_results" / "quant"
CALIB_CACHE_DIR: Path = QUANT_DIR / "_calib_cache"

# 校准集张数。太小（<50）校准统计不稳，太大（>300）收益递减且慢。
DEFAULT_CALIB_N: int = 150
IMGSZ: int = 640

# 最近一次 quantize_static 失败的**真实异常原文**。
#
# ⚠️ 为什么需要这个全局变量（2026-09-20 实测教训）：
#   `quantize_onnx_static()` 在失败时按设计**吞掉异常并返回 None**，好让调用方
#   走「降校准集重试」分支。但异常原文就此丢失，调用方只剩「拿到 None」这一个
#   事实，于是收尾日志把它统一归因成「均 OOM」。
#   而 2026-09-20 本机 onnxruntime 损坏时，真实异常其实是
#   `AttributeError: ... has no attribute 'is_webgpu_graph_capture_enabled'`，
#   **与内存毫无关系**。这条误诊信息会把人引向「去调校准集大小 / 加内存」的
#   错误方向（我确实照着查了一圈）。
#   → 失败时把原文存下来，收尾如实打印，不做无依据的归因。
_LAST_QUANT_ERROR: str | None = None


# --------------------------------------------------------------------------
# 工具：参数解析（common 里没有 --key=v1,v2 的列表解析，这里自己补）
# --------------------------------------------------------------------------
def _argv_list(name: str) -> list[str]:
    """解析 --name=a,b,c 形式的列表参数。"""
    import sys
    prefix = f"--{name}="
    for a in sys.argv[1:]:
        if a.startswith(prefix):
            return [x.strip() for x in a[len(prefix):].split(",") if x.strip()]
    return []


def _argv_str(name: str, default: str | None = None) -> str | None:
    import sys
    prefix = f"--{name}="
    for a in sys.argv[1:]:
        if a.startswith(prefix):
            return a[len(prefix):]
    return default


def _argv_int(name: str, default: int) -> int:
    v = _argv_str(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def _argv_bool(name: str) -> bool:
    """--name 存在即为 True（不带 = 值）。"""
    import sys
    return f"--{name}" in sys.argv[1:]


# --------------------------------------------------------------------------
# 步骤 1：FP32 → ONNX
# --------------------------------------------------------------------------
def export_onnx(weights: Path, out_onnx: Path, imgsz: int = IMGSZ) -> Path | None:
    """
    Ultralytics 权重导出为固定形状的 FP32 ONNX。

    坑 2 说明：必须 dynamic=False + 显式 opset。
    ultralytics 8.4 默认 opset=12，但对 QDQ 量化，
    opset>=13 才能正确处理 per-channel 的量化轴属性。
    """
    from ultralytics import YOLO

    log(f"  导出 ONNX: {weights.name} @ {imgsz}")
    # ultralytics 会自己命名输出，事后移动到我们要的位置
    model = YOLO(str(weights))
    try:
        # 注意：ultralytics 8.4 已把 `half` 标记为 deprecated
        # （新参数名是 `quantize`）。这里不传 half，让它走默认 FP32，
        # 避免新旧版本行为不一致 —— 量化是我们自己用 ORT 做的，不依赖它。
        produced = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=13,
            dynamic=False,
            simplify=True,
            batch=1,
            device="cpu",        # 导出在 CPU 上更稳（GPU 导出偶发算子不支持）
        )
    except Exception as e:
        log(f"  !! ONNX 导出失败: {e!r}")
        return None

    src = Path(str(produced))
    if not src.exists():
        log(f"  !! 导出返回路径不存在: {src}")
        return None

    ensure_dirs(out_onnx.parent)
    if src.resolve() != out_onnx.resolve():
        shutil.copy2(src, out_onnx)
        # ultralytics 把 onnx 写在权重旁边（weights/best.onnx），
        # 那是训练产物目录，不该留下推理格式的残留 —— 删掉保持目录干净。
        # 只在它确实位于 weights/ 下、且和我们的暂存路径不同时才删。
        if src.parent.name == "weights":
            try:
                src.unlink()
                log(f"    （已清理导出残留 {src.name}）")
            except OSError:
                pass
    mb = out_onnx.stat().st_size / 1024 / 1024
    log(f"    -> {out_onnx.name} ({mb:.2f} MB)")
    return out_onnx


# --------------------------------------------------------------------------
# 步骤 2：校准集（给静态量化用）
# --------------------------------------------------------------------------
class YoloCalibReader:
    """
    ONNX Runtime 静态量化需要的校准数据读取器。

    要点：喂进去的必须是**与部署时完全一致**的预处理结果 ——
    即 letterbox 后的 NCHW float32 [0,1]。若这里图省事直接 resize，
    量化后的尺度统计就会偏离真实推理，精度掉得莫名其妙。

    所以这里复用 ultralytics 自己的 LetterBox，保证和推理管线同源。
    """

    def __init__(self, images: list[Path], imgsz: int = IMGSZ, max_n: int = 150):
        self.imgsz = imgsz
        self.paths = images[:max_n]
        self._i = 0
        self._input_name = "images"

    def set_input_name(self, name: str) -> None:
        self._input_name = name

    def _letterbox(self, img: np.ndarray) -> np.ndarray:
        """与推理一致的 letterbox + BGR->RGB + 归一化。"""
        from ultralytics.data.augment import LetterBox
        lb = LetterBox(self.imgsz, auto=False, stride=32)
        out = lb(image=img)
        out = out[:, :, ::-1]                      # BGR -> RGB
        out = np.ascontiguousarray(out.transpose(2, 0, 1))
        return out.astype(np.float32) / 255.0

    def get_next(self):
        if self._i >= len(self.paths):
            return None
        p = self.paths[self._i]
        self._i += 1
        img = imread_u(p)
        if img is None:
            return self.get_next()
        x = self._letterbox(img)[None, ...]        # (1,3,H,W)
        return {self._input_name: x}

    def rewind(self) -> None:
        self._i = 0


def collect_calib_images(n: int) -> list[Path]:
    """
    优先从**训练集的 val 划分**取校准图，保证与训练分布一致。

    为什么不从测试集取：测试集是评估用的，拿它做校准属于信息泄漏，
    即使只用于统计量也不干净。val 集足够代表分布。
    """
    val_dir = DATASET_DIR / "images" / "val"
    if val_dir.exists():
        imgs = list_images(val_dir)
        if imgs:
            log(f"  校准集来源: {val_dir}  共 {len(imgs)} 张，取前 {min(n, len(imgs))}")
            return imgs[:n]
    log(f"  !! 未找到 {val_dir}，回退到训练集")
    tr = DATASET_DIR / "images" / "train"
    return list_images(tr)[:n] if tr.exists() else []


# --------------------------------------------------------------------------
# 步骤 3：ONNX Runtime 静态量化（主路线）
# --------------------------------------------------------------------------
def quantize_onnx_static(fp32_onnx: Path, int8_onnx: Path,
                         calib_images: list[Path],
                         imgsz: int = IMGSZ,
                         calib_method: str = "Entropy",
                         per_channel: bool = True,
                         calib_batch: int = 1,
                         max_arena_mb: int = 4096) -> Path | None:
    """
    主推路线：ORT 静态量化（QDQ 格式，激活与权重都是 INT8）。

    为什么选 Entropy 校准而不是默认的 MinMax：
      MinMax 用极值定标度，外墙图像里高光（阳光反射）和阴影会把范围拉得很宽，
      INT8 的有效分辨率被浪费。Entropy 用 KL 散度找最优截断，对小目标
      （本项目的裂缝/露筋是核心难点）更友好。代价是校准慢一些。

    ⚠️ 坑 6（v8s 才暴露出来，v8n 跑得过去所以一开始没发现）：
      ORT 的 `quantize_static` 会把**整个校准集**在图内做形状推断与张量驻留，
      显存/内存占用随 (模型参数量 × 校准图数) 增长。
      v8n（3.0M 参数）配 150 张能过；换成 v8s（11.1M 参数）后直接 OOM：

        onnxruntime::BFCArena::AllocateRawInternal
        Failed to allocate memory for requested buffer of size 983040

      注意报错发生在 **Conv 节点**、请求的 buffer 只有 983040 字节（约 1MB）——
      「只差 1MB」说明不是真缺内存，而是 ORT 默认 arena 上限吃满后
      连零头都分配不出来。
      → 对策有两条，**本函数同时采用**：
        (a) `sess_options.enable_cpu_mem_arena = False` + 降低 arena 上限，
            让 ORT 不用预分配大块 arena，改为按需分配；
        (b) 把校准图数据**分批喂**（`calib_batch`），并显式 `gc.collect()`，
            避免 Python 侧同时持有整批 float32 张量。
    """
    from onnxruntime.quantization import (
        CalibrationMethod, QuantFormat, QuantType,
        quantize_static,
    )

    global _LAST_QUANT_ERROR
    _LAST_QUANT_ERROR = None

    log(f"  静态量化: {fp32_onnx.name} -> {int8_onnx.name}")
    log(f"    校准法={calib_method}  per_channel={per_channel}  "
        f"校准图={len(calib_images)}  batch={calib_batch}  "
        f"arena<= {max_arena_mb}MB")

    method = {
        "MinMax": CalibrationMethod.MinMax,
        "Entropy": CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
        "Distribution": CalibrationMethod.Distribution,
    }.get(calib_method, CalibrationMethod.Entropy)

    reader = YoloCalibReader(calib_images, imgsz=imgsz, max_n=len(calib_images))
    # 从 onnx 图里读出真实输入名（可能是 images 也可能带前缀）
    try:
        import onnx
        m = onnx.load(str(fp32_onnx), load_external_data=False)
        reader.set_input_name(m.graph.input[0].name)
    except Exception:
        pass

    # ---- 坑 6 对策 (a)：控制 SessionOptions 与内存环境 ----
    # 注意：不能把自定义 SessionOptions 直接塞进 quantize_static ——
    # 该接口内部会自己建 session，传进去会被忽略或报错。
    # 真正生效的是**环境变量**（ORT 在首次创建 session 时读取并缓存），
    # 已在本模块 main() 最开头设置：
    #     OMP_NUM_THREADS=8             限制线程数 → 降低峰值内存
    #     ORT_DISABLE_ALLOCATOR_ARENA=1 关闭 arena 预分配
    #
    # ⚠️ 但实测这两条**不足以解决 v8s 的 OOM**（报错从
    # "Failed to allocate ... 983040" 变成 Concat 节点的 "bad allocation"，
    # 只是换了失败位置）。真正的根因见下方 extra_options 的注释。

    ensure_dirs(int8_onnx.parent)

    # ---- 坑 7（真正的根因）：校准阶段默认开 8 线程并行跑整个校准集 ----
    # ORT 的 `CalibrationContext` 默认会用多线程把校准数据**并发**送进
    # 推断 session，每个线程持有一份中间张量。对 v8s（11.1M 参数、
    # 640×640 输入、28.4 GFLOPs）来说，单次前向的中间激活就有数百 MB，
    # 8 路并发直接把地址空间打爆 → "bad allocation"。
    #
    # `quantize_static` 的 `extra_options` 里可以传
    # `CalibMaxBatchSize` / `CalibExtraThreads`，但**必须先禁用
    # 默认的并行校准**，即把额外线程数压到 0。
    extra = {
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
        # 关键：校准阶段只用主线程（ORT 内部名是 CalibExtraThreads）
        "CalibExtraThreads": 0,
        "CalibMaxBatchSize": calib_batch,
    }

    try:
        quantize_static(
            model_input=str(fp32_onnx),
            model_output=str(int8_onnx),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            reduce_range=False,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=method,
            op_types_to_quantize=None,
            use_external_data_format=False,
            extra_options=extra,
        )
    except Exception as e:
        # 参数不被当前 ORT 版本接受时，退化为最小参数集再试一次
        log(f"    带 extra_options 失败（{type(e).__name__}），退化为默认参数重试")
        try:
            quantize_static(
                model_input=str(fp32_onnx),
                model_output=str(int8_onnx),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                per_channel=per_channel,
                reduce_range=False,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=method,
                op_types_to_quantize=None,
                use_external_data_format=False,
            )
        except Exception as e2:
            # ⚠️ 必须在这里**吞掉异常并返回 None**，让调用方拿到 None
            # 后能走「降校准集重试」分支。此前这个异常直接向上抛穿，
            # 导致重试逻辑根本没机会执行 —— 已实测踩到。
            log(f"  !! 静态量化失败: {e2!r}")
            # 把原文留给调用方如实归因，别让它只能猜「大概是 OOM」
            _LAST_QUANT_ERROR = f"{type(e2).__name__}: {e2}"
            return None

    if not int8_onnx.exists():
        log("  !! 量化未产出文件")
        _LAST_QUANT_ERROR = "quantize_static 正常返回但未产出 INT8 文件"
        return None
    mb = int8_onnx.stat().st_size / 1024 / 1024
    log(f"    -> {int8_onnx.name} ({mb:.2f} MB)")
    return int8_onnx


def preprocess_onnx(fp32_onnx: Path) -> Path | None:
    """
    量化前的形状推断（坑 4）。

    ORT 官方要求：静态量化前跑 quant_pre_process，否则部分算子因
    无法推断形状而静默跳过量化。产出文件体积看起来正常，实际没量化到主干。
    这个坑很隐蔽，记录在此。
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process
    out = fp32_onnx.with_name(fp32_onnx.stem + "_preprocessed.onnx")
    try:
        quant_pre_process(
            input_model_path=str(fp32_onnx),
            output_model_path=str(out),
            skip_optimization=False,
            skip_symbolic_shape=False,
            auto_merge=True,
        )
        log(f"  形状推断完成: {out.name}")
        return out if out.exists() else None
    except Exception as e:
        log(f"  !! 形状推断失败（回退用原图量化）: {e!r}")
        return None


# --------------------------------------------------------------------------
# 步骤 4：测量（体积 / 延迟 / 精度）
# --------------------------------------------------------------------------
def measure_latency_onnx(onnx_path: Path, imgsz: int = IMGSZ,
                         n_warmup: int = 10, n_run: int = 50) -> dict:
    """
    用 ONNX Runtime 测纯推理延迟（不含前后处理）。

    诚实标注：这里测的是 **CPU** 上的 ORT 延迟。
    GPU 侧需 TensorRT，本项目不做（本机无 TensorRT 环境）。
    报告里必须写清测的是什么硬件，否则「<100ms」是没法比较的。
    """
    import onnxruntime as ort
    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(onnx_path), sess_opt, providers=providers)
    except Exception as e:
        return {"error": repr(e)}
    inp = sess.get_inputs()[0]
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    feed = {inp.name: x}
    for _ in range(n_warmup):
        sess.run(None, feed)
    ts = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        sess.run(None, feed)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        "provider": providers[0],
        "input_name": inp.name,
        "input_shape": list(inp.shape),
        "n_run": n_run,
        "latency_ms_mean": round(float(np.mean(ts)), 2),
        "latency_ms_p50": round(float(ts[len(ts) // 2]), 2),
        "latency_ms_p95": round(float(ts[int(len(ts) * 0.95)]), 2),
        "latency_ms_min": round(float(ts[0]), 2),
    }


def measure_latency_pt(weights: Path, imgsz: int = IMGSZ,
                       n_warmup: int = 10, n_run: int = 50) -> dict:
    """测 PyTorch FP32 的 GPU 推理延迟（与 ONNX CPU 对比时说明硬件差异）。"""
    import torch
    if not torch.cuda.is_available():
        return {"error": "CUDA 不可用"}
    from ultralytics import YOLO
    model = YOLO(str(weights))
    net = model.model.cuda().eval()
    x = torch.rand(1, 3, imgsz, imgsz, device="cuda")
    with torch.no_grad():
        for _ in range(n_warmup):
            net(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n_run):
            t0 = time.perf_counter()
            net(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        "provider": "CUDA",
        "n_run": n_run,
        "latency_ms_mean": round(float(np.mean(ts)), 2),
        "latency_ms_p50": round(float(ts[len(ts) // 2]), 2),
        "latency_ms_p95": round(float(ts[int(len(ts) * 0.95)]), 2),
        "latency_ms_min": round(float(ts[0]), 2),
    }


def evaluate_onnx_precision(int8_onnx: Path, data_yaml: Path,
                            name: str, split: str = "val") -> dict:
    """
    评估 INT8 ONNX 的检测精度。

    做法：用 ultralytics 的 YOLO 包住 onnx 文件做 val。
    ultralytics 支持 `YOLO("xxx.onnx")` 直接推理（内部用 ORT）。
    这样能和 FP32 的 val 指标直接对比，「掉了多少 mAP」有据可依。

    ⚠️ 坑 8（**导致 INT8 精度被误报为 0.0 的真凶**）：
      ultralytics 在**第一次 predict/val** 时会构造 ONNX 后端，若 device 判定为 CUDA，
      它就去检查 `onnxruntime-gpu`（见本模块顶部「触发链」注释）。
      本机只装了 CPU 版 `onnxruntime`，于是它触发
      `AutoUpdate` 去 pip 安装 —— 在离线/受限网络下**静默失败**，
      随后 `model.val()` 每个类都返回 0，**报告里看起来就是「量化把模型毁了」**。

      实测证据：同一个 INT8 文件用 ORT 直接跑，输出
      `(1, 11, 8400) float32, min 0.0, max 636.84, mean 72.07, finite=True`
      —— **模型本身完全正常**，问题在评估路径。

      → 对策（2026-09-20 修正）：**不要**去装 `onnxruntime-gpu`（那正是毁掉本机 ORT
      共用的 `onnxruntime/` 目录的那一步）。正确做法是两件都不做与网络/安装相关的事：
        1) 显式 `device="cpu"` —— 使 `onnx.py:63` 的 `cuda` 判定为 False，
           依赖检查退化为只需 `onnxruntime`（本机已装）→ 不触发安装；
        2) `ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1` —— 调用时求值、包解析前返回，
           作为与导入顺序无关的强保证。
      两者已在模块顶部与下方函数内同时设置。
    """
    import os
    os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"
    os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"
    # 注：此处**不**设 `YOLO_OFFLINE` —— 它管的是「联网下载权重」，不是「pip 安装」，
    #     设了会给人「已经拦住安装」的错觉（曾因此误判过，见模块顶部）。

    from ultralytics import YOLO
    try:
        model = YOLO(str(int8_onnx), task="detect")
        # ⚠️ `device="cpu"` 同样不可删：`Model.val` 在 `model.py:618` 把 kwargs 并进 `args`
        #    后传给 validator，`validator.py:186` 才 `select_device(self.args.device)`
        #    去构造 AutoBackend —— 所以写在 val() 里同样是构造之前生效的。
        metrics = model.val(data=str(data_yaml), split=split, imgsz=IMGSZ,
                            batch=1, device="cpu", workers=0,
                            conf=0.001, iou=0.6, plots=False, verbose=False)
        out = {
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "split": split,
        }
        # 全 0 是「评估路径坏了」的典型症状，不是模型坏了。
        # 必须显式标注，否则报告会误写成「量化导致精度归零」。
        if out["mAP50"] == 0.0 and out["mAP50_95"] == 0.0:
            out["warning"] = (
                "全部指标为 0：这是评估路径异常（ultralytics 在第一次 val 构造 ONNX 后端时"
                "把 device 判成 CUDA，于是检查 onnxruntime-gpu 并试图 AutoUpdate）。"
                "**不代表模型失效**。"
                "已用 ORT 直跑验证该模型输出正常（1,11,8400）。"
                "正确做法：给 val 显式传 device='cpu'，并设 "
                "ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1 抑制依赖检查；"
                "⚠️ 千万不要去装 onnxruntime-gpu —— 它与 CPU 版共用 "
                "onnxruntime/ 包目录，会直接覆盖本机 ORT 导致整条推理链报废。")
        return out
    except Exception as e:
        return {"error": repr(e)}


def sanity_check_int8(fp32_onnx: Path, int8_onnx: Path,
                      n_images: int = 8, conf: float = 0.05) -> dict:
    """
    INT8 交付前的**冒烟体检**：直接用 ORT 前向，比较 FP32 与 INT8 的
    原始输出张量是否还在合理范围。

    为什么必须有这一步（而不是只看 mAP）：
      ultralytics 的 `model.val()` 走 ONNX 时可能因环境问题静默返回全 0
      （见 evaluate_onnx_precision 的坑 8），**无法区分
      「评估坏了」和「模型坏了」**。
      本函数绕开 ultralytics，只做 ORT 前向 + 张量统计，判据是几何性的：
        - 输出是否 finite
        - 输出最大值是否还在合理量级（YOLO 输出含框坐标，几百是正常的）
        - INT8 与 FP32 的输出统计量是否同量级

    实测价值：本项目 v8s 用 32 张校准图量化出的 INT8 模型，
      体积 11.11MB（指标达标），但 ORT 直跑发现 max 虽正常、
      置信度通道却全为极小值 → 后续用 ultralytics 复测确认
      **conf>=0.001 时检出 0 个**（FP32 为 21 个）。
      若没有这一步，仅看体积会误判为「达标交付」。
    """
    import numpy as np
    import onnxruntime as ort

    out: dict = {}
    for tag, p in (("fp32", fp32_onnx), ("int8", int8_onnx)):
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(p), so,
                                        providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            stats = []
            for k in range(n_images):
                # 固定种子生成可复现的输入，避免每次巡检结论不同
                rng = np.random.default_rng(1000 + k)
                x = rng.random((1, 3, IMGSZ, IMGSZ), dtype=np.float32)
                y = np.asarray(sess.run(None, {name: x})[0])
                stats.append({
                    "max": float(y.max()),
                    "min": float(y.min()),
                    "mean": float(y.mean()),
                    "finite": bool(np.isfinite(y).all()),
                    "shape": list(y.shape),
                })
            out[tag] = {
                "all_finite": all(s["finite"] for s in stats),
                "shape": stats[0]["shape"],
                "max_of_max": max(s["max"] for s in stats),
                "mean_of_mean": round(float(np.mean([s["mean"] for s in stats])), 4),
            }
        except Exception as e:
            out[tag] = {"error": repr(e)}

    # 判定：**必须看「置信度通道」，不能只看整张量均值**
    #
    # ⚠️ 这里踩过一个会让体检形同虚设的坑：
    #   最初版本比较的是**整个输出张量的 mean**。
    #   YOLO 输出 (1, 4+nc, 8400) 里前 4 个通道是**框坐标**（数值几十到几百），
    #   后面 nc 个通道才是类别置信度（数值 0~1）。
    #   框坐标把均值彻底淹没，于是：
    #     废掉的 INT8 模型  整张量 mean=71.83 vs FP32 72.14 → ratio 0.9958 → 判 "ok"
    #   而同一个模型实际是：conf>=0.001 时检出 **0** 个（FP32 为 506 个/10 图）。
    #   **体检给出"ok"、模型却是废的** —— 这种假阴性比不体检更危险。
    #
    # → 改为只取**类别置信度通道**（第 4 通道起）做统计，
    #   并以「是否存在接近 1 的置信度」作为核心判据。
    out: dict = {}
    for tag, p in (("fp32", fp32_onnx), ("int8", int8_onnx)):
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(p), so,
                                        providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            conf_maxs, conf_means = [], []
            for k in range(n_images):
                rng = np.random.default_rng(1000 + k)
                x = rng.random((1, 3, IMGSZ, IMGSZ), dtype=np.float32)
                y = np.asarray(sess.run(None, {name: x})[0])
                # y: (1, 4+nc, N) -> 取置信度通道
                c = y[0, 4:, :]
                conf_maxs.append(float(c.max()))
                conf_means.append(float(c.mean()))
                last_finite = bool(np.isfinite(y).all())
            out[tag] = {
                "all_finite": last_finite,
                "shape": list(y.shape),
                "conf_channels": int(y.shape[1]) - 4,
                "conf_max": round(float(np.max(conf_maxs)), 4),
                "conf_mean": round(float(np.mean(conf_means)), 6),
            }
        except Exception as e:
            out[tag] = {"error": repr(e)}

    # 核心判据：FP32 在随机输入下也会有一批高置信度响应（模型对噪声的固有输出）；
    # INT8 若把置信度通道整体压死（conf_max 接近 0），即判定为量化失败。
    try:
        f, i = out.get("fp32", {}), out.get("int8", {})
        fm, im_ = f.get("conf_max", 0.0), i.get("conf_max", 0.0)
        out["conf_max_ratio"] = round(float(im_ / fm), 4) if fm else None
        # 双阈值：相对比值太低 **或** INT8 绝对置信度几乎为 0，都算失败
        if fm and (im_ / fm) <= 0.1:
            out["verdict"] = ("FAIL: INT8 置信度通道被压死"
                              f"（conf_max {im_} vs FP32 {fm}）。"
                              "放大校准集或换校准法重做。")
        elif im_ < 0.01:
            out["verdict"] = (f"FAIL: INT8 置信度接近 0（conf_max={im_}），"
                              "模型不可用。")
        else:
            out["verdict"] = "ok"
    except Exception:
        out["verdict"] = "unknown"
    return out


def verify_int8_detects(fp32_onnx: Path, int8_onnx: Path,
                        n_images: int = 10, conf: float = 0.001) -> dict:
    """
    **决定性验证**：用真实图像跑 ultralytics 推理，比较 FP32 与 INT8 的检出数。

    为什么张量统计（sanity_check_int8）不够：
      本项目实测到一个**假阴性** —— 张量统计判 "ok"（conf_max 比值正常），
      但同一个 INT8 模型在 10 张真实图上检出 **0** 个目标（FP32 为 506 个）。
      张量层面的统计量无法替代端到端行为验证。

    所以这一层必须存在，且用**最宽松的阈值 conf=0.001** ——
    只要模型还能检出任何东西，就说明量化没有把模型彻底打坏。
    conf=0.001 下仍为 0 → 模型确定不可用。

    诚实说明：本函数需要 ultralytics + 一点额外耗时（约 10~20 秒），
    但它是唯一能真正回答「量化后的模型还能不能用」的检验。
    """
    import os
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"
    os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"
    # 逐行核实的说明（ultralytics 8.4.138 源码，2026-09-20）：
    #   - `YOLO_OFFLINE=1` **不能**阻止 ultralytics 的自动安装：它管的是「权重下载」，
    #     安装的总闸是 `checks.py:625 if install and AUTOINSTALL`。这两行里只有
    #     `ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS` 与 provider 是真正有用的。
    #   - `YOLO("x.onnx")` 这一句**本身不会**触发依赖检查：`model.py:253-257` 对非 .pt
    #     文件只把 `self.model` 设成路径字符串，此时还没有 ORT session。
    #   - 真正的触发点是**下一行的 predict**：
    #       model.py:512  args = {**self.overrides, **custom, **kwargs}
    #       model.py:520  predictor = ...(overrides=args)  → setup_model
    #       predictor.py:429  AutoBackend(device=select_device(self.args.device))
    #       onnx.py:63   cuda = ... and self.device.type != "cpu"
    #       onnx.py:76   check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
    #     所以 **kwargs 里传 `device="cpu"` 是在 AutoBackend 构造之前生效的**，
    #     它让 cuda=False，依赖检查退化成只需 `onnxruntime`（本机已装）→ 不安装。
    #     这是本脚本据以自保的关键一行，**不可删**。

    try:
        from ultralytics import YOLO
    except Exception as e:
        return {"error": repr(e)}

    val_dir = DATASET_DIR / "images" / "val"
    imgs = list_images(val_dir)[:n_images]
    if not imgs:
        return {"error": f"{val_dir} 下无图像"}

    res: dict = {"n_images": len(imgs), "conf_threshold": conf}
    for tag, p in (("fp32", fp32_onnx), ("int8", int8_onnx)):
        try:
            m = YOLO(str(p), task="detect")
            total, mx = 0, 0.0
            for f in imgs:
                # ⚠️ `device="cpu"` 不可删：它是首次 predict 时构造 AutoBackend 的输入，
                #    决定 onnx.py:63 的 cuda 判定，进而决定会不会去装 onnxruntime-gpu。
                r = m.predict(str(f), conf=conf, imgsz=IMGSZ, verbose=False,
                              device="cpu")[0]
                n = 0 if r.boxes is None else len(r.boxes)
                total += n
                if n:
                    mx = max(mx, float(r.boxes.conf.max()))
            res[tag] = {"total_detections": total,
                        "max_conf": round(mx, 4)}
        except Exception as e:
            res[tag] = {"error": repr(e)}

    f, i = res.get("fp32", {}), res.get("int8", {})
    if isinstance(f.get("total_detections"), int) and \
       isinstance(i.get("total_detections"), int):
        if f["total_detections"] > 0 and i["total_detections"] == 0:
            res["verdict"] = ("FAIL: FP32 能检出但 INT8 检出 0 —— "
                              "量化已使模型失效，不可交付")
        elif f["total_detections"] == 0:
            res["verdict"] = "unknown: FP32 本身就检不出，检查权重或路径"
        else:
            ratio = i["total_detections"] / max(f["total_detections"], 1)
            res["recall_retention"] = round(float(ratio), 4)
            res["verdict"] = ("ok" if ratio >= 0.5 else
                              f"SUSPECT: INT8 检出数仅为 FP32 的 {ratio:.1%}")
    else:
        res["verdict"] = "unknown"
    return res


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def _ascii_stage() -> Path:
    """
    创建一个纯 ASCII 的临时工作目录（见文档「坑 5」）。

    ORT 的 quantize_static 在含中文的路径下必失败，所以整个量化流程
    必须在 ASCII 目录里跑，最后再把产物复制回项目目录。
    """
    import tempfile
    base = Path(tempfile.gettempdir()) / "wall_defect_quant"
    ensure_dirs(base)
    return base


def _is_ascii_path(p: Path) -> bool:
    try:
        str(p.resolve()).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def quantize_one(weights: Path, data_yaml: Path, calib_n: int,
                 do_eval: bool, calib_method: str) -> dict:
    name = weights.parent.parent.name if weights.parent.name == "weights" else weights.stem
    log("=" * 70)
    log(f"[{name}] 量化流程开始")
    log("=" * 70)

    # 自检报告 P1-1：该字段是**请求值**，实际用了几张由 OOM 重试决定
    # （记在 rec["calib_used"]）。原名 calib_n 会被误读为实际值，故改名。
    rec: dict = {"name": name, "weights": str(weights), "imgsz": IMGSZ,
                 "calib_method": calib_method,
                 "calib_n_requested": calib_n,
                 "calib_n_note": "请求的校准图张数；实际张数见 calib_used"
                                 "（OOM 时逐级下调，下界 MIN_CALIB_N）"}

    # 0) 原始 FP32 体积
    fp32_mb = weights.stat().st_size / 1024 / 1024
    rec["pt_fp32_mb"] = round(fp32_mb, 2)

    # 1) FP32 PyTorch GPU 延迟（基线）
    if do_eval:
        log("  [1/5] 测量 FP32 GPU 延迟基线")
        rec["latency_pt_fp32"] = measure_latency_pt(weights)

    # 全部量化中间产物走 ASCII 暂存区（坑 5）
    stage = _ascii_stage()
    log(f"  ASCII 暂存目录: {stage}")

    # 2) 导出 ONNX —— ONNX 导出本身中文路径没问题，
    #    但为省去再复制一次，直接导出到暂存区
    log("  [2/5] 导出 FP32 ONNX")
    fp32_onnx = stage / f"{name}_fp32.onnx"
    got = export_onnx(weights, fp32_onnx)
    if got is None:
        rec["status"] = "export_failed"
        return rec
    rec["onnx_fp32_mb"] = round(got.stat().st_size / 1024 / 1024, 2)

    # 3) 形状推断
    pre = preprocess_onnx(fp32_onnx)
    src_onnx = pre if pre is not None else fp32_onnx

    # 4) 静态量化（必须在 ASCII 路径下）
    log("  [3/5] ORT 静态量化 (INT8)")
    calib_imgs = collect_calib_images(calib_n)
    if not calib_imgs:
        rec["status"] = "no_calib_images"
        return rec
    staged_int8 = stage / f"{name}_int8.onnx"

    # ---- 坑 6/7 对策：OOM 时降校准集规模重试，但**降得太狠会毁掉模型** ----
    # 校准图数的下界是有物理意义的：Entropy 校准要在激活分布上做 KL 散度
    # 找最优截断，样本太少 → 分布估计偏差大 → 量化 scale 系统性偏错。
    #
    # 实测教训（**本项目踩到的最贵的一个坑**）：
    #   v8s640 在 150 张时 OOM → 自动降到 32 张 → 量化"成功"、
    #   体积 11.11MB 看着正常，但**推理输出全废**：
    #
    #     FP32-ONNX  conf>=0.001: n=21 detections, maxconf=0.843
    #     INT8-ONNX  conf>=0.001: n=0  detections, maxconf=0.000
    #
    #   —— 连置信度 0.001 都出不来，说明不是精度略降而是**输出完全崩坏**。
    #   体积指标（<10MB）能过，但模型不可用。这是典型的
    #   「**用降级换来的达标**」，在报告里必须如实说明，绝不能只报体积。
    #
    # → 因此设定 **MIN_CALIB_N = 64** 作为硬下界：
    #   宁可让量化失败并如实报告「v8s 的 INT8 量化未达标」，
    #   也不能交付一个体积好看但检不出东西的模型。
    #   （若确实需要更小校准集，应改用 MinMax 或 Percentile 校准法再评估。）
    MIN_CALIB_N = 64
    attempts = []
    for n_try in (len(calib_imgs), 128, 96, 64):
        n_try = min(n_try, len(calib_imgs))
        if n_try in attempts or n_try < MIN_CALIB_N:
            continue
        attempts.append(n_try)
        got8 = quantize_onnx_static(src_onnx, staged_int8, calib_imgs[:n_try],
                                    calib_method=calib_method)
        if got8 is not None:
            rec["calib_used"] = n_try
            if n_try < len(calib_imgs):
                # 同样不预设原因是 OOM：把上一轮的真实异常一并带出
                log(f"    （原定 {len(calib_imgs)} 张校准图失败，"
                    f"已降为 {n_try} 张后成功 —— 报告需注明）")
                if _LAST_QUANT_ERROR:
                    log(f"     上一轮失败原文: {_LAST_QUANT_ERROR}")
            break
        # ⚠️ 失败后必须主动回收（2026-09-20 补上）：
        #   ORT 的 session / arena 是 C++ 对象，Python 引用消失后**不保证立即释放**，
        #   要等 GC 才回收。本函数的 docstring 早就把「显式 gc.collect()」写成了
        #   对策 (b)，但**一直没真的实现**（全文只出现 `import gc` 缺失的状态）。
        #   实测后果：本机 15.7 GB 内存下，4 次重试的 Private 提交累积到约 18 GB，
        #   于是「降校准集重试」反而越试越糟。这里补上，让每次重试回到干净基线。
        gc.collect()
    else:
        got8 = None

    if got8 is None:
        rec["status"] = "quantize_failed"
        rec["attempts"] = attempts
        rec["min_calib_n"] = MIN_CALIB_N
        # ⚠️ 如实归因（2026-09-20 修正）：
        #   这里原本无条件写「均 OOM」，但失败原因**未必是内存**。
        #   实测踩到过一次 onnxruntime 安装损坏，抛的是 AttributeError
        #   （Python 层 1.30.0 调用本地 1.29.0 DLL 未导出的方法），
        #   与内存无关，却被这句日志误导去查内存。
        #   现在改为：打印真实异常原文；只有异常原文里确实出现内存字样时，
        #   才把它称作 OOM。
        err = _LAST_QUANT_ERROR
        rec["error"] = err
        log(f"  !! 量化失败：在 >= {MIN_CALIB_N} 张校准图的下界内均失败。")
        if err:
            log(f"     真实原因（最后一次尝试）: {err}")
        # ⚠️ 匹配必须**大小写不敏感**：真实的 OOM 原文是
        #    `MemoryError('bad allocation')` —— 首字母大写的 "Memory" 和
        #    小写的 "allocation"，用大小写敏感的朴素 `in` 检查会**判成非 OOM**
        #    （我自己第一版就踩了这个假阴性，实测在 16:26 那轮暴露）。
        _err_lc = (err or "").lower()
        _oom_keys = ("alloc", "memory", "out of memory", "oom", "bad_alloc")
        if err and any(k in _err_lc for k in _oom_keys):
            log("     ↑ 异常原文含内存相关字样，可判定为 OOM。")
        else:
            log("     ↑ 异常原文**不含**内存字样 —— 这不是 OOM，"
                "不要靠调校准集大小或加内存来解决。")
        log(f"     宁可报「未达标」，也不交付一个检不出目标的模型。")
        return rec

    # 复制回项目目录（交付物）
    int8_onnx = QUANT_DIR / f"{name}_int8.onnx"
    ensure_dirs(int8_onnx.parent)
    shutil.copy2(got8, int8_onnx)
    # FP32 基线也留一份，便于报告里说明「压缩了多少倍」
    fp32_keep = QUANT_DIR / f"{name}_fp32.onnx"
    shutil.copy2(fp32_onnx, fp32_keep)

    rec["onnx_int8_mb"] = round(int8_onnx.stat().st_size / 1024 / 1024, 2)
    rec["compression_vs_pt"] = round(fp32_mb / max(rec["onnx_int8_mb"], 1e-9), 2)
    rec["compression_vs_onnx_fp32"] = round(
        rec["onnx_fp32_mb"] / max(rec["onnx_int8_mb"], 1e-9), 2)
    # 报告口径说明：.pt 是 PyTorch 的压缩序列化格式，同一张网络存成 ONNX
    # 会更占空间（FP32 全展开），所以「vs .pt」的倍数天然偏小。
    # 要论证「INT8 省了多少」，应引用 compression_vs_onnx_fp32。
    rec["compression_note"] = ("compression_vs_onnx_fp32 才是量化收益的正确口径；"
                               "compression_vs_pt 因 .pt 本身已是压缩格式而偏小。")

    # 5) 延迟与精度（对项目目录里的最终文件做，确保交付物本身被测过）
    log("  [4/5] INT8 冒烟体检（ORT 直跑，绕开 ultralytics 评估路径）")
    rec["sanity_int8_vs_fp32"] = sanity_check_int8(fp32_keep, int8_onnx)
    verdict = rec["sanity_int8_vs_fp32"].get("verdict", "unknown")
    log(f"    verdict={verdict}  conf_max_ratio="
        f"{rec['sanity_int8_vs_fp32'].get('conf_max_ratio')}")

    log("  [5/5] 测量 INT8 延迟")
    rec["latency_onnx_int8"] = measure_latency_onnx(int8_onnx)

    # ---- 决定性验证：真实图像端到端检出对比 ----
    # 必须在延迟测量之后跑：这一步要加载 ultralytics 的推理会话，
    # 与 measure_latency_onnx 的纯 ORT session 混在一起会互相干扰计时。
    log("  [5b] 端到端检出验证（真实图像，FP32 vs INT8）")
    rec["verify_detects"] = verify_int8_detects(fp32_keep, int8_onnx)
    vd = rec["verify_detects"]
    log(f"    FP32 检出={vd.get('fp32', {}).get('total_detections')}  "
        f"INT8 检出={vd.get('int8', {}).get('total_detections')}  "
        f"verdict={vd.get('verdict')}")

    if do_eval:
        log("  [6/6] 评估 INT8 精度")
        rec["metrics_int8_val"] = evaluate_onnx_precision(int8_onnx, data_yaml, name)

    # 体检不过关 → 标记为失败，**不要把废模型当成交付物**
    verdict = str(rec["sanity_int8_vs_fp32"].get("verdict", ""))
    vdet = str(vd.get("verdict", ""))
    if verdict.startswith("FAIL") or vdet.startswith("FAIL"):
        rec["status"] = "int8_unusable"
        log(f"  !! INT8 验证未通过（张量体检={verdict}；端到端={vdet}）")
        log(f"     产物保留在 {int8_onnx} 供排查，但**不作为可交付模型**。")
        log(f"     —— 如实报告「该规格 INT8 未达标」，好过交付一个检不出东西的模型。")
        return rec

    rec["status"] = "ok"
    log(f"  [{name}] 完成: FP32 ONNX {rec['onnx_fp32_mb']}MB -> "
        f"INT8 {rec['onnx_int8_mb']}MB "
        f"(压缩 {rec['compression_vs_onnx_fp32']}x)")
    return rec


def main() -> None:
    ensure_dirs(QUANT_DIR)

    # ---- 坑 6：必须在**导入 onnxruntime 之前**设好内存相关环境变量 ----
    # ORT 的 C++ 层在首次加载时会缓存这些配置，之后再设无效。
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "8")          # 限制线程数，降峰值内存
    os.environ.setdefault("ORT_DISABLE_ALLOCATOR_ARENA", "1")

    data_yaml = Path(_argv_str("data", str(DATASET_DIR / DATASET_YAML_NAME)))
    calib_n = _argv_int("calib-n", DEFAULT_CALIB_N)
    calib_method = _argv_str("calib-method", "Entropy") or "Entropy"
    do_eval = not _argv_bool("no-eval")
    do_all = _argv_bool("all")

    if not data_yaml.exists():
        log(f"!! 数据集配置不存在: {data_yaml}")
        return

    # 收集待量化权重
    targets: list[Path] = []
    # 新增（2026-09-20）：--runs=v11s640,v8n640 —— 只量化**指定的 run**。
    # 为什么不直接用 --all：`--all` 会扫到当时正在**重训**的 run，它的
    # best.pt 每个 epoch 都在被覆写（实测 v11s640/weights/best.pt 写入时刻
    # 与训练进度同步），量化这种半成品既无意义、又可能读到写了一半的文件，
    # 而且报告会因此混入不属于本轮口径的数字。
    # 有了 --runs，就能在「有 run 正在训练」的前提下，仍然产出**单一、口径一致**
    # 的 quant_report.json（这正是「重跑一次量化、以那一轮产物统一报告」所需）。
    run_names = _argv_list("runs")
    if run_names:
        for rn in run_names:
            best = TRAIN_DIR / rn / "weights" / "best.pt"
            if best.exists():
                targets.append(best)
            else:
                log(f"!! --runs 指定的 {rn} 没有 best.pt，已跳过")
        log(f"--runs: 命中 {len(targets)} 个权重 -> "
            f"{[t.parent.parent.name for t in targets]}")
    elif do_all:
        for run_dir in sorted(TRAIN_DIR.glob("*")):
            best = run_dir / "weights" / "best.pt"
            if best.exists():
                targets.append(best)
        log(f"--all: 发现 {len(targets)} 个已训练权重")
    else:
        w = _argv_str("weights")
        if not w:
            # 默认用主力配置。2026-09-20 由 v8s640 切换为 v11s640：
            #   V3 独立测试集（285 图/1646 实例/7 类）：mAP50 0.7230 vs 0.7356
            #   （⚠️ v8s 反而高 0.0126 但 < 0.03 判读阈值 ⇒ 判不出高下；
            #    旧注释写的「0.7628 vs 0.6766，三项占优」是 V2 口径，已作废，2026-09-22 更正），
            #   因此真实依据是 mAP50-95 0.44105 vs 0.43244、体积 18.33 vs 21.51 MB、
            #   假阳 109 vs 119（V2 口径）。
            #   切换依据与合规性见 README §训练、07_report/技术报告.md §5.3.2。
            w = str(WEIGHTS_DIR / "v11s640_best.pt")
        p = Path(resolve_weight(w))
        if not p.exists():
            log(f"!! 权重不存在: {p}")
            log("   请先训练，或用 --weights= 指定")
            return
        targets.append(p)

    if not targets:
        log("!! 没有可量化的权重")
        return

    records = [
        quantize_one(w, data_yaml, calib_n, do_eval, calib_method)
        for w in targets
    ]

    report = {
        "note": ("INT8 量化报告。latency_* 字段测的是不同硬件："
                 "latency_pt_fp32 是 GPU(CUDA)，latency_onnx_int8 是 CPU(ORT)。"
                 "两者不可直接相减，报告引用时必须写明硬件。"),
        "imgsz": IMGSZ,
        "calib_n_requested": calib_n,
        "calib_n_note": "请求的校准图张数；各 run 的实际张数见 runs[].calib_used",
        "calib_method": calib_method,
        "data_yaml": str(data_yaml),
        "results": records,
    }
    out = QUANT_DIR / "quant_report.json"
    dump_json(report, out)
    log(f"报告已写入 {out}")

    log("=" * 70)
    for r in records:
        log(f"{r['name']:12s} {r.get('status','?'):16s} "
            f"FP32={r.get('pt_fp32_mb','-')}MB -> "
            f"INT8={r.get('onnx_int8_mb','-')}MB "
            f"压缩={r.get('compression_vs_pt','-')}x")
    log("=" * 70)


if __name__ == "__main__":
    main()
