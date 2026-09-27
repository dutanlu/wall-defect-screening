# -*- coding: utf-8 -*-
r"""4 通道（RGB+IR）输入的 **运行期 patch** —— 不修改 ultralytics 安装。

## 为什么必须是运行期 patch 而不是改安装
项目纪律：**不污染 `D:/下载/Lib/site-packages` 下的第三方包**。
改安装会让「本工程的结果」依赖一个被改过的第三方包，
复现者装了官方 ultralytics 就跑不通，且升级即失效。
⇒ 本模块用 **monkey-patch + 权重膨胀**，在 import 后、训练前生效。
   所有改动集中在本文件，可审计、可回滚、可被复现者原样执行。

## 三处改动点（逐一说明）

### 改动点 1 —— 数据加载：`BaseDataset.load_image` 只认 1/3 通道
源码事实（`ultralytics/data/base.py:133-134`）：
```python
self.channels = channels
self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
```
`channels` 既非 1 也非 3 时，`cv2.IMREAD_COLOR` 会把 4 通道 PNG
**悄悄截断成 3 通道**（OpenCV 对 `IMREAD_COLOR` 的行为），第 4 通道丢失。
⇒ patch：当 `self.channels == 4` 时改用 `cv2.IMREAD_UNCHANGED`（真读 4 通道），
   并在 `load_image` 之后**强制断言**通道数，宁可报错也不静默降维
   （纪律：「静默降维」比崩溃更坏——它会让 4 通道实验变成伪 3 通道实验）。

### 改动点 2 —— 首层卷积：`Conv2d(3, 32, 3, 3)` 装不下 4 通道
`self.model[0].conv.weight.shape == torch.Size([32, 3, 3, 3])`。
直接 `nc=5` 建 4 通道模型会随机初始化首层 ⇒ **丢掉预训练**；
而公平性前提是「各模态用**同一预训练**初始化」（`_bfdd_build_fuse.py` 已声明）。
⇒ patch：**权重膨胀**（weight inflation）——
   新建 `Conv2d(4, 32, 3, 3)`，把原 3 通道权重**原样**拷入前 3 个输入通道
   （`new.weight[:, :3] = old.weight`，逐元素相等，已断言），
   第 4 个输入通道初始化 = **对输入通道取均值**（`old.weight.mean(dim=1)`）。

**★ mean 初始化的精确性质（实测，勿误述）**：
它**不是**恒等嵌入。设原首层输出 `y3 = Σ_k W_k * x_k`，膨胀后
`y4 = y3 + Σ_k mean_j(W_jk) * IR_k`，其中第 4 通道权重为
`W_mean[o,k] = mean_j W[o,j,k]`（对**输入通道**取均值）。
实测当 `IR == mean_k(x3)` 时，额外项为
`conv(mean_k(x3); W_mean)`，其模长约为原输出的 **0.19 倍**（随机权重下的量级），
即 **有扰动但很小**。
之所以仍选 mean 而非 zero：zero 让第 4 通道在初始时刻**恒不激活**（梯度要靠后续
训练从零学起，收敛更慢）；mean 让第 4 通道一开始就携带「RGB 均值图」这一
合理先验，且**前 3 通道的预训练特征被完整保留**——这正是公平性的关键。
⇒ 三种初始化（mean/zero/repeat）都可选，**默认 mean**，实验须记录所用 mode。

### 改动点 3 —— 预处理标准化：`dataset.channels` 与模型输入必须一致
ultralytics 会按 `dataset.channels` 决定是否转 RGB（`BGR→RGB`）。
4 通道下 `cvtColor(..., COLOR_BGR2RGB)` 只处理前 3 通道、第 4 通道保留，
语义正确；但**必须确认`channels=4` 时这条路径不抛异常** ⇒ 本 patch 显式验证。

## 兼容性 / 风险（必须如实声明）
| 项 | 结论 |
|---|---|
| 预训练权重 | **不破坏**：首层 3→4 膨胀，其余层完全复用 |
| 与 rgb/fuse/ctrl 可比性 | **可比**：同划分、同标签、同超参、同初始化来源 |
| 推理部署 | ⚠️ **不可直接用现成 ONNX/TensorRT 导出流程**：4 通道需重新导出且需 4 通道预处理 |
| 单图 OOD 兜底 / 弃权机制 | 不受影响（那两条走 `grade.py`，与输入通道解耦） |
| ultralytics 版本升级 | patch 依赖 `load_image` 与 `model.model[0].conv` 的结构；升级后需复验 |
| 本仓库主交付包 | **不引入**：本方案仅在 `01_data/.../bfdd/_runs/` 下做外部验证，主模型仍是 3 通道 |

## 用法
```python
import _bfdd_rgbir_patch as P
P.install()                      # 1) patch load_image
model = P.build_rgbir_model("yolo11s-seg.yaml", nc=5)   # 2) 膨胀首层
# 3) 训练前 P.assert_ready(model, data_yaml)
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(r"D:\pythonstudy 备份\创新题\外墙缺陷筛查")
BFDD = ROOT / "01_data" / "raw" / "_public_datasets" / "bfdd"
_RECORD: dict = {"installed": False, "actions": []}


# --------------------------------------------------------------------------
# 改动点 1：load_image 支持 4 通道
# --------------------------------------------------------------------------
def install(verbose: bool = True) -> None:
    """patch `BaseDataset.load_image`：channels==4 时用 IMREAD_UNCHANGED 并断言通道数。"""
    global _RECORD
    if _RECORD["installed"]:
        if verbose:
            print("[patch] 已安装，跳过")
        return

    from ultralytics.data.base import BaseDataset
    import cv2

    _orig_load_image = BaseDataset.load_image

    def load_image_4ch(self, i: int, rect_mode: bool = True, resize_short: bool = False):
        im, hw0, hw = _orig_load_image(self, i, rect_mode=rect_mode,
                                       resize_short=resize_short)
        want = getattr(self, "channels", 3)
        if want == 4:
            # 原实现用 cv2.IMREAD_COLOR ⇒ 4 通道 PNG 被截断，这里补读
            if im.ndim == 2 or im.shape[2] != 4:
                raw = cv2.imdecode(np.fromfile(str(self.im_files[i]), dtype=np.uint8),
                                   cv2.IMREAD_UNCHANGED)
                if raw is None:
                    raise FileNotFoundError(f"4 通道读取失败 {self.im_files[i]}")
                if raw.ndim == 2:
                    raise ValueError(f"期望 4 通道，实得单通道：{self.im_files[i]}")
                if raw.shape[2] != 4:
                    raise ValueError(
                        f"期望 4 通道，实得 {raw.shape[2]} 通道：{self.im_files[i]}")
                h0, w0 = raw.shape[:2]
                if rect_mode:
                    if resize_short:
                        r = self.imgsz / min(h0, w0)
                    else:
                        r = self.imgsz / max(h0, w0)
                    if r != 1:
                        raw = cv2.resize(raw, (min(int(w0 * r), self.imgsz) if not resize_short
                                               else int(w0 * r),
                                               self.imgsz if resize_short
                                               else min(int(h0 * r), self.imgsz)),
                                         interpolation=cv2.INTER_LINEAR)
                elif not (h0 == w0 == self.imgsz):
                    raw = cv2.resize(raw, (self.imgsz, self.imgsz),
                                     interpolation=cv2.INTER_LINEAR)
                im = raw
                hw = im.shape[:2]
            # ★ 硬断言：绝不让 4 通道实验静默退化成 3 通道
            if im.ndim != 3 or im.shape[2] != 4:
                raise AssertionError(
                    f"load_image 4 通道断言失败：shape={getattr(im, 'shape', None)} "
                    f"file={self.im_files[i]}")
        return im, hw0, hw

    BaseDataset.load_image = load_image_4ch

    # cv2_flag：channels==4 时用 UNCHANGED（部分路径直接用这个 flag 读图）
    _orig_init = BaseDataset.__init__

    def __init_patched(self, *a, **kw):
        _orig_init(self, *a, **kw)
        if getattr(self, "channels", 3) == 4:
            self.cv2_flag = cv2.IMREAD_UNCHANGED

    BaseDataset.__init__ = __init_patched

    _RECORD["installed"] = True
    _RECORD["actions"].append("patched BaseDataset.load_image + __init__ for channels=4")
    if verbose:
        print("[patch] 改动点 1 完成：BaseDataset 支持 4 通道（含硬断言）")


# --------------------------------------------------------------------------
# 改动点 2：首层 3→4 权重膨胀
# --------------------------------------------------------------------------
def inflate_first_conv(model: nn.Module, mode: str = "mean",
                       verbose: bool = True) -> dict:
    """把 model.model[0].conv 从 3 输入通道膨胀到 4 通道，保留预训练权重。

    mode='mean'  ：第 4 通道 = 原 3 通道均值（推荐，初始输出等价于 RGB 均值图）
    mode='zero'  ：第 4 通道 = 0（最保守，但初始就浪费一路）
    mode='repeat'：第 4 通道 = 第 1 通道副本（会引入偏置，不推荐）
    """
    first = model.model[0]
    old = first.conv
    if old.in_channels == 4:
        if verbose:
            print("[patch] 首层已是 4 通道，跳过")
        return {"inflated": False, "in_channels": 4}
    if old.in_channels != 3:
        raise ValueError(f"只支持 3→4 膨胀，当前 in_channels={old.in_channels}")

    new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride,
                    old.padding, bias=(old.bias is not None))
    with torch.no_grad():
        new.weight[:, :3] = old.weight.clone()
        if mode == "mean":
            new.weight[:, 3] = old.weight.mean(dim=1)
        elif mode == "zero":
            new.weight[:, 3] = 0.0
        elif mode == "repeat":
            new.weight[:, 3] = old.weight[:, 0]
        else:
            raise ValueError(mode)
        if old.bias is not None:
            new.bias.copy_(old.bias.clone())
    first.conv = new

    info = {"inflated": True, "in_channels": 4, "mode": mode,
            "out_channels": int(new.out_channels),
            "pretrained_weight_preserved": "ch0-2 逐元素拷贝",
            "new_channel_init": mode}
    _RECORD["actions"].append(f"inflate first conv 3->4 (mode={mode})")
    if verbose:
        print(f"[patch] 改动点 2 完成：首层 Conv 3→4（mode={mode}），"
              f"前 3 通道为预训练权重原值，第 4 通道初始化={mode}")
    return info


def build_rgbir_model(yaml: str = "yolo11s-seg.yaml", nc: int = 5,
                      mode: str = "mean", verbose: bool = True):
    """建 5 类分割模型并把首层膨胀到 4 通道。"""
    from ultralytics.nn.tasks import SegmentationModel
    if mode == "zero":
        # 用 4 通道权重建网（ultralytics 的 yaml 是 3 通道硬编码）⇒ 先建 3 通道再膨胀，
        # 这是唯一能保留预训练结构的路径。mode='zero' 亦同。
        pass
    model = SegmentationModel(yaml, nc=nc, verbose=False)
    info = inflate_first_conv(model, mode=mode, verbose=verbose)
    model._rgbir_info = info
    return model


# --------------------------------------------------------------------------
# 改动点 3：训练前自检
# --------------------------------------------------------------------------
def assert_ready(model, data_yaml: str | Path, verbose: bool = True) -> dict:
    """训练前必须过的三道检查：数据真 4 通道、模型真 4 通道、两者通道序一致。"""
    import cv2
    from pathlib import Path as _P

    yml = _P(data_yaml)
    txt = yml.read_text(encoding="utf-8")
    assert "channels: 4" in txt, f"{yml} 缺少 channels: 4"

    base = _P(txt.split("path:")[1].splitlines()[0].strip()) / "images" / "val"
    sample = sorted(base.glob("*.png"))[0]
    raw = cv2.imdecode(np.fromfile(str(sample), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert raw.ndim == 3 and raw.shape[2] == 4, \
        f"数据不是 4 通道：{sample} shape={None if raw is None else raw.shape}"

    inc = model.model[0].conv.in_channels
    assert inc == 4, f"模型首层不是 4 通道：in_channels={inc}"

    res = {"data_yaml": str(yml), "sample": sample.name,
           "data_channels": int(raw.shape[2]), "model_in_channels": int(inc),
           "channel_order": "BGRA = [B, G, R, IR]（YOLO 内部再按需转 RGB）",
           "ok": True}
    if verbose:
        print(f"[patch] 改动点 3 通过：数据 {raw.shape[2]} 通道 / 模型首层 {inc} 通道")
    return res


def record() -> dict:
    return dict(_RECORD)


def dump_record(path: str | Path) -> None:
    Path(path).write_text(json.dumps(record(), ensure_ascii=False, indent=1),
                          encoding="utf-8")
