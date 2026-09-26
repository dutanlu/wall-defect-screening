# 训练期可并行性实测（2026-09-25 22:42）

## 结论一句话

**GPU 类活（训练/量化/eval）在训练期间不可并行**；
**纯 CPU 类活（OpenCV/numpy，不加载 torch）可以并行**，且成本小到可以忽略。

实测证据见下。

## ① 不可并行的边界（反复实测）

| 时刻 | 系统可用内存 | load | 说明 |
|---|---|---|---|
| 训练进行中 | 0.27–0.83 GB | 94–98% | 一个 640 训练即吃满 |
| 训练进行中 GPU | 5.6 GB / 12.2 GB，util 90–98% | — | 显存有余但内存决定 |

⇒ 第二个训练必然 OOM；`--workers=4` 的 dataloader 会各自缓存解码图（见 MEMORY §1.7）。
**判据看 `ullAvailPhys`，不看显存。**

## ② 可并行的类（实测通过）

**拼接链路 `stitch_facade.py`（→ `rectify.stitch_segments` + `gsd`）**

- 依赖：**仅 `cv2` + `numpy`**，不 import torch / ultralytics
  （`grep -l "import torch" 02_code/*.py` 不含 rectify/measure/grade/stitch_facade/gsd）
- 实测（训练 69 轮进行中，`avail=0.79 GB / load=94%`）：

| 指标 | 实测值 |
|---|---|
| 耗时 | **1.16 s**（3 段 1200x900） |
| 本进程峰值内存 | **412.7 MB** |
| 系统可用内存变化 | **−24 MB** |
| 结果 | ok=True，n_used=3/3，conf=high，内点 368/334 |
| 画布 | 2901x901（有效覆盖 100.0%） |
| 标定 | 4.1678 mm/px（brick_period）⇒ 覆盖 12.09 m |
| 能力评估 | 2/8 类可判读 |

⇒ **可在训练期间并行，无风险。** 复现脚本：`logs/_test_parallel_stitch.py`
（造图 `logs/_mk_stitch_test.py`，产物 `logs/_stitch_parallel_test/parallel_test.json`）

## ③ 本次踩到的两个坑（都是「静默失败」，比崩溃更坏）

### 坑 1：`cv2.imwrite` 对中文路径**静默返回 False**
本项目路径含中文（`D:\pythonstudy 备份\...`）。OpenCV 的 Windows 窄字符 API 对非 ASCII
路径**不抛异常、只返回 False** ⇒ 图根本没落盘，而脚本照常打印"已写出"。
**实测**：中文路径 `imwrite -> False, exists=False`；英文路径 `-> True`（cv2 5.0.0）。
⇒ **一律用 `common.imwrite_u` / `imread_u`。** 本脚本第一版就踩了，全盘找不到 `seg_*.png`。

### 坑 2：`stitch_segments` 收的是**图像数组列表**，不是路径列表
传 `list[str]` 会在 `ctx.cvtColor` 处 `AttributeError: 'str' object has no attribute 'ndim'`。
⇒ 调用方自己 `imread_u` 读成数组再传；`stitch_facade.stitch_and_assess` 同样收数组。

## ④ 因此可安全并行的候选清单

| 候选 | 是否可并行 | 说明 |
|---|---|---|
| 拼接链路（含正射）/ 尺度标定 / 分级判据 | ✅ 实测通过 | 纯 cv2+numpy |
| 交付包一致性核查 `_pack_vs_src.py` | ✅ | 纯文件 IO |
| 文档/Markdown 改写与行尾校验 | ✅ | 纯 IO |
| eval / 量化 / TTA / 训练 | ❌ | 需 GPU 或吃内存 |
