# 端到端分层耗时 · 完整口径（阶段 0，2026-09-26）

## 一、为什么必须重做这一版

原 `bench_pipeline.py` 的分层只到 `grade` 为止，**漏算了两个真实入口必跑的阶段**：
- `gsd.assess_image_quality(img)` —— 对**全分辨率原图**做成像自检（灰度化 + float32 Laplacian + 中值 + **两次全图 `np.median`**），纯 CPU；
- `grade.classify_unjudgeable` + `grade.apply_abstention` —— 弃权链（含 `building_risk_level`）。

而真实入口 `02_code/pipeline.py:239-259` 两段都要跑。⇒ **此前所有「端到端」数字系统性偏低**，
报告 §7.5 又正引用这些数字。

**本次改动**（`02_code/bench_pipeline.py`，纯 CRLF，+1772 B）：
新增 `quality_gate` 与 `abstain` 两段计时，严格照 `pipeline.py` 的顺序与参数对齐
（含 `apply_abstention` 的 `advise_fn`）。旧版备份：
`logs/_opt_baseline_20260926/02_code__bench_pipeline.py.before_stage0`。

**Gate 0 通过**：两档的「各阶段之和 / 独立墙钟」均为 **1.03**（判据 0.85~1.15）。

---

## 二、完整口径分层表

### 2.1 11.8MP 真机照（`sc_1a7e86db-9.png`，3968×2976，检出 1）

| 阶段 | 旧口径 ms | **完整口径 ms** | 占比 |
|---|---:|---:|---:|
| load | 130.54 | 113.70 | 16.5% |
| rectify | 0.00（默认关） | 0.00 | 0.0% |
| predict_ultralytics | 35.32 | 13.08 | 1.9% |
| calibrate / interpret / grade | 0.07 | 0.11 | ≈0% |
| **measure** | 5024.51 | **267.22** | 38.7% |
| **quality_gate** | **（未计）** | **252.01** | **36.5%** |
| abstain | （未计） | 0.09 | 0.0% |
| draw / save | 36.81 | 44.63 | 6.5% |
| **各阶段之和** | 5227.25 | **690.82** | 100% |
| 独立墙钟 p50 | — | **668.74** | — |

> ⚠️ 旧口径那一列的 `measure` 是**优化前**、其余阶段是**另一批次**的测量，
> 不可混用（详见第四节）。本表仅用于说明「漏了哪些阶段、量级多少」。

### 2.2 448² 小图（`04_results/vis`，n=16，像素量中位 0.20 MP）

| 阶段 | 完整口径 ms | 占比 |
|---|---:|---:|
| load | 1.04 | 5.8% |
| predict_ultralytics | **9.46** | **52.5%** |
| calibrate / interpret / grade | 0.06 | 0.4% |
| measure | 2.83 | 15.7% |
| **quality_gate** | **2.88** | **16.0%** |
| abstain | 0.05 | 0.3% |
| draw / save | 1.69 | 9.4% |
| 各阶段之和 | **18.00** | 100% |
| 独立墙钟 p50 | **17.53** | — |

**两档结论**：
- **小图等 GPU**（predict 占 52.5%）；
- **大图等 CPU**，且优化前后瓶颈已换人：`measure` 38.7% + **`quality_gate` 36.5%**。

---

## 三、★ 本次口径修正的最大产出：**发现一个新的 #2 瓶颈**

`quality_gate` 在 11.8MP 上 **252.01 ms = 36.5%**，几乎与 `measure`（267 ms）等量。

**它为什么贵**（`02_code/gsd.py:429-445`）：对全图依次做
`cvtColor` → `astype(float32)`（11.8MP 下 140 MB）→ float32 `Laplacian`（再 140 MB）
→ `medianBlur(3)` → `astype(float32)` → 相减（再一个 140 MB）→
**两次 `np.median`（各一次完整遍历）** → `mean()` / `std()`。

**耗时随像素量近线性**（`logs/_bench_extra_stages.txt`）：

| 分辨率 | 像素数 | 耗时 ms |
|---|---:|---:|
| 2976×3968 | 11,808,768 | **231.54** |
| 1488×1984 | 2,952,192 | 52.49 |
| 744×992 | 738,048 | 13.61 |

⇒ 这是阶段 1 之后**最值得继续优化的对象**（详见第五节）。

---

## 四、补测项（此前完全没有实测）

| 项 | 实测 | 说明 |
|---|---|---|
| `rectify` 开启时（11.8MP） | **111.83 ms（14.1%）** | 默认关闭；开启后总耗时 668.74 → 796.30 ms |
| `_skeleton_length_px`（纯 Python set 循环） | 大掩膜 **17~20 ms**，占 measure **6~14%** | 此前从未被单独计时，属隐性热点 |
| `abstain` 弃权链 | 0.05~0.09 ms | 纯 O(检出数) 规则运算，**不是热点** |
| `assess_image_quality` 随像素量 | 见第三节表 | 近线性 |

---

## 五、口径声明（写进报告必须原样带上）

1. **本表每一层单独列出，不可合成一个含糊的「总耗时」对外报。**
2. 纯模型推理（ultralytics 口径）与「端到端管线」量的不是同一件事。
3. 耗时几乎由像素量决定，**必须同时给出分辨率**；只报一个数无意义。
4. 已预热后测量，第 1 圈不计入；未预热首图会虚高数倍。
5. GPU 计时已用 `torch.cuda.synchronize()` 包夹；CPU 口径不适用此表。
6. **跨运行对比只认受控的那一段**：实测 `predict_ultralytics` 在两次相邻运行间
   从 **29.54 波动到 10.91 ms**（同一图、同一权重），属环境波动
   ⇒ **不要把 predict 的波动算成优化收益**。
7. ⚠️ **未实测项（如实列出）**：`06_deploy/app.py` 的 Gradio 端到端（请求→返回）、
   视频/批量入口的整段耗时（目前只能由「单图耗时 × N」估算，未直接测）。
   这两项不要凭估算写成实测。

---

## 六、留痕

| 文件 | 内容 |
|---|---|
| `04_results/eval/pipeline_bench_11_8MP_fullcaliber.json` | 11.8MP 完整口径（本版） |
| `04_results/eval/pipeline_bench_448_fullcaliber.json` | 448² 完整口径（本版） |
| `04_results/eval/pipeline_bench_11_8MP_rectify.json` | 开启 rectify 的完整口径 |
| `04_results/eval/pipeline_bench_11_8MP_after.json` | 阶段 1 优化后（旧口径，供对照） |
| `04_results/eval/pipeline_bench_11_8MP_baseline_rerun.json` | 阶段 1 同条件改前（旧口径） |
| `logs/_bench_extra_stages.txt` | `_skeleton_length_px` 与质量门的补测 |
| `logs/_bench_pipeline.txt` | bench 脚本自身输出的最新日志 |
