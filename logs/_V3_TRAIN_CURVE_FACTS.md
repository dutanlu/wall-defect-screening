# V3 训练曲线事实（只读提取，备料用）

生成时间：2026-09-21 14:50:54

来源：`04_results/train/<run>/results.csv`（ultralytics 逐 epoch 记录）

> 本文件**只是备料**，不构成报告改动。报告里 §5.2 / §5.3 的 V3 数字是否回填，待确认。

---

## `v8s640`

- 文件：`D:\pythonstudy 备份\创新题\外墙缺陷筛查\04_results\train\v8s640\results.csv`
- mtime：2026-09-21 11:40:05 ｜ 相对训练日志创建：**本次新训(V3)**
- 大小：12820 B

- 列：`epoch`, `time`, `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `val/box_loss`, `val/cls_loss`, `val/dfl_loss`, `lr/pg0`, `lr/pg1`, `lr/pg2`
- epoch 数：**100**

### 关键值

| 项 | 值 |
|---|---|
| 有效 epoch 数 | 100 |
| **best epoch（按 val mAP50）** | **ep 88** |
| best val mAP50 | **0.6763** |
| best val mAP50-95 | 0.4057 |
| best 时 P / R | 0.7386 / 0.6255 |
| 末 epoch | ep 100 |
| 末 val mAP50 / mAP50-95 | 0.6683 / 0.4001 |
| 末 val P / R | 0.7742 / 0.6004 |
| 是否跑满 100 epoch | 是（未早停） |

### 末 30 个 epoch 的 val mAP50 波动

- 区间：**0.6570 ~ 0.6763**（跨度 0.0193）
- 均值：0.6691

- 训练累计耗时（CSV `time` 列末值）：**129.2 分钟**（约 2.15 小时）
- 平均每 epoch：77.5 s

---

## `v11s640`

- 文件：`D:\pythonstudy 备份\创新题\外墙缺陷筛查\04_results\train\v11s640\results.csv`
- mtime：2026-09-21 14:27:46 ｜ 相对训练日志创建：**本次新训(V3)**
- 大小：12823 B

- 列：`epoch`, `time`, `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `val/box_loss`, `val/cls_loss`, `val/dfl_loss`, `lr/pg0`, `lr/pg1`, `lr/pg2`
- epoch 数：**100**

### 关键值

| 项 | 值 |
|---|---|
| 有效 epoch 数 | 100 |
| **best epoch（按 val mAP50）** | **ep 90** |
| best val mAP50 | **0.6862** |
| best val mAP50-95 | 0.4173 |
| best 时 P / R | 0.7146 / 0.6501 |
| 末 epoch | ep 100 |
| 末 val mAP50 / mAP50-95 | 0.6779 / 0.4126 |
| 末 val P / R | 0.7779 / 0.6082 |
| 是否跑满 100 epoch | 是（未早停） |

### 末 30 个 epoch 的 val mAP50 波动

- 区间：**0.6473 ~ 0.6862**（跨度 0.0389）
- 均值：0.6760

- 训练累计耗时（CSV `time` 列末值）：**167.5 分钟**（约 2.79 小时）
- 平均每 epoch：100.5 s

---

## `v8n640`

- 文件：`D:\pythonstudy 备份\创新题\外墙缺陷筛查\04_results\train\v8n640\results.csv`
- mtime：2026-09-21 14:50:47 ｜ 相对训练日志创建：**本次新训(V3)**
- 大小：3458 B

- 列：`epoch`, `time`, `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `val/box_loss`, `val/cls_loss`, `val/dfl_loss`, `lr/pg0`, `lr/pg1`, `lr/pg2`
- epoch 数：**26**

### 关键值

| 项 | 值 |
|---|---|
| 有效 epoch 数 | 26 |
| **best epoch（按 val mAP50）** | **ep 22** |
| best val mAP50 | **0.5333** |
| best val mAP50-95 | 0.2904 |
| best 时 P / R | 0.6310 / 0.4888 |
| 末 epoch | ep 26 |
| 末 val mAP50 / mAP50-95 | 0.5307 / 0.2822 |
| 末 val P / R | 0.5718 / 0.5180 |
| 是否跑满 100 epoch | 否，止于 ep 26 |

### 末 26 个 epoch 的 val mAP50 波动

- 区间：**0.1291 ~ 0.5333**（跨度 0.4042）
- 均值：0.3969

- 训练累计耗时（CSV `time` 列末值）：**22.5 分钟**（约 0.37 小时）
- 平均每 epoch：51.9 s

---

## `v8s1024`

- 文件：`D:\pythonstudy 备份\创新题\外墙缺陷筛查\04_results\train\v8s1024\results.csv`
- mtime：2026-09-20 20:07:14 ｜ 相对训练日志创建：**陈旧(可能为 V2)**
- 大小：12042 B

- 列：`epoch`, `time`, `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `val/box_loss`, `val/cls_loss`, `val/dfl_loss`, `lr/pg0`, `lr/pg1`, `lr/pg2`
- epoch 数：**94**

### 关键值

| 项 | 值 |
|---|---|
| 有效 epoch 数 | 94 |
| **best epoch（按 val mAP50）** | **ep 76** |
| best val mAP50 | **0.6767** |
| best val mAP50-95 | 0.3765 |
| best 时 P / R | 0.7248 / 0.6321 |
| 末 epoch | ep 94 |
| 末 val mAP50 / mAP50-95 | 0.6715 / 0.3769 |
| 末 val P / R | 0.7030 / 0.6358 |
| 是否跑满 100 epoch | 否，止于 ep 94 |

### 末 30 个 epoch 的 val mAP50 波动

- 区间：**0.6473 ~ 0.6767**（跨度 0.0294）
- 均值：0.6627

- 训练累计耗时（CSV `time` 列末值）：**68.0 分钟**（约 1.13 小时）
- 平均每 epoch：43.4 s

---

## 判读要点

- 本轮 `_TRAIN_V3.md` 是一份**串联日志**，含 3 个训练轮次，
  起点分别在日志第 **66 / 10349 / 20636** 行 —— 前两轮 `v8s640`、`v11s640` 已完成，
  第三轮 `v8n640` 进行中，`v8s1024` 尚未开始。**不是重启，是排队。**
- V3 的 val 集为 **568 图 / 3985 实例**（7 类），与 V2 的 134 图不是一回事；
  因此 V3 的 val mAP50 与 V2 的 0.76xx **不可直接比较** —— V3 多出 rust 等 3 个难类，
  且 rust 独占 train 实例的 **74.7%**（9433 / 12631）。
- §7.1.1 已声明 **val 数字不可作上报口径**，本文件仅用于描述训练过程（学习曲线）。