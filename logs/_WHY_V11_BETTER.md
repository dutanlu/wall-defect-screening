# 核查：`v11s640` 比 `v8s640` 高 0.086 —— 是真实差异还是口径误解？

> 由 `logs/_why_v11_better.py` 自动生成，**只做核查，不下武断结论**。

---

## A. 评估口径核查：四个模型是否用同一 `imgsz` / `split` / `data`

⚠️ **重要发现（2026-09-20 20:15）**：实验链的 `eval` 步骤以 `--weights=<单个>` 方式运行，
**覆盖**了 `--all` 产出的汇总件 `test_eval_summary.json`，现在该文件**只剩 `v8s640_final` 一行**。
因此本核查**改从四个单模型件 `*_test_metrics.json` 取数**（这些文件未被覆盖，数据完好）。

| run | 使用的单模型件 | weights | weights_mb | split | imgsz |
|---|---|---|---|---|---|
| `v8s640` | `v8s640_test_metrics.json` | v8s640_best.pt | 21.51 | test | **640** |
| `v11s640` | `v11s640_test_metrics.json` | v11s640_best.pt | 18.33 | test | **640** |
| `v8n640` | `v8n640_test_metrics.json` | v8n640_best.pt | 5.99 | test | **640** |
| `v8s1024` | `v8s1024_test_metrics.json` | v8s1024_best.pt | 21.58 | test | **640** |

- 四个 run 的 `imgsz` 取值集合 = **{640}**（一致：True）
- 四个 run 的 `split` 取值集合 = **{'test'}**（一致：True）

⇒ **四个模型在同一 `imgsz` 下评估，横向可比。**
  ⚠️ 但 `v8s1024` **训练在 1024**，评估却用 **640** ⇒
     该行数字**被低估**，属**评估口径问题**（须在报告中说明或重评），
     但它**不影响** v11s 与 v8s 的对比（两者训练/评估都是 640）。

**参考**：被覆盖前的汇总件内容已由 `logs/_v2_metrics_snapshot.json` 保留（20:09 抓取）。

## B. 数据口径核查：train/val/test 是否唯一且未被改动

```json
{
  "input": "D:\\pythonstudy 备份\\创新题\\外墙缺陷筛查\\01_data\\unified",
  "output": "D:\\pythonstudy 备份\\创新题\\外墙缺陷筛查\\01_data\\dataset",
  "ratio": {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
  },
  "seed": 42,
  "n_pairs": 674,
  "n_orphan_images": 0,
  "orphan_examples": [],
  "duplicates_removed_in_src": [],
  "n_duplicates_removed": 0,
  "splits": {
    "train": {
      "n_images": 473,
      "n_labels": 473,
      "n_instances": 1734,
      "class_counts": {
        "crack": 298,
        "spalling": 224,
        "efflorescence": 817,
        "exposed_rebar": 0,
        "rust": 0,
        "delamination": 0,
        "moss": 395
      }
    },
    "val": {
      "n_images": 134,
      "n_labels": 134,
      "n_instances": 501,
      "class_counts": {
        "crack": 113,
        "spalling": 59,
        "efflorescence": 217,
        "exposed_rebar": 0,
        "rust": 0,
        "delamination": 0,
        "moss": 112
      }
    },
    "test": {
      "n_images": 67,
      "n_labels": 67,
      "n_instances": 259,
      "class_counts": {
        "crack": 58,
        "spalling": 30,
        "efflorescence": 118,
        "exposed_rebar": 0,
        "rust": 0,
        "delamination": 0,
        "moss": 53
      }
    }
  },
  "intersection_check": {
    "train-val:names": 0,
    "train-val:md5": 0,
    "train-val:dhash": 0,
    "train-test:names": 0,
    "train-test:md5": 0,
    "train-test:dhash": 0,
    "val-test:names": 0,
    "val-test:md5": 0,
    "val-test:dhash": 0
  },
  "leak_free": true,
  "dataset_yaml": "D:\\pythonstudy 备份\\创新题\\外墙缺陷筛查\\01_data\\dataset\\wall_defects.yaml",
  "roboflow_yaml": "D:\\pythonstudy 备份\\创新题\\外墙缺陷筛查\\02_code\\roboflow_yaml\\wall_defects.yaml"
}
```

**各 split 的标签文件数（实测）**：
- `train`：images = **473**，labels = **473**
- `val`：images = **134**，labels = **134**
- `test`：images = **67**，labels = **67**

## C. 权重来源核查：`03_weights/*_best.pt` 是否就是对应 run 的产物

| run | 03_weights 副本 | 源文件 (train/<run>/weights/best.pt) | 大小一致 | md5(前4MB) 一致 |
|---|---|---|---|---|
| `v8s640` | 21.51 MB | 21.51 MB | True | True |
| `v11s640` | 18.33 MB | 18.33 MB | True | True |
| `v8n640` | 5.99 MB | 5.99 MB | True | True |
| `v8s1024` | 21.58 MB | 21.58 MB | True | True |

⇒ 若「大小一致 + md5 一致」，则**不存在权重错配**（不会把 v11s 的权重当成 v8s 评）。

## D. 早停差异核查：是否「训练更久」造成的不公平

| run | 实际 epoch | 结束方式 | best epoch |
|---|---|---|---|
| `v8s640` | **100** | 跑满 100 | ep74 |
| `v11s640` | **100** | 跑满 100 | ep62 |
| `v8n640` | **85** | 早停 | ep60 |
| `v8s1024` | **94** | 早停 | ep76 |

⇒ `v11s640` 与 `v8s640` **都是 100 轮**（未早停）⇒ **两者训练预算相同**，
   「v11s 训练更久所以更好」这个解释**不成立**。

## E. 超参差异核查（来自 `train.py` 的 RECIPES）

| run | 预训练权重 | imgsz | batch | 与 v8s640 的差异 |
|---|---|---|---|---|
| `v8s640` | `yolov8s.pt` | 640 | 24 | —（基准） |
| `v11s640` | `yolo11s.pt` | 640 | 24 | **仅预训练权重不同** |
| `v8n640` | `yolov8n.pt` | 640 | 24 | 权重（n 规格） |
| `v8s1024` | `yolov8s.pt` | **1024** | **12** | 分辨率 + batch |

⇒ **`v11s640` vs `v8s640` 是完全受控的对照**：
   同数据 / 同 imgsz / 同 batch / 同 epochs / 同 patience / 同增强，
   **唯一变量是预训练权重（yolo11s 架构 vs yolov8s 架构）**。
   这正是答辩必答题「为何选该 YOLO 版本」要的证据 —— 设计上就是干净对照。

## F. 逐类证据：优势是「全局偏移」还是「集中在特定类」

| 类别 | 实例 | v11s AP50 | v8s AP50 | 差值 | 半宽(±) | 可判读 |
|---|---|---|---|---|---|---|
| crack | 58 | 0.7201 | 0.7894 | -0.0693 | ±0.11 | ⚠️ 不可判读 |
| spalling | 30 | 0.7319 | 0.5163 | +0.2156 | ±0.16 | ✅ 超阈值 |
| efflorescence | 118 | 0.7926 | 0.7154 | +0.0772 | ±0.08 | ⚠️ 不可判读 |
| moss | 53 | 0.8068 | 0.6853 | +0.1215 | ±0.12 | ✅ 超阈值 |

**整体**：
- `v8s640`：mAP50 = **0.6766**，mAP50-95 = 0.3945
- `v11s640`：mAP50 = **0.7628**，mAP50-95 = 0.4407


## G. 泄漏检查：test 是否与 train 有内容交集

- train 图 473 张，test 图 67 张，**内容交集 = 0**
  ⇒ **无泄漏**（md5 前 4MB 比对）

---

## 结论（待人工确认）

见对话正文。**关键判定**：
1. 口径是否一致；2. 是否受控对照；3. 优势是否超判读阈值；4. 是否有泄漏。
