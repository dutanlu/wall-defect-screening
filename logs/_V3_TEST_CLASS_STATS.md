# V3 数据集逐类实例数 / 图片数（直接点数标签文件 · 只读）

来源：`01_data/dataset/labels/<split>/*.txt`（YOLO 格式，逐行一个实例）

> 说明：这两列是**数据集属性，与权重无关** —— 一旦数据集定稿即定稿，
> 不随重训变化。故先在这里点清备料，待 V3 重训出 AP 后一并回填报告 §7.1。

## train（标签文件 1995 / 图片文件 1995 / 空标签 0 / 解析失败 0）

| 类别 | 实例数 | 图片数 |
|---|---:|---:|
| `crack` | **299** | 143 |
| `spalling` | **207** | 110 |
| `efflorescence` | **841** | 270 |
| `exposed_rebar` | **973** | 324 |
| `rust` | **9433** | 837 |
| `delamination` | **486** | 362 |
| `moss` | **392** | 75 |
| **合计** | **12631** | 2121（逐类图数之和，含跨类重复计数） |

## val（标签文件 568 / 图片文件 568 / 空标签 0 / 解析失败 0）

| 类别 | 实例数 | 图片数 |
|---|---:|---:|
| `crack` | **112** | 41 |
| `spalling` | **63** | 32 |
| `efflorescence` | **195** | 72 |
| `exposed_rebar` | **269** | 92 |
| `rust` | **3103** | 239 |
| `delamination` | **140** | 104 |
| `moss` | **103** | 25 |
| **合计** | **3985** | 605（逐类图数之和，含跨类重复计数） |

## test（标签文件 285 / 图片文件 285 / 空标签 0 / 解析失败 0）

| 类别 | 实例数 | 图片数 |
|---|---:|---:|
| `crack` | **58** | 22 |
| `spalling` | **43** | 17 |
| `efflorescence` | **116** | 36 |
| `exposed_rebar` | **147** | 46 |
| `rust` | **1149** | 120 |
| `delamination` | **68** | 52 |
| `moss` | **65** | 10 |
| **合计** | **1646** | 303（逐类图数之和，含跨类重复计数） |

---

## 与 `04_results/eval/split_report.json` 的交叉核对

```json
{
  "input": "D:\\pythonstudy 备份\\创新题\\外墙缺陷筛查\\01_data\\_v3_build",
  "output": "D:\\pythonstudy 备份\\创新题\\外墙缺陷筛查\\01_data\\dataset",
  "ratio": {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
  },
  "seed": 42,
  "n_pairs": 2848,
  "n_orphan_images": 0,
  "orphan_examples": [],
  "duplicates_removed_in_src": [],
  "n_duplicates_removed": 0,
  "splits": {
    "train": {
      "n_images": 1995,
      "n_labels": 1995,
      "n_instances": 12631,
      "class_counts": {
        "crack": 299,
        "spalling": 207,
        "efflorescence": 841,
        "exposed_rebar": 973,
        "rust": 9433,
        "delamination": 486,
        "moss": 392
      }
    },
    "val": {
      "n_images": 568,
      "n_labels": 568,
      "n_instances": 3985,
      "class_counts": {
        "crack": 112,
        "spalling": 63,
        "efflorescence": 195,
        "exposed_rebar": 269,
        "rust": 3103,
        "delamination": 140,
        "moss": 103
      }
    },
    "test": {
      "n_images": 285,
      "n_labels": 285,
      "n_instances": 1646,
      "class_counts": {
        "crack": 58,
        "spalling": 43,
        "efflorescence": 116,
        "exposed_rebar": 147,
        "rust": 1149,
        "delamination": 68,
        "moss": 65
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

