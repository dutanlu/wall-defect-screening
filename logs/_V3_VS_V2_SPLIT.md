# V3 划分是否继承 V2 test？（MD5 + 去前缀双路比对 · 只读）

> 口径说明：V3 的图片命名规则是 `<源>__<原 stem>.jpg`（例 `greybrick-yolo__1174.jpg`），
> 归档里的 V2 图仍是原名（`1174.jpg`）⇒ 比对时需**去掉 `__` 前缀**，
> 并同时用**内容 MD5** 交叉确认，避免只靠命名。

## 一、目录实况（MD5 去重后）

| 集合 | 路径 | 图数 | MD5 重复 |
|---|---|---:|---:|
| `v2_test` | `\_archive_20260920\dataset_v2_20260921\images\test` | 67 | 0 |
| `v1_test` | `\_archive_20260920\dataset_v1_20260920\dataset\images\test` | 65 | 0 |
| `v3_train` | `\01_data\dataset\images\train` | 1995 | 0 |
| `v3_val` | `\01_data\dataset\images\val` | 568 | 0 |
| `v3_test` | `\01_data\dataset\images\test` | 285 | 0 |

## V2 test 的 67 张图在 V3 中的去向

| 去向 | MD5 命中 | 去前缀名命中 | 占 V2 test |
|---|---:|---:|---:|
| v3_test | **23** | 23 | 34.3% |
| v3_val | **10** | 10 | 14.9% |
| v3_train | **34** | 34 | 50.7% |
| **不在 V3 任何 split** | **0** | | 0.0% |

### 判定

> ❌ **未继承**：V2 test 的 67 张图里，只有 23 张留在 V3 test，44 张被重分到 val/train，0 张已不在 V3。
> ⇒ V3 是一次**完全重划分**（对 2848 图整体做 7:2:1），旧图的归属并未保留。

## 二、V3 test（285 张）的来源构成

- `urban_yolo_rust_delam` : 172
- `greybrick-yolo` : 65
- `hrcds_yolo_exposed_rebar` : 46
- `rebar_structural` : 2

- 其中来自 **V2 test** 的：**23** 张（8.1% of V3 test）

## 三、同一批图的标签在 V2 → V3 之间有没有变？

在 V3 test 中匹配到 **23** 张 V2 test 旧图；其标签逐类对比如下：

| 类别 | V2 test 标签 | 同图 V3 标签 | 差 |
|---|---:|---:|---:|
| `crack` | 55 | 55 | +0 |
| `spalling` | 11 | 11 | +0 |
| `efflorescence` | 12 | 12 | +0 |
| `exposed_rebar` | 0 | 0 | +0 |
| `rust` | 0 | 0 | +0 |
| `delamination` | 0 | 0 | +0 |
| `moss` | 7 | 7 | +0 |
| **合计** | **85** | **85** | **+0** |


## 四、V3 test 逐类实例数 / 图片数（备料，供报告 §7.1 回填）

| 类别 | 实例数 | 图片数 |
|---|---:|---:|
| `crack` | **58** | 22 |
| `spalling` | **43** | 17 |
| `efflorescence` | **116** | 36 |
| `exposed_rebar` | **147** | 46 |
| `rust` | **1149** | 120 |
| `delamination` | **68** | 52 |
| `moss` | **65** | 10 |
| **合计** | **1646** | 303 图（285 张唯一图） |

