# 数据与照片补齐情况核查

生成时间：2026-09-21 08:49:13
性质：**只读盘点，未修改任何内容**

---

## 一、公开数据集目录 `01_data/raw/_public_datasets/`

| 目录 | 占用 |
|---|---:|
| `bfdd` | 527.7 MB |
| `construction_defect_yolo` | 851.0 MB |
| `hrcds` | 195.4 MB |
| `hrcds_yolo_exposed_rebar` | 78.7 MB |
| `mbdd2025` | 2414.2 MB |
| `rc1841` | 143.1 MB |
| `rc2119` | 152.8 MB |
| `roi1555` | 46.9 MB |
| `urban_infra` | 1108.7 MB |

## 二、★ 三个空缺类的补齐现状

### 2.1 `exposed_rebar`（HRCDS 转换产物）

- `images/` → **464 个文件**  78.7 MB
- `labels/` → **464 个文件**  0.1 MB
- 标签文件 464 个，空标签 0 个
- 类别索引分布：{'2': 1392}（总框数 1392）

### 2.2 `rust` / `delamination`（Urban Infra）

- 归档目录：`urban_infra` → 1108.7 MB
  - `urban-infrastructure-anomalies.zip`  1108.7 MB
- 已解压的 Urban 目录候选：** 无（仍为 zip，未解压）**
- 是否有 urban 的 YOLO 转换产物（rust / delamination）：**无**

### 2.3 其他曾被列为空缺的类

- `void`：**无文件名命中**
- `rust`：**无文件名命中**
- `delamination`：**无文件名命中**
- `exposed_rebar`：**无文件名命中**
- `spalling`：**无文件名命中**

## 三、照片收集站 `08_photo_collector/`

| 条目 | 类型 | 大小 |
|---|---|---:|
| `_backup_before_private` | 目录 | 0.1 MB |
| `_patch_promo_storage.txt` | 文件 | 0.0 MB |
| `_patch_promo_storage2.txt` | 文件 | 0.0 MB |
| `_patch_site_copy.txt` | 文件 | 0.0 MB |
| `_patch_storage_private.txt` | 文件 | 0.0 MB |
| `site` | 目录 | 0.1 MB |
| `发布话术.md` | 文件 | 0.0 MB |
| `推广文案.md` | 文件 | 0.0 MB |

### 投稿内容统计


- **投稿图片合计 = 0 张**

### 投稿登记文件

- `_patch_promo_storage.txt`  0.0 MB
- `_patch_promo_storage2.txt`  0.0 MB
- `_patch_site_copy.txt`  0.0 MB
- `_patch_storage_private.txt`  0.0 MB
- `发布话术.md`  0.0 MB
- `推广文案.md`  0.0 MB
- `_backup_before_private\发布话术.md`  0.0 MB
- `_backup_before_private\推广文案.md`  0.0 MB

## 四、已构建数据集 `01_data/dataset/`

- `images`  95.3 MB
- `labels`  0.3 MB
- `wall_defects.yaml`  0.0 MB

### `01_data\dataset\wall_defects.yaml`

    # 外墙缺陷筛查数据集配置（由 common.write_dataset_yaml 生成）
    # 注意：path 为绝对路径，避免 Ultralytics 在中文/非标准目录下解析失败
    path: D:/pythonstudy 备份/创新题/外墙缺陷筛查/01_data/dataset
    train: images/train
    val: images/val
    test: images/test
    
    names:
      0: crack
      1: spalling
      2: efflorescence
      3: exposed_rebar
      4: rust
      5: delamination
      6: moss

### `01_data\dataset\wall_defects.yaml`

    # 外墙缺陷筛查数据集配置（由 common.write_dataset_yaml 生成）
    # 注意：path 为绝对路径，避免 Ultralytics 在中文/非标准目录下解析失败
    path: D:/pythonstudy 备份/创新题/外墙缺陷筛查/01_data/dataset
    train: images/train
    val: images/val
    test: images/test
    
    names:
      0: crack
      1: spalling
      2: efflorescence
      3: exposed_rebar
      4: rust
      5: delamination
      6: moss
