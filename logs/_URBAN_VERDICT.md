# Urban Infrastructure Anomalies 判读报告

> 生成时间：2026-09-20 深夜
> 数据来源：Kaggle `sanjuchinni/urban-infrastructure-anomalies`
> 归档文件：`01_data/raw/_public_datasets/urban_infra/urban-infrastructure-anomalies.zip`
> **归档大小：1,162,572,641 B（实测 = GCS 真实对象大小，精确一致）**
> 校验：`zipfile` 可打开，条目 34,632，`testzip = None`（零损坏）

---

## 一、★★★ 首要结论：Kaggle 报的 totalBytes 是错的

| 项 | 值 |
|---|---|
| Kaggle `view` API 报的 `totalBytes` | `1,174,716,064` |
| **GCS 对象真实大小** | **`1,162,572,641`** |
| 差值 | **`12,143,423` B（Kaggle 多报 12.14 MB）** |

**决定性证据**（`_gcs_quota_test.py`，用全新签名 URL 探测）：

```
offset 1,162,572,641 → Range 请求返回 HTTP 416（Range Not Satisfiable）❌
offset 1,174,710,000 → Range 请求返回 HTTP 416 ❌
seg7 起点 1,027,876,556 → 206 完整 ✅
seg7 中段 1,076,823,058 → 206 完整 ✅
seg7 后段 1,125,769,560 → 206 完整 ✅
```

⇒ 按 `1,174,716,064` 切分 8 段时，**最后一段的上界越界**，
所以 seg7 反复重下（try#5+）却**永远无法「达标」**。
这与 mbdd2025 是**同一类问题**（见 `_mbdd_real_size.txt`，那边差 29.76 MB）。

**教训（已写入技能）**：Kaggle 的 `totalBytes` **必须实测**，
不能直接拿来切分段边界；正解是用 Range 二分定位真实大小。

---

## 二、目录结构（实测）

```
Urban_Infrastructure_Anamolies/          ← 注意：原数据集名拼写有误 Anamolies
├── data.yaml
├── train/images      12,877 张
├── train/labels      13,354 个
├── train/labels.cache
├── valid/images       2,772 张
├── valid/labels       2,975 个
├── valid/labels.cache
├── test/images        1,274 张
├── test/labels        1,376 个
└── yolov8s.pt         ← 附赠官方基线权重
```

---

## 三、官方类别定义（`data.yaml` 实证）

```yaml
train: train/images
val: valid/images
nc: 13
names: ['corrosion', 'delamination', 'cover_detachment', 'efflorescence', 'crack',
        'rust', 'scaling', 'spalling', 'void', 'tile_crack', 'tile_delamination',
        'tile_loss', 'peeling']
```

---

## 四、★★★ 全量类别分布（实测，非抽样）

### train（标签 13,354 个）

| # | 类别 | 实例数 | 图数 |
|---|---|---|---|
| 0 | corrosion | 2,859 | 837 |
| 1 | **delamination** | **694** | **518** |
| 2 | cover_detachment | 5,322 | 2,252 |
| 3 | efflorescence | 2,807 | 1,238 |
| 4 | crack | 10,001 | 5,022 |
| 5 | **rust** | **52,280** | **4,616** |
| 6 | scaling | 167 | 106 |
| 7 | spalling | 5,800 | 2,362 |
| 8 | **void** | **1,240** | **516** |
| 9 | tile_crack | 631 | 323 |
| 10 | tile_delamination | 305 | 230 |
| 11 | tile_loss | 837 | 424 |
| 12 | peeling | 113 | 95 |
| | **TOTAL** | **83,056** | |

### valid（标签 2,842 个）

| # | 类别 | 实例数 | 图数 |
|---|---|---|---|
| 0 | corrosion | 834 | 253 |
| 1 | delamination | 203 | 148 |
| 2 | cover_detachment | 1,482 | 654 |
| 3 | efflorescence | 796 | 364 |
| 4 | crack | 2,365 | 986 |
| 5 | rust | 9,661 | 945 |
| 6 | scaling | 60 | 45 |
| 7 | spalling | 1,593 | 683 |
| 8 | **void** | **0** | **0** |
| 9 | tile_crack | 151 | 75 |
| 10 | tile_delamination | 96 | 69 |
| 11 | tile_loss | 172 | 91 |
| 12 | peeling | 67 | 59 |
| | **TOTAL** | **17,480** | |

### test（标签 1,309 个）

| # | 类别 | 实例数 | 图数 |
|---|---|---|---|
| 0 | corrosion | 432 | 127 |
| 1 | delamination | 108 | 80 |
| 2 | cover_detachment | 725 | 308 |
| 3 | efflorescence | 398 | 186 |
| 4 | crack | 1,032 | 476 |
| 5 | rust | 3,387 | 353 |
| 6 | scaling | 12 | 8 |
| 7 | spalling | 840 | 346 |
| 8 | **void** | **0** | **0** |
| 9 | tile_crack | 93 | 44 |
| 10 | tile_delamination | 51 | 40 |
| 11 | tile_loss | 116 | 57 |
| 12 | peeling | 45 | 33 |
| | **TOTAL** | **7,239** | |

### ★ 我们关心的三类（跨 split 汇总）

| 类别 | 实例 | 图 | train 图 | valid 图 | test 图 |
|---|---|---|---|---|---|
| **rust** | 65,328 | **5,914** | 4,616 | 945 | 353 |
| **delamination** | 1,005 | **746** | 518 | 148 | 80 |
| **void** | 1,240 | **516** | 516 | **0** | **0** |

> ⚠️ **重要修正**：此前（承上段）我只从 seg0 抽取了 test 集，误判「`void` 完全没有」。
> **全量实测：`void` 在 train 有 516 图 / 1,240 实例**，只是 valid/test 为空。
> ⇒ 这是**划分不当**，不是数据缺失。

---

## 五、★★ 看图判读（四要素）

### 5.1 `delamination` ✅ 语义匹配

**判读样本**：4 张（`logs/_preview_urban_test/`）

| 样本 | 场景 | 现象 | 判读 |
|---|---|---|---|
| delamination #1 | 白色涂装外墙（多层住宅） | 墙角涂层起皮、成片脱落露底 | ✅ 饰面层与基层脱开 |
| delamination #2 | 浅色抹灰外墙 | 抹灰层空鼓翘边、边缘卷起 | ✅ 同上 |
| delamination #3 | 涂料饰面 | 大片涂层剥离 | ✅ 同上 |
| delamination #4 | 混凝土面 | 表层剥落 | ✅ 同上 |

- **场景**：建筑外墙（含混凝土面 + 涂装面）✅
- **粒度**：区域级（成片），框贴合 ✅
- **语义**：「饰面层/抹灰层与基层脱开」✅
- **格式**：YOLO txt 归一化框 ✅

### 5.2 `rust` ✅ 语义匹配（但注意内涵）

| 样本 | 场景 | 现象 | 判读 |
|---|---|---|---|
| rust #1 | 混凝土墙面 | 沿模板缝下淌的**锈渍条带** | ✅ 表面锈迹污染 |
| rust #2 | 桥梁/水工结构 | 表面返锈 | ✅ |
| rust #3 | 混凝土面 | 点状锈斑 | ✅ |
| rust #4 | 混凝土面 | 大面积淡锈色区域 | ⚠️ 粒度偏粗 |

- **★ 关键**：`rust` = **表面锈迹 / 锈水污染**，**不是钢筋外露**。
  ⇒ 与 `exposed_rebar`（HRCDS，混凝土剥落露出锈蚀钢筋）**语义不同，不能互相替代**。
- ⚠️ **瑕疵 1**：标注粒度偏粗，有些框圈住大片淡锈色区域。
- ⚠️ **瑕疵 2**：相当比例来自**桥梁/水工结构**，而非建筑外墙。

---

## 六、结论与建议

### 6.1 结论

| 类别 | 判定 | 依据 |
|---|---|---|
| **`rust`** | ✅ **可用** | 5,914 图（train 4,616），语义 = 表面锈迹 |
| **`delamination`** | ✅ **可用** | 746 图（train 518），语义 = 饰面层脱开 |
| **`void`** | ⚠️ **train 可用，valid/test 缺** | train 516 图，但划分不当 |
| 其余 10 类 | 与本项目 7 类不匹配或冗余 | — |

### 6.2 建议

1. **`rust` 与 `delamination` 建议并入训练集**（两类各取 train 即可，避免污染 test）。
2. **`void` 若要用，须自己重划分**（把 train 的 516 图按比例拆出 valid/test）。
3. ⚠️ **不建议并入**：`cover_detachment` / `peeling` / `scaling` / `tile_*`
   —— 语义与我们 7 类重叠或属瓷砖场景，强并会引入标签歧义。
4. **test 集不能直接用**：`void` 在 test 为 0，且原 test 与我们的独立测试集口径不同。

### 6.3 待用户拍板

- [ ] `rust` 是否并入？（我建议：并入）
- [ ] `delamination` 是否并入？（我建议：并入）
- [ ] `void` 是否并入？（train 有 516 图，但需自己重划分 → 建议**暂不并入**，标注成本高）
- [ ] `cover_detachment`（train 2,252 图）是否并入？（我倾向**不并入**，与 `spalling` 歧义）

---

## 七、★ 方法论收获（已写入技能）

### 7.1 从「未下完的 ZIP」提前提取内容

- ZIP 是顺序写入 ⇒ **seg0 常含前部完整条目**（yaml / 标签 / 部分图片）。
- 做法：**暴力扫 `PK\x03\x04`（local file header）+ 解析**。
- **★★ 两个必踩坑**：
  1. **不能用 `zipfile`** —— 中央目录在文件末尾，缺尾部抛 `BadZipFile`。
  2. **必须处理 ZIP64** —— 当 `usize`/`csize == 0xFFFFFFFF` 时，真实值在
     **extra field `0x0001`** 里；漏了就读出 `4,294,967,295`。
- **实测收益**：urban 只下到 **12.5%（seg0）** 就抽出了 `data.yaml` + 完整 test 集，
  **省掉数小时等待**。

### 7.2 Kaggle `totalBytes` 必须实测

见第一节。**这是本段最大的工程发现**，直接导致：
- urban 的 seg7 反复重下无法完成；
- mbdd2025 的 seg7 同样卡死。

**正解**：用签名 URL + Range 二分定位真实上界。

### 7.3 串行 >>> 并行（GCS 场景）

| 模式 | 实测速度 |
|---|---|
| 8 段并行（+ 其他下载抢带宽） | **~27 KB/s/段** ⇒ 147 MB 40 分钟下不完 |
| **串行（1 段独占）** | **~2.4 MB/s** ⇒ 147 MB 只用 **60 秒** |

⇒ 收尾补段**必须串行**。

---

## 八、留痕

| 文件 | 说明 |
|---|---|
| `logs/_urban_final2.py` | ★ 最终拼接器（纯拼接，不下载） |
| `logs/_urban_final_log.txt` | 拼接日志（含条目聚合） |
| `logs/_urban_stats.py` / `_urban_stats.txt` | 全量 13 类分布统计 |
| `logs/_urban_partial2.py` | ZIP64 修正版局部解析器 |
| `logs/_preview_urban_test.py` / `_preview_urban_test/` | 看图判读（8 张） |
| `logs/_gcs_quota_test.py` / `_gcs_quota_test.txt` | ★ 416 探测（决定真实大小） |
| `logs/_urban_join.py` | 早期（失败的）重算边界版，留作教训 |
| `logs/_segtop.py` | 串行补段器（实测 60 秒/147 MB） |
