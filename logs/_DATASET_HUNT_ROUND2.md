# 补充数据源检索（第二轮）—— 趁下载期间扩源

> 记录时间：2026-09-20 21:33
> 目标：为 `rust` / `delamination` / `exposed_rebar` 三类找**可匿名下载**的补充数据。
> 前提约束：Kaggle / Zenodo / HuggingFace / Figshare 在本机网络均不可用。

---

## 一、平台可用性复测（第二轮，新增平台）

| 平台 | 结果 | 说明 |
|---|---|---|
| Figshare API | ❌ 403 | 被拦截 |
| Zenodo API | ❌ 403 | 同 IP 反爬，稳定失败 |
| **Harvard Dataverse** | ✅ 通 | 有数据，但**全是土木材料实验数据**（X-ray CT、配合比、耐火试验），非图像检测集 |
| Roboflow Universe | ✅ 通 | 可用，但需甄别 |
| IEEE DataPort | ✅ 通 | 待细查 |
| OSF | ❌ exit=3 | 不可达 |
| arXiv API | ✅ 通 | 可用于找论文，再顺藤找数据 |

**结论**：可用的匿名源基本只剩 **Mendeley + Roboflow + GitHub + arXiv**。

---

## 二、★ 本轮最有价值的发现

### 2.1 CODEBRIM（CVPR 2019，质量最高但**下不了**）
- **类别**：`crack`、`spallation`、`efflorescence`、**`exposed bars`（露筋）**、**`corrosion (stains)`（锈迹）**、background —— **与我们的需求高度吻合**
- **关键优势**：`CODEBRIM_original_images` 子集包含**原始全分辨率图 + 边界框标注**（不只是分类标签）
- **出处**：法兰克福大学，`10.5281/zenodo.2620293`；许可证**仅限非商业与教育用途**
- **❌ 阻塞**：托管在 Zenodo，本机访问 **403 稳定失败**；OpenDataLab 只是转发 Zenodo 源，不托管文件
- **可尝试的替代路径**：OpenDataLab（`openxlab.org.cn/datasets/OpenDataLab/CODEBRIM`）若注册后提供镜像下载，可绕过 Zenodo —— **待验证**

### 2.2 香港工程师学会论文数据集（HKIE Transactions）—— **未公开**
- 内容：1,907 图，8 类，类别**几乎与我们的 7 类一一对应**：
  `cracks`(684) / `delamination`(462) / `exposed reinforcement`(692) / `rust stains`(489) / `spalling`(493) / `tile cracks`(364) / `tile delamination`(254) / `tile loss`(433)
- 来源：香港 30–50 年楼龄停车场与住宅楼实拍，Canon EOS 70D + iPhone 12，LabelImg 标注，YOLOv5 格式
- **❌ 结论**：论文全文我已下载并提取（`logs/_hkie_paper.pdf`），**通篇未给出公开下载地址**，仅说明上传至 Roboflow 做预处理。**数据集不可获取。**

### 2.3 CSDN / 腾讯云中文数据集（内容吻合但**商业渠道**）
- **建筑墙壁损伤缺陷数据集**：6,872 图 / 19 类 / VOC+YOLO 格式 / 54,179 框
  - 关键类别框数：`Rust` **12,844**、`Spalling` **8,484**、`Cavity` 8,119、`Weathering` 4,066、`Efflorescence` 3,454、`Crack` 3,155、`ExposedRebars` **1,755**、`Hollowareas` **1,362**
  - 还有 labelme 版本：7,820 图 / 20 类 / polygon 标注
- **【73】墙壁建筑缺陷数据集**：1,522 图 / 8 类，类别名 `crack` / `delamination` / `exposed reinforcement` / `rust stain` / `spalling` / `tile crack` / `tile delamination` / `tile loss`，**与 HKIE 论文同源**
- **❌ 结论**：全部为 CSDN 积分 / 付费 / 引流渠道（发布者 `firc-dataset`、`nept` 等），**非公开免费**。

### 2.4 MACILLAS/Structural_Defects（GitHub）
- 聚合 CODEBRIM / Zhang / QuakeCity / S2DS 四个数据集，统一转 YOLO 格式
- **❌ 结论**：仓库**不含数据文件**，只是转发 Roboflow 链接（`universe.roboflow.com/cvisslab/*`）
- **✅ 有价值信息**：其中 **Zhang 数据集**含 `corrosion-rebar` 类，YOLOv8x 上 mAP50 达 **0.593**（255 实例），是已验证可训的 rust 类数据源

---

## 三、Roboflow 可用候选（网络通，但需逐个验证导出权限）

| 数据集 | 规模 | 相关类别 | 备注 |
|---|---|---|---|
| `cvisslab/zhang-3seb8` | 289 图 | `corrosion-rebar` 255 实例 | mAP50 0.593，已验证可训 |
| `cvisslab/s2ds` | 122 图 | `corrosion-rebar` 320 实例 | S2DS 原版 |
| `cvisslab/codebrim-poidd` | 213 图 | `corrosion-rebar` 61 实例 | CODEBRIM 的检测版 |
| `yolo-concrete-damage-detection/yolov11_cdam_test` | 137 图 | `Crack`/`Rebar`/`Spall` | CC BY 4.0，YOLOv11 专用 |

⚠️ Roboflow 导出通常需要登录账号并获取 API key，**未必能匿名下载** —— 需实测确认。

---

## 四、当前行动结论

1. **已确定可下且正在下载**：Mendeley 三个数据集（RC2119 / RC1841 / HRCDS），共 491 MB，**能覆盖全部 3 个空缺类**。
2. **CODEBRIM 值得再试一次**：通过 OpenDataLab 注册后看是否提供国内镜像直链，这是**质量最高、语义最对口**的备选。
3. **中文数据集不可用**：CSDN/腾讯云渠道均为付费，且原始出处不明，不符合「忠于原始资料」原则。
4. **Roboflow 作为二线备选**：若 Mendeley 数据的掩膜转框质量不理想，可回头测 Roboflow 的 `cvisslab/zhang-3seb8`（已有 rust 类验证结果）。

## 五、方法论教训（追加）

- **Zenodo 在本机网络稳定 403**，凡托管在 Zenodo 的数据集（含 CODEBRIM）都需找镜像或替代源。
- **中文数据集平台的"数据集介绍"文章 ≠ 可下载**：腾讯云/CSDN 上的数据集文章多是售卖号的引流内容，**看文章不如先测链接**。
- **论文里描述的数据集常常并未公开**：HKIE 那篇 8 类数据集极为对口，但全文无下载地址 —— **必须读原文确认，不能只看搜索结果摘要**。
- **GitHub 聚合仓库未必含数据**：`MACILLAS/Structural_Defects` 声称"accumulate several defect detection datasets"，实际只有转发链接。
