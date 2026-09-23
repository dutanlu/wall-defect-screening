# 外墙缺陷数据集 · 网络检索结果（2026-09-20）

> **检索动机**：本项目 7 类体系中 **`exposed_rebar`（露筋）只有 9 框 / 2 图、
> `rust`（锈迹）0 框、`delamination`（空鼓分层）0 框** —— 后两类完全没有训练数据。
> 技术报告 §4 把这三类写成「无数据支撑」，答辩时必然被追问。
> 本次检索目标是**找可获取的公开数据集来补这三类**。
>
> **本文档只做检索与核查，不下载、不改动任何现有文件。**

---

## 一、结论速览（按价值排序）

| 优先级 | 数据集 | 能补哪几类 | 规模 | 格式 | 许可 | 可直接下载？ |
|---|---|---|---|---|---|---|
| **P0** | **MBDD2025**（Ultralytics 镜像） | crack / leakage / abscission / corrosion / **bulge（空鼓）** | **14,363 图**（11.5k/2.9k） | YOLO 检测 | MIT（Kaggle 页） | ✅ 平台内直接可用 |
| **P0** | **Urban Infrastructure Anomalies**（Kaggle） | **delamination / cover_detachment / void** / efflorescence / rust / scaling / spalling …**共 13 类** | **16,946 图** | YOLOv8 txt | Apache 2.0 | ✅ Kaggle 下载 |
| **P1** | **建筑外墙面缺陷检测数据集**（CSDN，意大利数据） | **Armatura in vista（露筋）3673 框** / **Delaminazione（分层）1005 框** / ruggine（锈迹）2550 框 / Fessura（裂缝）4513 框 / Spalling / Scaling / Umidita | **5,737 图 / 15,733 框** | VOC + YOLO | CC BY（社区转发） | ⚠️ CSDN 需积分 |
| **P1** | 混凝土缺陷检测数据集（腾讯云同源） | 同上 7 类（多一档规模） | **7,513 图 / 40,324 框** | VOC + YOLO | 社区转发 | ⚠️ 需积分 |
| **P1** | **Urban Infrastructure Anomalies** 之外：**Kaggle Concrete Surface Defects** | exposed reinforcement / rust stain / Crack / Spalling / Efflorescence / **delamination** | 7,353 图 | YOLO | — | ⚠️ 需核查 |
| **P2** | **BFDD**（Mendeley，RGB-IR 成对） | Cracks / Peeling / **Hollow Areas（空鼓）** / Stains / Erosion | 788 对齐图像对 | 语义分割（像素级） | **CC BY 4.0** | ✅ 直链 528 MB |
| **P2** | **RC segmentation 1841**（Mendeley） | **major spalling/delamination** / rebar corrosion / 8 细分类 | 1,841 图 | 语义分割掩膜 | **CC BY 4.0** | ✅ 直链 143 MB |
| **P2** | **MDMCS**（Mendeley） | cracking / spalling / corrosion / **exposed rebar** | 1,200 图 | 像素级分割 | **CC BY 4.0** | ✅ 直链 195 MB |
| **P3** | 墙面缺陷红外热成像（CSDN） | crack / drop（掉皮）/ **hollow（空鼓）** / leakage | 463 图 | YOLO/COCO/VOC | 社区转发 | ⚠️ CSDN 积分 |
| **P3** | **Roboflow: Defects in Facade Building** | crack / rust / corrosion / **delamination** / dirty,mold / paint defect | 841 图 | 检测框 | CC BY 4.0 | ✅ Roboflow 可导出 |
| **P3** | **Roboflow: elia** | 23 类，含 **Exposed rebar / Growing moss / Concrete spalling** | 3.6k 图 | 检测框 | CC BY 4.0 | ✅ Roboflow 可导出 |
| **P3** | **BD3**（GitHub） | Algae / Major Crack / Minor Crack / Peeling / Spalling / Stain / Normal | 3,965 图 | **仅分类标签**（非检测框） | 论文可引 | ⚠️ 仓库只有样本图，完整集需申请 |

---

## 二、最值得动手的两个（P0 详解）

### 2.1 ★ MBDD2025 —— 唯一能直接补「空鼓」的大规模 YOLO 数据集

- **来源**：Kaggle `mennamahmoudd/mbdd2025-building-defects`；
  **镜像**：`platform.ultralytics.com/dgfd-dfd/datasets/mbdd2025-4`
- **规模**：**14,363 图**（Train 11.5k / Val 2.9k），**2.7 GB**
- **类别（5 类）**：`crack` 裂缝 / `leakage` 渗漏 / `abscission` 剥落 /
  `corrosion` 腐蚀 / **`bulge` 空鼓（墙鼓）**
- **格式**：PASCAL VOC（XML）→ Ultralytics 平台版已是 YOLO 格式
- **采集**：**UAV（无人机）** 拍摄，覆盖 6 种建筑结构类型
  （钢/木/砌体/砖木/砖混/钢筋混凝土）
- **许可**：MIT（Kaggle 页标注）

**对本项目的意义**：
- **`bulge` 是唯一能直接对应我们 `delamination`（空鼓）的公开标注类** ——
  这是我们 7 类里**完全没有数据**的那一类。
- 无人机航拍视角 → 与我们的「地面拍高层」场景**同属远距/斜视**，
  比近距特写数据集更贴近真实使用场景（也正是本项目 `.workbuddy` 里
  「27 cm 拍摄距离做不到」那条物理约束的痛点）。

**注意**：`bulge` 在 Kaggle 原文里是「wall bulges」，语义上偏向
**「墙面鼓包（外观可见的外凸）」**，与我们定义的「**饰面层与基层脱开**」
（内部界面、外观可能正常）**不是同一个物理量**。
⇒ **如果采纳，报告里必须写清楚这是「外观可见鼓包」而非「内部空鼓」**，
不能直接改名成 `delamination` 充数。**这一点需要你来拍板。**

### 2.2 ★ Urban Infrastructure Anomalies —— 类目最全（13 类）

- **来源**：Kaggle `sanjuchinni/urban-infrastructure-anomalies`（1.17 GB）
- **规模**：**~16,946 图**（Train 12.9k / Val 2,772 / Test 1,274），全部框标注
- **类别（13 类）**：`corrosion`、**`delamination`**、**`cover_detachment`**、
  **`efflorescence`**、`crack`、**`rust`**、`scaling`、`spalling`、
  **`void`**、`tile_crack`、`tile_delamination`、`tile_loss`、`peeling`
- **格式**：**YOLOv8 `.txt`** —— 与我们的训练管线**完全兼容，零转换成本**
- **许可**：Apache 2.0（页面另注「学术与非商业用途」）
- **基准**：官方用 YOLOv8s 跑出 mAP@50 **0.8031** / mAP@50-95 0.5970

**对本项目的意义**：
- **同时覆盖我们缺的三类中的两类**：`rust`（锈迹）、`delamination`（空鼓/分层），
  外加 `cover_detachment`（保护层脱落）和 `void`（空洞）两个近义类。
- 还**已有 `efflorescence`（泛碱）** —— 与我们的类别名**逐字对应**。
- YOLOv8 格式 + Ultralytics 兼容 = **可以最快跑通验证**。

**注意**：官方提示 **`void` / `tile_loss` 类样本偏少**，
需加权采样；且 `tile_*` 三类是**瓷砖墙面**，与我们的混凝土/砖墙场景不同源。

---

## 三、中等价值（P1）：能补「露筋」的意大利数据集

**「建筑外墙面缺陷检测数据集 5737 张（VOC+YOLO）」**（CSDN，原始来源疑为
意大利语标注的公开数据，GitHub 仓库名 `datasets_sl`）

| 意大利语类名 | 中文 | 框数 | 图片数 |
|---|---|---|---|
| **Armatura in vista** | **可见钢筋（露筋）** | **3,673** | 2,186 |
| **Delaminazione** | **分层（空鼓）** | **1,005** | 746 |
| Fessura | 裂缝 | 4,513 | 2,509 |
| Spalling | 剥落 | 2,004 | 1,218 |
| **Tracce di ruggine** | **锈迹** | **2,550** | 978 |
| Umidita | 潮湿/渗水 | 1,748 | 903 |
| Scaling | 掉块 | 240 | 160 |
| **合计** | | **15,733** | 5,737 |

- 分辨率 **416×416**，已做增强，未划分训练/验证/测试集。
- 腾讯云有同源版本 **7,513 图 / 40,324 框**，类别多一个 `Efflorescenza`（泛碱）。

**对本项目的意义**：
- **`Armatura in vista` 3,673 框是我们 `exposed_rebar` 缺口的最大单一来源**
  （我们现在只有 9 框）。
- **`Delaminazione` 1,005 框** 也能补空鼓。
- 缺点：**分辨率仅 416×416**，低于我们 640 的输入规格；
  且需 CSDN 积分才能下载（我不能代你购买/登录）。

---

## 四、可直接下载的 CC BY 4.0 学术数据集（P2）

这三个都是 **Mendeley Data 直链、CC BY 4.0、无需登录**，
如需我可以直接抓取——但都是**语义分割掩膜**，不是检测框，
**与我们的 YOLO 检测管线不兼容**，需要自己转框（或改成分割任务）。

| 数据集 | 内容 | 规模 | 直链 |
|---|---|---|---|
| **BFDD** | RGB-IR **成对**图，5 类含 **Hollow Areas（空鼓）** | 788 对 / 640×512 | `data.mendeley.com/public-api/zip/9ych7czvyg/download/1`（528 MB） |
| **RC 1841** | 8 细分类含 **major spalling/delamination**、**rebar corrosion** | 1,841 图 | `data.mendeley.com/public-api/zip/yfhwfcrfmk/download/1`（143 MB） |
| **MDMCS** | cracking / spalling / corrosion / **exposed rebar** | 1,200 图 | `data.mendeley.com/public-api/zip/6x4dzzrs2h/download/2`（195 MB） |

**BFDD 特别值得注意**：它是**红外（IR）+ 可见光（RGB）严格对齐**的成对数据 ——
红外模态能看见 **亚表面脱层（sub-surface delamination）**，
这正是我们在技术报告里写的「**空鼓在可见光下本质上无法检测**」的**技术解法**。
⇒ 如果答辩想回答「那空鼓你们打算怎么办」，**BFDD 是最好的引用依据**。

---

## 五、建议的下一步（需你拍板）

我可以立刻执行的动作，以及各自的影响面：

| 选项 | 动作 | 影响 |
|---|---|---|
| **A** | 下载 **Urban Infrastructure Anomalies**（1.17 GB），抽出 `rust` / `delamination` / `cover_detachment` / `void` 四类 | **可直接补 2 类**（锈迹、空鼓）；YOLOv8 格式零转换 |
| **B** | 抓取 **MBDD2025 镜像**（2.7 GB），抽出 `bulge` 类 | 补「外观鼓包」；但**语义与我们的空鼓不同**，需在报告里注明 |
| **C** | 下载 **BFDD**（528 MB，CC BY） | 不直接用于训练，但**作为「空鼓需红外」的学术论据**写进 §4/§6 |
| **D** | 下载 **RC 1841 + MDMCS**（共 338 MB，CC BY） | 补露筋/锈迹的**分割掩膜**，需转框 |
| **E** | 全部先不动，仅把本检索结果写进技术报告 §4「数据来源调研」 | 零风险，答辩时能答「我们检索过，指出这些公开集」 |

**我的建议**：**先做 A + C**（A 直接可用、C 补最硬的论据），
B 和 D 需要你先确认语义/格式是否接受。

**必须先请你确认的三件事**：
1. **`bulge`（外观鼓包）能否算作我们的 `delamination`（内部空鼓）？**
   我倾向**不能直接等同**，但可以标注为「近似类」。
2. **接受外来数据集后，类别定义是否需要重新对齐？**
   例如 Urban Infrastructure 的 `delamination` 与我们的定义是否一致。
3. **是否要重训？** 一旦并入新类别数据，按「1重训2改正3ok」的约定，
   需要重跑全管线（这会改变已定稿的 V2 指标）。

---

## 六、检索方法说明（可复现）

- 检索词：`exposed rebar defect dataset facade`、`外墙缺陷数据集 露筋 锈迹 空鼓`、
  `concrete delamination dataset annotation`、`building facade defect dataset`、
  `空鼓 红外 热成像 数据集`、`Roboflow building facade defect`。
- 逐一 WebFetch 核实每个数据集的**原始页面**（Kaggle / Mendeley / Roboflow / GitHub），
  提取**精确**的图片数、类别名、标注格式、许可、下载链接。
- **未核实项已明确标注**（如 CSDN 类条目的许可与真实性无法从原页面确认，
  只能引述社区转载内容）。
- **未下载任何文件**，未改动项目内任何数据。
