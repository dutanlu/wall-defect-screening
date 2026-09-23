# HRCDS 数据集人工判读结论（抽样看图）

> 生成时间：2026-09-21 凌晨
> 数据源：`01_data/raw/_public_datasets/hrcds/HRCDS.zip`（204.92 MB，3711 条目，ZIP 校验通过）
> 判读人：AI（自主决策模式）
> 渲染脚本：`logs/_preview_hrcds.py` → 输出 `logs/_preview_hrcds/` 16 张带框图

---

## 一、数据集基本信息（已实测）

| 项 | 值 |
|---|---|
| 归档大小 | 204,919,973 B（204.92 MB）；目标 204,920,000 B，差 27 B（尾部 padding，可视为完整） |
| 条目总数 | 3,711 |
| 目录结构 | `HRCDS/{train,val,test}_{annotations,image,mask}` + `test_mark_color` |
| 图片 | 1,200 jpg（1080×720，**全部同一尺寸**）+ 1,300 png（二值/彩色掩码） |
| 标注 | 1,200 个 JSON |
| 划分 | train 1001 / val 101 / test 101（图数各 1001 / 101 / 101） |
| 标注格式 | **LabelMe 5.5.0 多边形**（`shapes[].points`，**全部 `shape_type = polygon`，4536 个**，无矩形） |
| 图像尺寸 | 100% 为 1080×720，无例外 |

## 二、★ 类别分布（实测）

| label | 形状数 | 出现图数 | 图占比 |
|---|---:|---:|---:|
| **`exposed rebar`** | **1578** | **553** | **46.1%** |
| `spalling` | 1550 | 1020 | 85.0% |
| `corrosion` | 753 | 326 | 27.2% |
| `crack` | 655 | 202 | 16.8% |

每图形状数：均值 3.78 / 中位 3 / 最大 30 / P95 11。

**注意两点**：
1. `exposed rebar` 是**带空格**的写法（不是 `exposed_rebar`）⇒ 并入时需要改名。
2. 多类**共存于一图**（如 `test_0068` 同时有 spalling + exposed rebar；`train_0955` 同时有 1 个 spalling + 4 个 exposed rebar）。

## 三、★★ 看图判读：四要素结论

### 3.1 场景（scene）—— ✅ **高度对口**

抽样 16 张（exposed rebar 4 / spalling 4 / corrosion 4 / crack 4），**全部为混凝土结构表面近距离拍摄**：

| 编号 | 文件 | 判读 |
|---|---|---|
| 09 | `exposed_rebar_train_0286` | 剥落坑内**一根竖向锈蚀钢筋**露出，红框精准贴合 |
| 10 | `exposed_rebar_val_0087` | 混凝土剥落区露出**蓝色箍筋 + 白色析出物** |
| 11 | `exposed_rebar_train_0955` | 同一破损区：大面积 spalling + **4 处 exposed rebar** |
| 12 | `exposed_rebar_test_0068` | spalling 区域内中央露筋一条 |

⇒ 与我们「外墙/混凝土结构病害」的场景设定**一致**，属结构本体特写，不是航拍大场景。

### 3.2 粒度（granularity）—— ✅ **可用**

- 目标本体占图面积：露筋样本平均框占比 **12%–47%**，属中大型目标。
- 但存在小目标：`train_0392` 的 corrosion 平均框占比仅 **0.74%**、`train_0816` 的 crack 仅 **1.05%**。
- ⇒ **目标尺寸跨度大**（0.7% ~ 86%），训练时需注意小目标召回；与现有数据集混训需保持 640 输入下的一致性。

### 3.3 语义（semantics）—— ✅ **匹配，且有额外价值**

| HRCDS 类 | 我们的对应类 | 匹配度 | 说明 |
|---|---|---|---|
| `exposed rebar` | `exposed_rebar` | **★ 完美** | 正是一直观望的「钢筋外露」，**553 张图、1578 个框**，量足够 |
| `spalling` | `spalling` | ✅ 一致 | 混凝土保护层剥落掉块 |
| `corrosion` | ⚠️ **本项目 7 类中无 `corrosion` 这一类** | 🔶 不得写「一致」 | 语义 = 表面锈迹/锈水渗流（`train_0327` 为典型锈迹带），最接近本项目的 `rust`；<br>本项目 **不并入该类**（见 §四），故不影响结论，但**映射关系不得表述为同名同类** |
| `crack` | `crack` | ✅ 一致 | 混凝土裂缝（`test_0033` 为典型贯穿裂缝） |

**额外价值**：`train_0955` 等图给出了 **spalling ↔ exposed_rebar 的同框对照标注**，为区分这两类提供了明确的标注范式（先剥落、剥落深处露筋）。

### 3.4 格式（format）—— ⚠️ **需转换，成本可控**

| 项 | HRCDS | 我们管线 | 转换成本 |
|---|---|---|---|
| 标注类型 | LabelMe **多边形**（points） | YOLO **矩形框**（cx cy w h） | 取多边形外接矩形即可，脚本 30 行内 |
| 类名 | `exposed rebar`（空格） | `exposed_rebar`（下划线） | 一行 `replace(' ', '_')` |
| 划分方式 | 已划好 train/val/test | 需重划或用现有 | 建议**只取 train 集**并入我们训练集，避免与我们的 test 冲突 |
| 图片格式 | jpg 1080×720 | 通用 | 无需转换 |

⇒ **转换成本低**：一个脚本即可产出 YOLO 格式。

## 四、★ 结论与建议（供用户决策）

**结论：HRCDS 是本轮唯一能够真正补上 `exposed_rebar` 空缺的数据集，建议纳入。**

**建议的并入方案（待用户确认）**：

1. **只并入 `exposed rebar` 一类**（1578 框 / 553 图）。
   - 理由：其余三类（spalling / corrosion / crack）我们已有充足数据，并入会改变既有类的分布，收益低、风险高。
   - 且 HRCDS 画风与现有数据虽同属结构特写，但仍是不同来源，只并就补空缺类是**最小侵入**做法。
2. **训练/验证划分**：从 HRCDS 的 **train 集（1001 图）** 中取含 `exposed rebar` 的图并入我们的 train；**不要**用它的 test/val（避免与我们独立测试集混淆）。
3. **多边形 → 外接矩形** 转换，类名改为 `exposed_rebar`。
4. **⚠️ 前置校验**：并入前需抽查转换后的框是否贴合（多边形外接矩形对细长钢筋可能偏大），建议渲染转换后结果再目视一次。

**⚠️ 需用户拍板的三点**：
- (a) 是否接受「多边形 → 外接矩形」的框精度损失？
- (b) 是否同意只并入 `exposed_rebar` 单类（而非全部 4 类）？
- (c) HRCDS 的 `exposed rebar`（转换后 **464 图 / 1,392 框**，见 §5.1 实测）
  与此前 famerL 源的 79 框裂缝是否同时并入？

## 五、★★ 转换已实测跑通，并做了二次目视校验（今晚已完成，不需 GPU）

### 5.1 转换脚本已写好并跑通

`logs/_hrcds_to_yolo.py`（LabelMe 多边形 → YOLO 矩形，类名 `exposed rebar` → `exposed_rebar`）

**dry-run 统计（仅 train 划分、仅 exposed_rebar 单类）**：
```
导出图片数 : 464
导出框数   : 1392
丢弃(小框) : 0
跳过(类别) : 2650   （其余三类不导）
```

**实际导出已执行**，产物在：
`01_data/raw/_public_datasets/hrcds_yolo_exposed_rebar/{images,labels}/`
- `images/`：464 张 jpg（命名 `hrcds_train_XXXX.jpg`，避免与既有文件重名）
- `labels/`：464 个 txt（格式 `0 cx cy w h`，单类 `exposed_rebar` 索引 0）

### 5.2 ★★ 二次目视校验 —— 外接矩形贴合度**通过**

担心「多边形 → 外接矩形」对细长钢筋会偏大，故写 `logs/_verify_hrcds_yolo.py`
把转换后的 txt 框画回图，抽样 8 张目视：

| 图 | 框数 | 平均面积占比 | 判读 |
|---|---:|---:|---|
| `train_0404` | 4 | 1.79% | ✅ 4 框精准圈住 4 根外露钢筋，贴合好 |
| `train_0784` | 7 | 0.81% | ✅ 7 条竖向细长钢筋全部框准（含最小者） |
| 其余 6 张 | 1–3 | 0.12%–5.03% | ✅ 未见明显偏大或错位 |

⇒ **「多边形 → 外接矩形」的精度损失可接受，转换质量已验证。**
⇒ 带框图见 `logs/_verify_hrcds_yolo/`（8 张），清单 `logs/_verify_hrcds_yolo.txt`。

### 5.3 ⚠️ 仍未执行的（需用户拍板后才做）

**没有把它并入现有训练集** —— 因为并入需用户确认（见 §四）。当前只是把转换产物**备好**，
放在 `hrcds_yolo_exposed_rebar/` 独立目录，**未触碰任何现有数据集**。

---

## 六、留痕

| 文件 | 内容 |
|---|---|
| `logs/_preview_hrcds.py` | 抽样看：多边形+外接矩形叠加渲染（16 张） |
| `logs/_preview_hrcds/` | 第一次目视判读的带框图 |
| `logs/_preview_hrcds_report.txt` | 上述清单 |
| `logs/_hrcds_probe.py` → `_hrcds_probe.txt` | zip 结构 / 目录分布 / JSON 字段探测 |
| `logs/_hrcds_labels.py` → `_hrcds_labels.txt` | 类别分布 / 尺寸分布 / bbox 覆盖率统计 |
| **`logs/_hrcds_to_yolo.py`** | **转换器（可复用，支持 --dry-run / --all-classes / --splits）** |
| `logs/_hrcds_convert_report.txt` | 转换统计 |
| **`logs/_verify_hrcds_yolo.py`** | **转换质量校验器** |
| `logs/_verify_hrcds_yolo/` + `.txt` | 校验用带框图（8 张） |
| `01_data/raw/_public_datasets/hrcds_yolo_exposed_rebar/` | **转换产物（464 图 / 1392 框，待并入）** |
