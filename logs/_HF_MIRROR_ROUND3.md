# hf-mirror 通道扩源核查报告

生成时间：2026-09-20 22:25
执行人：WorkBuddy
工作目录：`D:\pythonstudy 备份\创新题\外墙缺陷筛查\`

---

## 一、本报告要解决的问题

承接上一轮结论：Mendeley 三数据集 + BFDD 的下载速度实测仅 **15 KB/s**（国际出口被限速，
链路 中国 → Cloudflare 法兰克福 → AWS S3 爱尔兰 eu-west-1），剩余约 1 GB 需约 18 小时。

**用户决策：先走方案 3 —— 去 GitHub 上找镜像 / 备选通道。**

---

## 二、通道探测结果（本轮新增）

### 2.1 探测总表

| 通道 | 状态 | 实测速度 | 说明 |
|---|---|---|---|
| **hf-mirror.com** | ✅ **可用** | **54.8 KB/s** | **本轮最大收获，比 Mendeley 快 3.7 倍** |
| huggingface.co（官方） | ❌ 不通 | — | `CONNECT tunnel failed, response 502` |
| GitHub 网页 / raw | ✅ 可用 | 0.9–3.0 KB/s | 慢，只能拿小文件 |
| GitHub API | ⚠️ 限流 | — | 未认证 60 次/小时，已触发 `rate limit exceeded` |
| ghproxy.net | ✅ 可用 | 1.1 KB/s | 不理想 |
| gh-proxy.com | ❌ 不通 | — | 502 |
| gitee.com | ✅ 可用 | 890 KB/s | 仅代码托管，无本项目数据 |
| modelscope.cn 首页 | ✅ 可用 | 28.9 KB/s | **但数据集检索 API 返回空，未验证出可用数据** |

### 2.2 关键突破：hf-mirror 访问方式

**元信息接口**（无需登录）：
```
GET https://hf-mirror.com/api/datasets/<owner>/<name>
    → 返回 downloads / likes / gated / siblings(文件清单) / cardData

GET https://hf-mirror.com/api/datasets/<owner>/<name>/tree/main/<dir>?recursive=true&expand=true
    → 返回目录下的文件与 oid

GET https://hf-mirror.com/datasets/<owner>/<name>/raw/main/README.md
GET https://hf-mirror.com/datasets/<owner>/<name>/resolve/main/<path>   ← 单文件下载
```

**搜索接口**：
```
GET https://hf-mirror.com/api/datasets?search=<kw>&limit=30
```

### 2.3 ★ 必须记录的坑

1. **目录接口有 50 条硬上限**：`?recursive=true` 不生效于深层目录，
   `1260/` 只返回 49 个文件，`scen1/test2017` 只返回 50 个。
   → **绕法：靠文件名规律构造 URL 逐个探测**（本项目 ROI-1555 文件名是
   `000000000001.jpg` 十二位零填充，完全可预测）。
2. **PowerShell 会吞掉 URL 中的 `?`**：`$u = '.../tree/main/1260?recursive=true'`
   里 `?r` 被解析成参数。必须用**单引号 + 字符串拼接**：
   `'https://.../tree/main/' + $d + '?recursive=true&expand=true'`。
3. **PowerShell 执行策略禁止运行 .ps1**：`无法加载文件...因为在此系统上禁止运行脚本`。
   → **改为把脚本内容内联到 PowerShell 工具调用里**，不要落地成 .ps1 再执行。

---

## 三、★ 本轮最有价值的发现：ROI-1555

### 3.1 基本信息

| 项 | 值 |
|---|---|
| 仓库 | `tsrobcvai/ROI-1555_Rebar_Detection_and_Instance_Segmentation_Dataset` |
| 全名 | ROI-1555: Rebar Detection and Instance Segmentation Dataset |
| 规模 | **1555 张露筋（rebar）图像** + 精细标注 |
| 标注形式 | **LabelMe 格式 JSON**（多边形 + 外接矩形） |
| 标注类别 | `straight`（直筋）、`hoop`（箍筋） |
| 许可 | 学术可用，需引用论文 |
| 论文 | Sun, Fan, Shao. *Deep Learning-based Rebar Detection and Instance Segmentation in Images*. Advanced Engineering Informatics, 65:103224, 2025 |
| 机构 | McGill University（Shao Lab） |
| 作者联系 | tao.sun@mail.mcgill.ca |
| **补我们的** | **`exposed_rebar`（露筋）—— 零数据类之一** |

### 3.2 目录结构（实测）

```
1260/img_label/     1260 张 + json      ← 主集
scen1/test2017/       ~90 张 + json     ← 场景 1
scen2/test2017/       ~68 张 + json     ← 场景 2
scen3/test2017/      ~137 张 + json     ← 场景 3
doc/                 3 张示意图
tools/               labelme2coco_instance.py 转换脚本
```

⚠️ **注意**：README 写的 `1260` 是主集张数，但**全库共 1555 张**。
`1260/` 下的编号实测到 `000000001300` 仍为 200，且 `000000000999` 缺号，
说明编号不连续，**必须按 404 停止法逐个探测**。

### 3.3 标注格式实例（`1260/img_label/000000000001.json`）

```json
{"version": "4.5.9", "flags": {}, "shapes": [{"label": "straight",
  "points": [[120.27,184.58],[135.28,193.59],...],
  "group_id": null, "shape_type": "polygon", "flags": {}}],
 "imagePath": "000000000001.jpg", "imageData": null,
 "imageHeight": 800, "imageWidth": 1333}
```

- 标准 **LabelMe polygon**，`points` 是像素坐标
- **可直接算外接矩形转 YOLO 检测框**（不需要掩膜）
- `imageHeight` / `imageWidth` 齐备，归一化没有问题

### 3.4 下载测速

| 方式 | 速度 |
|---|---|
| 单文件（354 KB jpg） | 46.3 KB/s |
| Range 读 10 MB | **54.8 KB/s** |

### 3.5 规模估算

- 平均单图 **319.5 KB**
- 1555 张 × 319.5 KB ≈ **485 MB**（仅图像）
- 加标注 JSON ≈ **500 MB**
- @54.8 KB/s ≈ **2.5 小时**

---

## 四、其他候选核查结果

### 4.1 ✅ 可用（已确认结构）

| 仓库 | 规模 | 类别 | 补我们的 | 备注 |
|---|---|---|---|---|
| `famerL/construction_defect_yolo` | 1 个 val80 zip | 建筑缺陷 | ? | 需下载解压看类别，仅 9 次下载，来源不明 |
| `mohammadnajeeb/concrete_crack_images` | 10K–100K | 混凝土裂缝 | `crack` | **分类**数据集（imagefolder），非检测 |
| `kamyarazimi/ConcreteCrackDataset` | — | 裂缝 + 掩膜 | `crack` | `imgs_masks.zip`，是**分割**不是检测 |
| `varcoder/crack-segmentation-dataset` | 9603 训练 / 1695 测试 | 裂缝分割 | `crack` | parquet 格式，1.26 GB，**分割** |
| `fcakyon/crack-instance-segmentation` | 323/73/37 | `cracks-and-spalling` | `crack`,`spalling` | 已充足，暂不补 |
| `lipi17/building-cracks` | 1490/433/211 | `crack` | `crack` | 已充足 |
| `lipi17/Building-Cracks-Merged` | 947/433/11 | `crack`,`stairstep_crack` | `crack` | 已充足 |
| `Francesco/wall-damage` | 1K–10K | `wall-damage` 等 4 类 | ? | rf100 子集，类别语义偏"旋转"，需看图 |
| `onlywonhand/SDD-Seg_Structural_Damage_Dataset` | **100K–1M** | 结构损伤分割 | ? | 韩国 AI Hub，cc-by-nc-4.0，17 个 7z 分卷，**体量可能极大** |

### 4.2 ❌ 排除

| 对象 | 排除原因 |
|---|---|
| **`jiange1236/StructuralCracksDataset`** | **★ 判错纠正**：见下节 |
| `seshing/buildingfacade` | 任务类型 `zero-shot-classification`，是**建筑外观分类**不是缺陷检测 |
| `l985215117/welding-defect-object-detection` | 焊接缺陷，与建筑外墙无关 |
| 各种 `car-damage` / `road-damage` / `wood_surface_defect` | 领域不匹配 |
| `keremberke/construction-safety-object-detection` | 是**施工安全**（安全帽/背心），非缺陷 |

### 4.3 ★ `jiange1236/StructuralCracksDataset` —— 判错纠正记录

**初判（错误）**：README 类别列出「Reinforcement exposed and corroded（钢筋露筋锈蚀）」
「Sag of protecting coating（保护层剥落）」等，看似直接命中我们的 3 个空缺类，一度被视为
本轮第二重大发现。

**复判（正确）**：下载实际标注后推翻。理由：

1. **规模只有 27 张**（`annotations.jsonl` 27 行），完全没有实用价值。
2. **定位是 Qwen-VL 多模态大模型微调，不是目标检测**。标注产物是自然语言描述，
   例如：`"一个 '露筋' 位于图像的 中部偏左区域。"` —— 用的是**方位词**。
3. **坐标被丢弃**。查 `convert_labels.py:114-139`：Label Studio 导出的
   `x/y/width/height` 百分比坐标**只被用于生成"顶部中心区域"这类文字**，
   从未写入 jsonl。注释 `# width/height设为0，只用中心点` 是直接证据。

**结论**：**无法用于 YOLO 训练**。归入「看着完美但不可得」类别，
与 HKIE Transactions 论文数据集同列。

**教训**：`README.md` 里列的类别清单是**声明**，不是**可用的目标框**。
必须下载真实标注文件、验证它含**像素/归一化坐标**，才能判定可用于检测训练。

---

## 五、当前行动状态

### 5.1 已启动

**ROI-1555 全量下载**（后台任务 `13e1r4`）

- 目标：`01_data/raw/_public_datasets/roi1555/`
- 覆盖目录：`1260/img_label`（上限 1700）、`scen1/test2017`（250）、
  `scen2/test2017`（250）、`scen3/test2017`（350）
- 策略：按编号规律构造 URL，逐张探测；**连续 30 个 404 即停止**该目录
- 幂等：已存在且 >200 B 的文件跳过；非 200 响应删除残留
- 日志：`logs/_dl_roi1555.log`

### 5.2 待处理

- Mendeley 的 2 个 curl 进程仍在跑（`_dl_all` 批次），**建议停掉**以释放带宽
- `openxlab` 安装仍未成功（已 40+ 分钟），CODEBRIM 能否下到仍是未知数

---

## 六、方法论教训（追加）

1. **`hf-mirror.com` 是国内访问 HuggingFace 的可用通道**，速度是 Mendeley S3 的 3.7 倍。
   应优先用 HF 而非 Mendeley 找数据集。
2. **README 的类别清单 ≠ 可用标注**。必须下载真实标注文件，验证含坐标。
3. **目录接口有 50 条上限**。文件名有规律时可绕过清单直接构造 URL。
4. **PowerShell 会吞 URL 里的 `?`**。用单引号 + 拼接。
5. **本机禁止执行 .ps1**。脚本要内联到工具调用里。
6. **分类数据集 ≠ 检测数据集**。`imagefolder` / parquet 常是分类，不能直接训 YOLO。
