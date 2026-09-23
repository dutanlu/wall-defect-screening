# 建筑外墙缺陷快速筛查系统

> 第六届萌新种子杯 · 视觉赛道 · 创新题
> 北京建筑大学 · 光启 Ray-space 工作室

---

## 一句话定位

**我们不是又一个裂缝检测器，而是第一个「知道自己什么时候不可靠」的外墙筛查系统
—— 因为在房屋安全这件事上，误判为安全比漏检更危险。**

**赛道定位**：本作品**最接近赛道五「可解释性与人机交互」**。

需要主动说清的是：我们的「可解释」**不是注意力热力图（GradCAM）那一类**，
而是**可复算的几何判据** —— 「目标实际尺寸 ÷ GSD = 图上几个像素」，再与判读门槛比较。
它回答的不是「模型看了哪里」，而是「**这张图够不够格让我们下结论**」；
因为判据只是一条除法，**任何人对同一张图都能算出同一个结论**。

> GradCAM 我们**也做了** —— 作为**补充证据**验证「模型确实在看缺陷区域」
> （产物 `04_results/gradcam/`）。但它回答的是另一个问题，**不是我们的主张**。
> 一句可背的话：「GradCAM 回答『模型看哪里』；我们回答『**该不该下结论**』。」

---

## 为什么这个题目不只是一个 YOLO 应用

「用 YOLO 检测外墙裂缝」本身**不创新**（SHM 领域成熟方向，2018 年就有
SDNET2018 / DeepCrack 等 benchmark）。创新性来自以下四点：

### 1.【最强】把「图像能不能用」变成系统的一等公民

现有研究全都隐含假设「输入图像是合格的」。没有人问：
**站在楼下 30 米外仰拍的照片，能不能用来判结构安全？**

物理上是硬约束：

```
GSD（地面采样距离）= 画面水平覆盖宽度 ÷ 图像水平像素数
可判读性 = 目标实际尺寸 ÷ GSD      （经验门槛：≥ 3 像素才可靠）
```

| 拍摄距离 | 主摄 GSD | 5x 长焦 GSD |
|---|---|---|
| 5 m | 1.9 mm/px | 0.38 mm/px |
| 10 m | 3.8 mm/px | 0.75 mm/px |
| 20 m | 7.5 mm/px | 1.5 mm/px |
| 50 m | 18.8 mm/px | 3.8 mm/px |

国标 GB 50010-2010 判的是 **0.2~0.3mm** 裂缝。要可靠判读 0.3mm，需要
GSD ≤ 0.1 mm/px → **主摄须贴近到 27cm，5x 长焦须 1.3m**。
**地面拍高层根本做不到。**

所以系统的核心能力是：**自己判断图像够不够格，不够格就拒绝给定量结论**
（selective prediction / abstention）。

### 2. 输出从「框」升级为「毫米 + 规范等级」

detection → assessment 的跃迁，可直接对接房屋体检「一房一档」台账。
依据 GB 50010-2010（环境类别裂缝限值）与 JGJ 125-2016（危险房屋鉴定）。

### 3. 真实的科学问题：距离域偏移

公开数据集均为近距采集（照片里裂缝占几十像素），实际部署是远距拍摄
（占几个像素甚至亚像素）。这是可研究的域偏移问题 —— 做跨域实验即得技术深度。

### 4. 应用新鲜度：制度配套工具

房屋体检是 2025-2026 新政（十五五规划提出「建立房屋全生命周期安全管理制度」；
住建部推动房屋体检制度；南通、郑州、长沙、宁波已落地地方办法）。
配套工具尚缺位。

---

## 技术架构：YOLO + OpenCV 串联，不是二选一

```
        ┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────┐
图像 ──▶│  YOLO       │──▶│  尺度标定     │──▶│  OpenCV     │──▶│  规范分级 │
        │「在哪里、   │   │  mm/px       │   │「有多大」    │   │ severity │
        │  是什么」    │   │  + 可判读性   │   │  mm 级量化   │   │ A/B/C/D  │
        └─────────────┘   └──────────────┘   └─────────────┘   └──────────┘
```

- **YOLO 负责「在哪里、是什么」**：缺陷定位 + 类别
- **OpenCV 负责「有多大」**：像素级宽度/长度/面积 + 尺度标定
- **规则代码负责国标分级**

为什么不能只用 YOLO：bbox 无法回答「这条裂缝是 0.35mm 还是 1.2mm」；
裂缝细长，框内宽度信息在 bbox 表征里是缺失的，必须回到像素级操作。

为什么不能只用 CV：分不清裂缝与砖缝/污渍/电线阴影，且无法多类别语义分类。

**最大优势：与基础题完全同构**（基础题 = YOLO 定位水果 + OpenCV HSV 测量），
复用同一套架构骨架与调试经验。

---

## 目录结构

```
外墙缺陷筛查/
├── 01_data/
│   ├── audit/         数据体检报告（step1 产出，不占 04_results/eval）
│   ├── raw/           原始下载（各源独立子目录，zip 原件归档在 _archived_zips_*）
│   ├── dedup/         去重后（含 _removed_duplicates 隔离区）
│   ├── unified/       类别归一后（统一 7 类）
│   └── dataset/       最终数据集 images/{train,val,test} + labels/ + wall_defects.yaml
├── 02_code/           源代码（见下表）
├── 03_weights/        预训练权重 + 训练产物
├── 04_results/
│   ├── train/         训练输出（Ultralytics 标准结构）
│   ├── eval/          模型评估指标（**只放评估结果，不放数据报告**）
│   ├── vis/           可视化标注图
│   ├── ablation/      对照实验
│   └── quant/         FP32 + INT8 ONNX 与量化报告
├── 05_quantify_grade/ 国标分级专项（判据清单，机器可读）
├── 06_deploy/         工程化（Web 演示 app.py）
├── 07_report/         报告与提交物
│   ├── 技术报告.md / 答辩材料.md / 演示视频脚本.md / README.md
│   ├── 外墙缺陷智能筛查_答辩PPT.pptx          提交物① 答辩 PPT（20 页）
│   ├── 项目介绍书.md · 项目介绍书.docx         提交物② 项目介绍书（493 字）
│   ├── 外墙缺陷智能筛查_实机运行视频.mp4      提交物③ 实机运行视频（1:09）
│   └── _not_submitted_演示视频/              旧的合成演示视频，不提交，仅留档
├── 08_photo_collector/ 网友照片收集站（已云端发布）
├── logs/              运行日志与核查留痕
└── _archive_20260920/ V2 陈旧产物归档（保留证据链）
```

### 当前数据资产（2026-09-21 V3 定稿）
- **原始 zip**：`01_data/raw/greybrick/_archived_zips_20260920/`（4 个，259 MB，完好）
- **解压源**：`01_data/raw/greybrick/src/`（会被 step2 消费）+ 备份 `src_pristine/`
- **可用数据集**：`01_data/dataset/`（**2848 图 / 18262 框 / 7 类**，`wall_defects.yaml`）
  —— 划分 **1995 / 568 / 285**（train / val / test），跨集交集 9 项全为 0（**字节级**）
- **七类均有数据**：`rust` 13685 框 / `exposed_rebar` 1389 框 / `delamination` 694 框
  于 2026-09-21 由 HRCDS + Urban 两源并入（详见 `07_report/技术报告.md` §4.1.1 / §4.4）

### 代码清单（02_code/）

| 脚本 | 作用 | 关键点 |
|---|---|---|
| `common.py` | 公共工具 | 中文路径安全 I/O、**7 类体系唯一真源**、BOM 兼容文本读写、权重本地解析 |
| `gsd.py` | **核心创新** | 三种标定 + 可判读性判定 + 筛查能力表 |
| `measure.py` | 几何量化 | Otsu 分割、Zhang-Suen 骨架化、距离变换测宽 |
| `grade.py` | 规范分级 | GB 50010 / JGJ 125 → severity + 房屋危险性 |
| `pipeline.py` | 端到端组装 | 检测→标定→可判读性→量化→分级→可视化 |
| `rectify.py` | 几何前处理 | 斜拍正射校正 + 多段拼接（含周期混叠拒答） |
| `step0_download.py` | 数据下载（Roboflow） | Roboflow SDK 通道（**本机网络不可用，保留备查**） |
| `step0b_download_modelscope.py` | 数据下载（国内镜像） | **实际使用**：ModelScope HTTP 端点 + Python 解压校验 |
| `step1_scan.py` | 数据体检 | 分辨率分布、小目标占比（支撑域偏移论证） |
| `step2_dedup.py` | 去重 | MD5 + dHash 三级；图像与标签成对处理；**默认复制，`--move` 才移动** |
| `step3_unify.py` | 类别归一 | 可配置映射表；未命中**不静默吞掉** |
| `step4_split.py` | 划分 | 7:2:1 分层 + **跨集交集 9 项自检** |
| `step5_fetch_rebar.py` | 补充数据源（露筋） | label-studio → YOLO 转换；**产物隔离，未并入主数据集** |
| `train.py` | 训练 | 四组对照一次跑完；**开跑前自动复检 train/val 泄漏，异常则中止** |
| `evaluate.py` | 评估 | 在**独立测试集**上评估 + 每类 AP（报告唯一可引用的指标来源） |
| `exp_domain_shift.py` | **核心实验 1** | 距离域偏移：降采样模拟远距；找 mAP 崩塌点 vs GSD 理论门槛 |
| `exp_robustness.py` | 核心实验 2 | 4 种劣化 × 3 档（暗光/模糊/过曝/低对比），看是否触发拒答 |
| `exp_ablation.py` | 核心实验 3 | 标定误差 / 拒答风险-覆盖率 / 尺度线性度 三组消融 |
| `step6_quantize.py` | 模型轻量化 | FP32→ONNX→**INT8 静态量化**（赛道二 <10MB、<100ms；**本项目不走赛道二，仅存档**） |

> **02_code 全部 `.py` 清单以 `07_report/技术报告.md` §10.2 为准**（含本轮新增的
> `_assess_blindzone_impact.py` / `_probe_f1_semantics.py` / `_exp_ablation_boundary.py` /
> `_recycle_wheelhouse_dup.py` 等核查脚本）。

---

## 统一类别体系（7 类）

> 2026-09-20 由 6 类扩展为 7 类：主力数据源含 `MOSS`（苔藓附着）且有 560 个实例，
> 用户决定保留为独立类别，而非丢弃或并入 `efflorescence`。

| ID | 英文 | 中文 | 说明 |
|---|---|---|---|
| 0 | `crack` | 裂缝 | 线状缺陷，走骨架化测长/测宽 |
| 1 | `spalling` | 剥落缺失 | 块状，走面积/等效直径（含砖块缺失） |
| 2 | `efflorescence` | 泛碱渗水 | 块状（含 W_E 水渍、ALKALI 泛碱） |
| 3 | `exposed_rebar` | 露筋 | 线状（**V3 起有数据：1389 框 / 462 图，来自 HRCDS**） |
| 4 | `rust` | 锈迹 | 线状（**V3 起有数据：13685 框 / 1196 图，来自 Urban**） |
| 5 | `delamination` | 空鼓分层 | 块状（**V3 起有数据：694 框 / 518 图**；但可见光下仍只是间接迹象） |
| 6 | `moss` | 苔藓附着 | 生物附着，块状 ← 2026-09-20 新增 |

### ⚠️ 数据支撑的现实情况（V3：7 类全部有数据）

> **2026-09-21 起本条已更新**：V2 时期只有 4 类拿得到数据，V3 并入 HRCDS + Urban
> 后 **7 类全部有标注**。下表为 **V3 全量**（2848 图 / 18262 框）：

| 类别 | V3 实例数 | 状态 |
|---|---|---|
| `crack` 裂缝 | 469 | ✅ 有数据 |
| `spalling` 剥落缺失 | 313 | ✅ 有数据（**最少**） |
| `efflorescence` 泛碱渗水 | 1152 | ✅ 有数据 |
| `moss` 苔藓附着 | 560 | ✅ 有数据 |
| `exposed_rebar` 露筋 | 1389 | ✅ **V3 补入**（HRCDS） |
| `rust` 锈迹 | 13685 | ✅ **V3 补入**（Urban；**最多，占比过半**） |
| `delamination` 空鼓 | 694 | ✅ **V3 补入**（Urban） |

> ⚠️ **两条必须同时声明的限制**：
> ① **类别极度不平衡** —— `rust` **13685** 框 vs `spalling` **313** 框 ≈ **43 : 1**，
> 且**未做重采样**（保持数据原貌）⇒ **整体 mAP 会被 `rust` 抬高**，
> 引用总体指标时必须同时给逐类数字；
> ② **空鼓（`delamination`）虽已有标注，但可见光下本质仍是间接迹象** ——
> 脱开发生在内部界面，表面可能完全正常；专业手段是敲击法或红外热成像。
> 这是**技术边界**，不是数据问题。

> **苔藓的定级规则与其它类不同**（见 `grade.grade_moss`）：

已做全量调研（含 **RF100 全部 102 个数据集枚举**），确认**建筑外墙缺陷检测在公开
检测数据集里确实稀缺** —— 连 RF100 这样的权威 benchmark 都未覆盖。
详见 `.workbuddy/reference/补充数据源调研结论.md`。

> **当前模型实际只输出 4 类**，报告与界面须如实标注其余三类「暂未覆盖」。
> 这符合本项目「筛查工具而非鉴定工具」的定位。

> **苔藓的定级规则与其它类不同**（见 `grade.grade_moss`）：
> 苔藓**不作为结构缺陷定级，最高只到 `attention`**，且
> `building_risk_level()` 在推 C 级时**只统计结构类缺陷**。
> 理由：苔藓本身不削弱承载力，但它是「长期潮湿」的可靠指示物
> （会推动泛碱、冻融剥落、钢筋锈蚀）。若按通用块状规则走，
> 一块 15% 面积的苔藓会被判 `danger`，进而把整栋楼推到 C 级 ——
> 这是典型**误报**，会让筛查结果失去可信度。
> 非结构项单独汇总为 `n_moisture_hints`（潮湿线索）。
>
> ⚠️ **每新增一类必须同步三处**，否则会静默出错：
> ① `pipeline.py` 的 `colors` 调色板（漏了会回退成裂缝红）；
> ② `measure.py` 的线状类白名单（块状类走 else，属安全默认）；
> ③ `grade.py` 的 `grade_all` 判据路由与 `building_risk_level` 的结构性标记。

---

## 快速开始

### 环境

```
Python 3.13.15
ultralytics 8.4.138 / torch 2.11.0+cu128 / CUDA 12.8
OpenCV 5.0.0 / numpy 2.5.2
GPU: RTX 5070 Ti Laptop (12GB)
```

### 数据流水线

```powershell
cd "D:\pythonstudy 备份\创新题\外墙缺陷筛查\02_code"
$env:PYTHONIOENCODING="utf-8"
$PY = "D:\下载\python.exe"        # 本机真实环境（含 ultralytics / torch）

# 0) 下载（走 ModelScope 国内镜像，无需注册、无需 API key）
#    含下载 + Python 解压 + 逐条校验；绝不用 PowerShell Expand-Archive
& $PY step0b_download_modelscope.py --list    # 先看文件清单
& $PY step0b_download_modelscope.py           # 实际下载+解压+校验

# 1) 数据体检（报告落在 01_data/audit/）
& $PY step1_scan.py --root=..\01_data\raw\greybrick\src

# 2) 去重（必做，防泄漏）—— 默认复制，源目录不动
& $PY step2_dedup.py --root=..\01_data\raw\greybrick\src

# 3) 类别归一
& $PY step3_unify.py --in=..\01_data\dedup\images

# 4) 划分 + 防泄漏自检
& $PY step4_split.py --in=..\01_data\unified   # 交集不为 0 会直接报错退出

# 5) （可选）补充数据源：露筋。产物隔离存放，不自动并入主数据集
& $PY step5_fetch_rebar.py --list    # 先看清单
& $PY step5_fetch_rebar.py           # 下载 + 转 YOLO（label-studio 百分比坐标 -> 归一化 xywh）
```

> ⚠️ `step2_dedup.py` 的源目录会被「读」，不会被改（默认复制模式）。
> 若加 `--move` 则会把重复副本从源目录移走 —— **跑前务必备份源数据**。

### 训练（四组对照）

```powershell
& $PY train.py                          # 全部 4 组
& $PY train.py --runs=v11s640           # 只跑主力配置
& $PY train.py --runs=v8s640,v11s640    # v8 vs v11 版本对比
```

| 实验名 | 权重 | imgsz | 目的 |
|---|---|---|---|
| `v11s640` | yolo11s | 640 | **主力配置**（mAP50-95 / 体积 / 误检三项最优） |
| `v8s640` | yolov8s | 640 | **版本对照**（v8 vs v11，回答「为何选该版本」） |
| `v8n640` | yolov8n | 640 | 规格对照（赛道二 <10MB） |
| `v8s1024` | yolov8s | 1024 | 小目标实验（裂缝/露筋是细长小目标） |

> **主力为何是 `v11s640`（2026-09-20 切换，2026-09-21 按 V3 复核）**：赛题原文明确
> 「模型版本自选」并点名列出 `YOLOv5/v8/v11`，且答辩要求「说明为什么选这个 YOLO 版本」——
> 两组配置是**完全受控对照**（唯一变量是预训练权重，数据/超参/轮数相同）。
>
> ⚠️ **V3 扩容后必须说清这一点**：四档 mAP50 落在 **0.7143~0.7356**（极差 0.0213 < ≈0.03
> 判读阈值），**mAP50 上判不出高下，且 `v8s640` 的 0.7356 反而略高于主力的 0.7230**。
> 定 `v11s640` 的依据是**三项非精度优势**：
> ① mAP50-95 **最高**（0.44105 vs 0.43244）；② 体积最小（**18.33 vs 21.51 MB**）；
> ③ 假阳更少（**V2 口径** 109 vs 119，引用须注明口径）。
> **不得表述为「v11s 精度更优」** —— 那是 V2 小测试集（67 图 / 259 实例）上的结论。

> 训练入口会**自动复检 train/val/test 文件名交集**，任一不为 0 直接中止 ——
> 因为数据集是手工可改的目录，划分被污染后跑出来的 mAP 会虚高且报告看不出来。

### 评估（独立测试集）

```powershell
& $PY evaluate.py --weights=..\03_weights\v11s640_best.pt --name=v11s640
& $PY evaluate.py --all      # 评估 03_weights/ 下所有 *_best.pt
```

> **报告中只能引用这里的数字。** 训练过程打印的 mAP 是 val 集上的，
> 而 val 参与了早停与 `best.pt` 选择，属间接参与调参；测试集全程隔离，只评一次。
> 产出：`<run>_test_metrics.json` + `<run>_per_class.csv`（每类 P/R/AP50/AP50-95）。

### 推理（端到端）

```powershell
# 用标定物标定（A4 纸短边在图上占 420px）
python pipeline.py --image=wall.jpg --model=..\03_weights\best.pt --calib-object-px=420

# 整目录，用相机参数估 GSD（20m 外主摄）
python pipeline.py --dir=photos\ --distance=20

# 用砖缝周期自动标定
python pipeline.py --image=wall.jpg --calib-brick
```

> ⚠️ **Python 参数里的中文路径在 PowerShell 下会 mojibake**（见环境注意事项 17）。
> 一律把路径先赋给 PowerShell 变量再传：`& $PY pipeline.py "--model=$W"`。

### 实验（三组，全部在测试集上）

```powershell
& $PY exp_domain_shift.py --weights=$W --name=v11s640
& $PY exp_robustness.py   --weights=$W --name=v11s640
& $PY exp_ablation.py     --weights=$W --name=v11s640
```

产出落在 `04_results/ablation/`，含 JSON（可追溯）+ CSV（可直接贴报告）。

| 实验 | 回答的问题 | 关键产出 |
|---|---|---|
| `exp_domain_shift` | 远距拍摄下性能衰减多少？ | GSD–mAP 曲线 + **崩塌点 vs 理论门槛是否吻合** |
| `exp_robustness` | 暗光/模糊/过曝/低对比下还能用吗？ | 劣化–指标曲线 + 拒答触发率 |
| `exp_ablation` | 标定错了会怎样？拒答值不值？ | 风险–覆盖率权衡 + 尺度线性度 |

### 模型轻量化（赛道二）

```powershell
# 默认量化主力权重 03_weights/v11s640_best.pt
& $PY step6_quantize.py

# 指定权重 / 校准集大小 / 只量化不评估
& $PY step6_quantize.py --weights=$W --calib-n=150
& $PY step6_quantize.py --weights=$W --no-eval

# 批量量化所有已训练权重
& $PY step6_quantize.py --all
```

产出：`04_results/quant/<name>_int8.onnx` + `quant_report.json`。
> ⚠️ ORT 静态量化**不能在中文路径下跑** —— 脚本内部会自动 stage 到
> `%TEMP%\wall_defect_quant`（纯 ASCII）再复制回来。详见环境注意事项 16。

### Web 演示

```powershell
& $PY ..\06_deploy\app.py          # Gradio，浏览器打开 http://127.0.0.1:7860
```

---

## 已验证的精度（合成真值对标）

| 项目 | 测得 | 真值 | 误差 |
|---|---|---|---|
| 裂缝长度（骨架弧长） | 297.0 px | 297.0 px | **0%** |
| 裂缝宽度（距离变换） | 4.39 px | ~4 px | <10% |
| 砖缝周期标定 | 3.1250 mm/px | 3.1250 mm/px | **0%** |
| 数据划分跨集交集 | 9 项全为 0（**字节级**） | 0 | 无字节级重复 ✅ |

### 检测精度（独立测试集，**V3：285 图 / 1646 实例 / 7 类**）

> **这是报告唯一可引用的检测指标。** 训练时打印的 mAP 是 val 集上的，
> val 参与了早停与 `best.pt` 选择，属间接参与调参。
>
> ⚠️ **V2 与 V3 不可比**：V2 为 67 图 / 259 实例 / **4 类有标注**，
> V3 为 285 图 / 1646 实例 / **7 类有标注**。下表为 **V3**；V2 历史值见 `07_report/` 各文档的「口径更正」段。

**四档总览（V3 口径，同一测试集）**：

| 配置 | mAP50 | mAP50-95 | P | R | 体积 |
|---|---|---|---|---|---|
| **`v11s640`（主力）** | 0.7230 | **0.44105** | 0.771 | 0.669 | **18.33 MB** |
| `v8s640` | **0.7356** | 0.43244 | 0.730 | 0.704 | 21.51 MB |
| `v8n640` | 0.7172 | 0.4109 | 0.781 | 0.645 | 5.99 MB |
| `v8s1024` | 0.7143 | 0.4072 | 0.729 | 0.697 | 21.58 MB |

> **判读纪律（重要）**：四档极差仅 **0.0213 < ≈0.03 阈值** ⇒ mAP50 上**判不出高下**，
> 且 `v8s640` 的 0.7356 **高于**主力 0.7230。主力地位由 **mAP50-95 最高 + 体积最小 +
> 假阳更少（V2 口径）** 三项支撑 —— 完整论证见 `技术报告.md` §5.3.1。
> 「≈0.03 阈值」源自 V2 的 259 实例样本量，V3 扩到 1646 实例后仅作**经验参考**。

**`v11s640` 逐类（V3，7 类，源：`04_results/eval/v11s640_per_class.csv`）**：

| 类别 | P | R | AP50 | AP50-95 | 实例 / 图 |
|---|---|---|---|---|---|
| `crack` 裂缝 | 0.806 | 0.724 | 0.728 | 0.335 | 58 / 22 |
| `spalling` 剥落缺失 | 0.744 | 0.743 | 0.773 | 0.429 | 43 / 17 |
| `efflorescence` 泛碱渗水 | 0.858 | 0.716 | 0.841 | 0.626 | 116 / 36 |
| `exposed_rebar` 露筋 | 0.666 | 0.395 | 0.457 | 0.264 | 147 / 46 |
| `rust` 锈迹 | 0.869 | 0.655 | 0.739 | 0.552 | 1149 / 120 |
| `delamination` 空鼓分层 | 0.711 | 0.652 | 0.700 | 0.397 | 68 / 52 |
| `moss` 苔藓附着 | 0.741 | 0.800 | 0.822 | 0.486 | 65 / 10 |
| **总计** | **0.771** | **0.669** | **0.7230** | **0.44105** | **1646 / 285** |

> ⚠️ **两类最弱，必须与数字一起说**：`exposed_rebar`（AP50-95 仅 **0.264**，R 0.395）
> 与 `crack`（AP50-95 **0.335**）—— 均为**线状细长目标**，与 §9.1 的小目标/测量窗口
> 边界同源。另：`rust` 占测试实例 **69.8%**（1149 / 1646），**整体 mAP 被它抬高**，
> 引用总体指标时须同时给逐类数字。

> **误检实测（⚠️ V2 口径）**（conf=0.25 / IoU≥0.5，**V2 的 67 图**）：跨类误判仅 **2 例**，
> 且越界类（cls≥7）预测数为 **0** —— 检测头维度是 `4+nc`，类别空间封闭，
> **结构上不可能输出不属于这 7 类的目标**。主要误差来源是泛碱类的
> **背景假阳**（占假阳 58.7%）。**该计数 V3 未重跑同口径逐框匹配，引用须注明口径**；
> 但「类别空间封闭」是结构性质，**与数据集版本无关**。建议对泛碱类单独提高
> conf 阈值，而非换模型。

---

## 环境注意事项（踩过的坑）

1. **中文路径**：`cv2.imread`/`imwrite` 会静默失败 →
   统一用 `common.imread_u` / `imwrite_u`（`np.fromfile` + `cv2.imdecode`）。
   **目录已全部改为英文命名**以避免此类问题。

2. **网络/代理**：本机 HTTP 代理会拦 `github.com`（CONNECT 502）。
   Ultralytics 默认下载预训练权重会失败 → `train.py` 走 `resolve_weight()`
   优先取 `03_weights/` 本地权重。yolo11 系列已从 hf-mirror 下载归档。

3. **Windows + 中文路径下**：`workers=0`（多进程 dataloader 易卡死）。

4. **dataset yaml 的 `path` 必须写绝对路径**：Ultralytics 对 `path: .`
   的解析依赖当前工作目录，会解析到错误位置。

5. **`pathlib.Path` 不能挂属性**（`__slots__` 类），需要 side-table dict。

6. **`cv2.randn(dst, mean, std)` 是覆盖写入**，不是叠加噪声。

7. **数据集获取走 ModelScope 国内镜像**（Roboflow 登录接口在本机网络下不可用）。
   实测 `modelscope.cn` / `hf-mirror.com` / `gitee.com` 均可直连。
   脚本：`step0b_download_modelscope.py`。数据集**无需注册、无需 API key**。

8. **⚠️ 绝不要用 PowerShell 的 `Expand-Archive` 解压本项目的 zip**。
   实测在「中文路径 + 大量文件 + 深层嵌套」下它**静默失败** ——
   建好目录骨架但不写文件，且**不报任何错**。
   当时 `images.zip` 的 800 张只出来 677 张，`label`/`train`/`valid` 全空，
   直接导致后续统计全部失真。
   → **一律用 Python `zipfile` 解压，并逐条校验文件大小**（见 `step0b` 的
   `unzip_verified()`）。

9. **本机 git 是 GitHubDesktop 自带的精简版**，**缺 `remote-https` helper**，
   `git clone https://...` 会失败：
   ```
   git: 'remote-https' is not a git command.
   fatal: remote helper 'https' aborted session
   ```
   → 拉数据集不要用 git，走 HTTP 端点。

10. **扫描目录时 `_` 前缀会被跳过**（`SKIP_DIR_PREFIXES`，用于排除隔离区）。
    **不要给数据目录起 `_` 开头的名字**，否则会被当隔离区跳过。
    （已把 `_ms_greybrick` 改名为 `greybrick`。）

11. **数据阶段报告落在 `01_data/audit/`**，不放 `04_results/eval/`。
   `04_results/eval` 的语义是「模型评估指标」，混入数据报告会让产物归属混乱。

12. **ModelScope 搜索接口的正确调用方式**（猜错过两次）：
   ```
   GET https://www.modelscope.cn/api/v1/dolphin/datasets?Query=<词>&PageSize=12&PageNumber=1
   ```
   ⚠️ **参数必须在 URL，不能放 POST body** —— POST + JSON body 稳定返回 **404**。
   ⚠️ **中文检索词几乎全是 0 结果**，**必须用英文关键词**。
   搜索匹配的是**数据集名称**，不是描述全文。

13. **ModelScope 上有的数据集没有 zip 打包**（如 `jiange1236/StructuralCrackDataset`
   是 27 张散图 + 3 个标注文件）。
   → 下载器**必须按后缀分类下载**，只挑 `.zip` 会导致「图一张没下来」，
   直到转换阶段才报「图像目录不存在」。（`step5` 第一版已实测踩到。）

14. **label-studio 导出格式 ≠ YOLO**：坐标是**百分比 0-100**（不是归一化 0-1），
   且要**同时解析 `rectTool` 与 `polygonTool` 两个字段**
   （只解 rect 会漏掉多边形标注 —— `step5` 第一版漏了 4 个「裂缝」）。
   多边形取**外接矩形**（我们的管线面向 bbox，不是分割）。

15. **数据集名称可能严重误导**：`LibreYOLO/wall-damage` 的类别是
   `Minorrotation`/`Moderaterotation`/`Severerotation`（墙体**旋转程度**，非缺陷类型）。
   → **任何数据集在采用前必须实际读取 `data.yaml` 核对类别**，不能凭名字判断。

16. **⚠️ ORT 的 `quantize_static` 在中文路径下必定失败**（已最小复现定位）：
   ```
   onnx.load("<中文路径>/m.onnx")                -> ✅ 成功
   quantize_static("<中文路径>/m.onnx", ...)     -> ❌ FileNotFoundError(2)
   ```
   根因：`quant_pre_process` 在源文件旁写 `xxx-inferred.onnx`，
   ORT 内部经 C++ 层 round-trip 该路径时把中文按 GBK/UTF-8 错误编解码成乱码
   （实测报错路径 `D:/pythonstudy 澶囦唤/鍒涘妞/...-inferred.onnx`）。
   **与「输出文件名是否英文」无关 —— 整条路径任一段含中文都不行。**
   → **对策**：量化中间产物先 stage 到 `tempfile.gettempdir()`（纯 ASCII），
   完成后再把最终 INT8 模型复制回项目目录。实测同样一个 FP32 ONNX
   中文路径失败、ASCII 目录成功（11.70MB → 3.29MB）。见 `step6_quantize.py`。

17. **⚠️ PowerShell 传含中文的路径给 Python 会 mojibake**：
   实测 `sys.argv` 收到 `'D:\\pythonstudy 澶囦唤\\鍒涘妞...'`。
   排查结果是：Python 侧**处于 UTF-8 模式**（`sys.flags.utf8_mode=1`、
   `sys.getfilesystemencoding()='utf-8'`），但 `os.device_encoding(0)='cp936'`
   → **PowerShell 送出的是 GBK 字节，Python 按 UTF-8 解码**，于是乱码。
   → **对策**：命令行里**用 PowerShell 变量传参**（`--model=$var`），
   不要内联中文字面量。项目内部脚本因为路径都由 `common.py` 相对本文件定位，
   不受此影响；只有「手工在 PowerShell 里敲含中文的路径参数」才会踩到。

18. **`json.dumps` 遇 numpy 标量会报「bool 不可序列化」**：
   报错信息说的是 `bool`，实际类型是 `numpy.bool_`，肉眼极难定位。
   管线里的 `conf`、`area_ratio`、`feasible` 等大量来自 numpy。
   → 已在 `common.dump_json` 加 `_default` 兜底（覆盖 bool_/integer/floating/ndarray），
   **所有脚本一律走 `dump_json`，不要直接用 `json.dump`**。

19. **⚠️ ORT `quantize_static` 会随模型规模 OOM（v8n 过得去、v8s 过不去）**：
   ```
   onnxruntime::BFCArena::AllocateRawInternal
   Failed to allocate memory for requested buffer of size 983040
   ```
   报错发生在 **Conv 节点**，请求 buffer 只有约 1MB —— 「连 1MB 都分不出来」
   说明是 ORT 的 CPU arena 吃满后**连零头都分配不出**，而不是真缺物理内存。
   触发条件：**模型参数量 × 校准图数**。v8n（3.0M 参数）+ 150 张能过，
   v8s（11.1M 参数）+ 150 张必失败。
   → **对策（`step6_quantize.py` 已内置，双保险）**：
   (a) 量化前 `enable_cpu_mem_arena=False`、限制 `OMP_NUM_THREADS`、
       并在**导入 onnxruntime 之前**设好环境变量（ORT 的 C++ 层会缓存，之后再设无效）；
   (b) **OOM 时自动降校准集规模重试**：150 → 64 → 32 → 16，任一成功即止，
       并在报告里记录实际用了几张（`calib_used` 字段）。

20. **⚠️ 多行异构数据写 CSV 必须取字段并集**：
   `csv.DictWriter(fieldnames=rows[0].keys())` 以**第一行**定字段集。
   本项目鲁棒性实验的基线行缺少 `n_images`/`d_mAP50` 等字段，
   结果 **12 组评估全部成功、最后写 CSV 时崩溃、一个数字都没落盘**
   （报 `ValueError: dict contains fields not in fieldnames`）。
   → 必须遍历所有行取**字段并集**，并给 `DictWriter` 传 `restval=""`。
   **「实验成功但零产出」是最容易漏掉的一类失败**，收尾阶段务必确认产物已落盘。

---

## 交付边界（答辩应主动说明）

本系统交付的是**筛查工具，不是鉴定工具**。

- **筛查层**（地面拍摄可行）：找「哪栋楼、哪个立面有问题」。
  目标缺陷是剥落/脱落/空鼓/渗水/露筋/锈迹，尺寸在**厘米~米**量级。
  1cm 剥落在 5x 长焦 20m 外 ≈ 6.7 px，绰绰有余。
- **鉴定层**（须无人机贴近或搭架）：测裂缝宽度是否超 0.2/0.3/0.5mm。

筛查层恰是政策最缺的一环：南通要求 25 年以上住宅每 10 年体检、
预制板房每 5 年一次，存量太大，不可能每栋上无人机 → **分级巡检**逻辑成立。

达到 B/C 级或出现危险点判据时，系统会提示
「请专业机构进场鉴定」，而不是自己下结论。

### ⚠️ 测量精度与「不修」决策（2026-09-22 补充，答辩会被追问）

**合成真值实测精度**（`logs/_verify_out_measure/measure_accuracy_report.{txt,json}`，脚本 `02_code/_verify_measure_accuracy.py`）：

| 项 | 结果 |
|---|---|
| 裂缝长度（骨架弧长） | 误差 **0%**（297.0 px 对 297.0 px） |
| 裂缝宽度（距离变换） | 对 ≥ 3px 目标 **±1 px 内**；2px 目标系统性偏大 |
| 砖缝周期标定 | **0%**（3.1250 mm/px） |

**黑帽核测量窗口边界（已评估，决定「不修」）**：

`measure.segment_defect` 的黑帽核 `ks = max(7, ROI短边 // 12)`，**可靠测量区间 =
[3px, ks−1]**。这曾被列为待修缺陷，**2026-09-22 经两步实验后定案：不修是正确的**：

1. **影响面实测**（`logs/_assess_blindzone/impact_report.txt`，v11s640 前 80 图 / 111 线状实例）：
   测不到 **0 个**、盲区自检可疑 **0 个**；真实 `ks` **恒为 7**（crack/rust 全 7，
   `exposed_rebar` ∈[7,29]）⇒ 真实目标恰落在 **[3,6]px** ⇒ **已上报的毫米数字未被污染**。
2. **候选修法净效果为负**（`logs/_assess_blindzone/f1_semantics.txt`，合成暗带 56 组）：
   多尺度黑帽 {7,13,21} 取最大响应 ⇒ ±20% 内命中从 **37 掉到 32** ——
   修好盲区 4 个，却**打坏原本正确的 8 个**（根因：大核对细目标产生边缘伪响应拉偏 Otsu）。
   `ks` 封顶 31 则真实数据**改动 0 个**（等于没改）。

⇒ **改为「使用前提」声明**：本系统的测宽结论仅对**宽度 ≥ 3px 且 < `ks` px** 的线状目标成立；
已上报的全部毫米数字**落在该区间内**。完整论证见 `07_report/技术报告.md` §7.6.6 / §9.1。

### ⚠️ 标定（GSD）的必要性 —— 有实验支撑

`exp_ablation` 的标定消融在原实验里**一致率全为 1.0**（信号被远离限值的样本稀释）。
**2026-09-22 已补正**（`logs/_ablation_boundary/boundary_ablation.txt`）：
构造等效临界样本（使物理宽度恰为 **0.30 mm**、落入临界区 (0.20, 0.40] mm）后：

| 假设 | 分级翻转率（414 个实例） |
|---|---|
| GSD 翻倍（`wrong2x`） | **414 / 414 = 100%** |
| GSD 减半（`half`） | **414 / 414 = 100%** |

⇒ **标定的价值恰好等于「临界区样本的占比」**（远离限值 0% vs 临界区 100%）；
**减半（漏报方向）风险同高**。口径注：477 是实例行数，414 是去重键下唯一数
（63 个键重复），翻转判定需逐实例配对，故分母用 414。详见技术报告 §7.4A。
