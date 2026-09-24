# 远程仓库快照（实测抓取）

> 抓取时间：2026-09-24 20:57:58（北京时间）
> 抓取方式：GitHub REST API（`api.github.com`，本机可直连）
> 说明：本文件是**当时远程状态的原始记录**，不是复述。

---

## 〇、如何自行复核（不依赖本文件的数字）

本文件是**某一时刻的快照**；仓库内容若在此之后有更新，下方数字会随之变化。
因此这里给出**三条可独立执行的复核命令** —— 评审可直接运行，得到的永远是当下真值：

```bash
# 1. 仓库是否公开可访问、许可证是什么
curl -s https://api.github.com/repos/dutanlu/wall-defect-screening

# 2. 当前 main 的提交号与时间
curl -s https://api.github.com/repos/dutanlu/wall-defect-screening/commits/main

# 3. 全量文件列表（可自行统计文件数与体积）
curl -s "https://api.github.com/repos/dutanlu/wall-defect-screening/git/trees/main?recursive=1"
```

或直接用 git 读取远程 HEAD（不需要克隆）：

```bash
git ls-remote https://github.com/dutanlu/wall-defect-screening.git HEAD
```

---

## 一、仓库元信息

| 项目 | 值 |
|---|---|
| 仓库全名 | `dutanlu/wall-defect-screening` |
| 主页 | https://github.com/dutanlu/wall-defect-screening |
| 克隆地址 | `https://github.com/dutanlu/wall-defect-screening.git` |
| 可见性 | **public（公开）** |
| 默认分支 | `main` |
| 许可证 | **MIT**（MIT License）|
| 创建时间 | 2026-09-24T02:18:45Z |
| 最后推送 | 2026-09-24T12:56:32Z |
| Stars / Forks | 0 / 0 |

> 注：旧地址 `github.com/dutanlu/-` 由 GitHub 自动 **301 重定向**到新地址。

---

## 二、提交与标签

| 项目 | 值 |
|---|---|
| main HEAD | `3aa5db9e887a6195a2c65e04c492d8a24c14b65b` |
| 提交信息 | 开源收尾完成报告：改名 + 许可证修复 + tag 全部完成 |
| 提交时间 | 2026-09-24T12:56:20Z |
| tag 名 | `v1.0-submission` |
| tag 类型 | 附注标签（annotated）|
| tag 指向 | `3aa5db9e887a6195a2c65e04c492d8a24c14b65b` |

---

## 三、内容清单

- **文件数：219**
- **体积：2.57 MB**

### 顶层结构

```
logs                            115 个文件     1218.2 KB
02_code                          29 个文件      505.5 KB
07_report                         9 个文件      378.0 KB
04_results                       41 个文件      276.8 KB
(根文件)                             8 个文件      101.6 KB
08_photo_collector               10 个文件       78.5 KB
06_deploy                         4 个文件       62.8 KB
05_quantify_grade                 3 个文件       11.6 KB
```

### 完整文件清单

```
     3144  .gitignore
    17704  02_code/_assess_blindzone_impact.py
    10429  02_code/_exp_ablation_boundary.py
     5451  02_code/_extract_grading_rules.py
     5556  02_code/_probe_f1_semantics.py
     6919  02_code/_recycle_wheelhouse_dup.py
    30032  02_code/_verify_measure_accuracy.py
    38746  02_code/_verify_rectify.py
    17625  02_code/bench_pipeline.py
    15465  02_code/common.py
    15289  02_code/evaluate.py
    16507  02_code/exp_ablation.py
    12435  02_code/exp_domain_shift.py
    12203  02_code/exp_robustness.py
    13691  02_code/grade.py
    20844  02_code/gsd.py
    17489  02_code/measure.py
    15694  02_code/pipeline.py
    76700  02_code/rectify.py
      410  02_code/roboflow_yaml/wall_defects.yaml
     9570  02_code/run_experiments.py
     5565  02_code/step0_download.py
     8728  02_code/step0b_download_modelscope.py
     9574  02_code/step1_scan.py
    12568  02_code/step2_dedup.py
    18520  02_code/step3_unify.py
    11027  02_code/step4_split.py
    16932  02_code/step5_fetch_rebar.py
    55271  02_code/step6_quantize.py
    20731  02_code/train.py
     1659  04_results/eval/aug_ab_comparison.json
     2638  04_results/eval/class_unify_report.json
     3240  04_results/eval/clsbal_ab_comparison.json
      355  04_results/eval/collapse_detector_calibration.json
     5973  04_results/eval/collapse_detector_multiscale.json
   195081  04_results/eval/dedup_report.json
     3820  04_results/eval/image_quality_gate.json
     1545  04_results/eval/pipeline_bench.json
     1554  04_results/eval/pipeline_bench_11_8MP.json
    10928  04_results/eval/pipeline_bench_448.json
    11427  04_results/eval/risk_coverage_curve.json
     1007  04_results/eval/robustness_v11s640.csv
     3781  04_results/eval/robustness_v11s640.json
     1007  04_results/eval/robustness_v8s640.csv
     3780  04_results/eval/robustness_v8s640.json
     1829  04_results/eval/split_report.json
     2704  04_results/eval/test_eval_summary.json
     2300  04_results/eval/train_summary.json
      470  04_results/eval/v11s640_aug_per_class.csv
     2040  04_results/eval/v11s640_aug_test_metrics.json
      470  04_results/eval/v11s640_clsbal_per_class.csv
     2031  04_results/eval/v11s640_clsbal_test_metrics.json
      470  04_results/eval/v11s640_onnx_v3_per_class.csv
     2027  04_results/eval/v11s640_onnx_v3_test_metrics.json
      470  04_results/eval/v11s640_per_class.csv
      470  04_results/eval/v11s640_pt_cpu_per_class.csv
     2020  04_results/eval/v11s640_pt_cpu_test_metrics.json
     2013  04_results/eval/v11s640_test_metrics.json
      470  04_results/eval/v11s640_v3_fp16_per_class.csv
     2033  04_results/eval/v11s640_v3_fp16_test_metrics.json
      470  04_results/eval/v11s640_v3_int8A_per_class.csv
     2041  04_results/eval/v11s640_v3_int8A_test_metrics.json
      470  04_results/eval/v11s640_v3_int8B_per_class.csv
     2043  04_results/eval/v11s640_v3_int8B_test_metrics.json
      470  04_results/eval/v8n640_per_class.csv
     2013  04_results/eval/v8n640_test_metrics.json
      470  04_results/eval/v8s1024_per_class.csv
     2016  04_results/eval/v8s1024_test_metrics.json
      470  04_results/eval/v8s640_per_class.csv
     2012  04_results/eval/v8s640_test_metrics.json
     1319  04_results/quant/quant_report.json
     6525  05_quantify_grade/README.md
     1095  05_quantify_grade/_extract_report.txt
     4241  05_quantify_grade/grading_rules.json
     4910  06_deploy/README.md
    56285  06_deploy/app.py
     1080  06_deploy/run_app.bat
     1981  06_deploy/启动说明.txt
    19596  07_report/README.md
     7482  07_report/先例与创造性核查_20260923.md
     9380  07_report/创新方向评估_20260923.md
    69659  07_report/外墙缺陷智能筛查_答辩PPT.pptx
     5615  07_report/开源发布记录.md
   208581  07_report/技术报告.md
    17989  07_report/演示视频脚本.md
    46947  07_report/答辩材料.md
     1808  07_report/项目介绍书.md
      691  08_photo_collector/_patch_promo_storage.txt
      218  08_photo_collector/_patch_promo_storage2.txt
      278  08_photo_collector/_patch_site_copy.txt
      349  08_photo_collector/_patch_storage_private.txt
    13340  08_photo_collector/site/app.js
     4537  08_photo_collector/site/cloud.js
      990  08_photo_collector/site/config.js
    38245  08_photo_collector/site/index.html
    10064  08_photo_collector/发布话术.md
    11721  08_photo_collector/推广文案.md
     1094  LICENSE
     1814  LICENSE-NOTES.md
    37069  README.md
     7833  logs/V3数据补齐核查报告_20260921.md
     3557  logs/_ALT_SOURCE_PLAN.md
     3024  logs/_AUG_AB_COMPARISON.md
      952  logs/_BENCH_BATCH.md
     1344  logs/_BENCH_DECOMPOSE.md
     1280  logs/_BENCH_GPUUTIL.md
      910  logs/_BENCH_OVERHEAD.md
     1731  logs/_BENCH_SPEEDUP.md
    15139  logs/_BFDD_VERDICT.md
     6780  logs/_BFDD_论文检索_20260921.md
     2511  logs/_CHECK.md
     8257  logs/_CLEANUP_CONFIRM.md
     2264  logs/_CLEANUP_EXECUTED.md
     6104  logs/_CLSBAL_AB_COMPARISON.md
     3762  logs/_CMP.md
     3721  logs/_COLLAPSE_DETECTOR.md
     5401  logs/_COLLAPSE_DETECTOR_V2.md
    10954  logs/_DATASET_HUNT_RESULT.md
     5635  logs/_DATASET_HUNT_ROUND2.md
     3255  logs/_DATA_PROGRESS_CHECK.md
     3749  logs/_DIAGNOSIS_20260920.md
     8530  logs/_DRAFT_赛题四问答法.md
      428  logs/_EVAL_AFTER_TRAIN.md
     9906  logs/_FAMERL_VERDICT.md
     3927  logs/_FILL_MAP.md
     7077  logs/_FILL_SCAN.md
    11617  logs/_FILL_SCAN2.md
     2953  logs/_FP16_DETECT_LEVEL.md
     1894  logs/_FP16_ONNX_REPORT.md
     2513  logs/_FP16_ONNX_REPORT_v11s640_best_v3.md
      704  logs/_GRADCAM_REPORT.md
     9479  logs/_HF_MIRROR_ROUND3.md
      670  logs/_HRCDS_INDEX_FIX.md
     8444  logs/_HRCDS_VERDICT.md
     5949  logs/_IMAGE_QUALITY_GATE.md
     9538  logs/_KAGGLE_BREAKTHROUGH.md
     3515  logs/_LINEAGE_PROBE.md
    11687  logs/_LINT_MARKER.md
     9846  logs/_MBDD2025_VERDICT.md
     6550  logs/_NAME_PROBE.md
    30362  logs/_NIGHT_SHIFT_20260920.md
     9311  logs/_NIGHT_SHIFT_结果报告_20260923.md
     6685  logs/_OFFLINE_INSTALL.md
     4415  logs/_OPENDATALAB_ACCESS.md
     3647  logs/_PAUSED_类平衡训练.md
    14607  logs/_PLAN_提分与技术方案_20260923.md
     5990  logs/_PLAN_量化再分析_20260923.md
     3798  logs/_POST_TRAIN.md
     1365  logs/_POST_TRAIN_SUMMARY.md
     1838  logs/_PROBE_FILES.md
     1072  logs/_PROBE_NOW.md
     6387  logs/_PROBE_NOW2.md
     1206  logs/_PROBE_NOW3.md
      239  logs/_PROBE_NOW4.md
     3250  logs/_PT_SCAN.md
     3712  logs/_PT_WIDE.md
     4174  logs/_QUANT_RESCUE_CONCLUSION.md
    18879  logs/_REMOTE_EVIDENCE.md
     8760  logs/_RESTORE_GRADIO.md
     9088  logs/_RISK_COVERAGE.md
      871  logs/_STATUS_NOW.md
     1898  logs/_STEP4_V3.md
     5519  logs/_STORAGE_SCOPE_FINDING.md
     7449  logs/_SUMMARY_train_and_gap.md
     6496  logs/_THIN_SPEEDUP_CONCLUSION.md
     6492  logs/_UNPACK.md
     9303  logs/_URBAN_VERDICT.md
     2678  logs/_V11_ADVANTAGE_SOURCE.md
     5824  logs/_V2_TEST_RESULTS_ANALYSIS.md
    24868  logs/_V3BF_EXTRACT.md
     1649  logs/_V3_BUILD.md
     8195  logs/_V3_FACTS_PROBE.md
     1699  logs/_V3_PROBE2.md
      875  logs/_V3_RAWCOUNT.md
     3351  logs/_V3_SPLIT_LINEAGE.md
   641705  logs/_V3_SPLIT_PROBE.md
     3574  logs/_V3_TEST_CLASS_STATS.md
     5310  logs/_V3_TRAIN_CURVE_FACTS.md
     1560  logs/_V3_VALIDATE.md
     1207  logs/_V3_VS_V2_BYHASH.md
     2417  logs/_V3_VS_V2_SPLIT.md
     2518  logs/_V711_RECALC.md
     8882  logs/_VERIFY_V3_EVAL.md
     5822  logs/_WHEEL_RESTORE.md
     6384  logs/_WHY_V11_BETTER.md
     3559  logs/_day19.md
     2927  logs/_day20.md
     3162  logs/_day21.md
     3996  logs/_day22.md
     3907  logs/_day23.md
     3627  logs/_day24.md
     3948  logs/_day25.md
     5162  logs/_day26.md
     3445  logs/_day27.md
     4161  logs/_day28.md
     2319  logs/_day29.md
     1102  logs/_err_probe.md
     1143  logs/_famerL_analysis.md
     4151  logs/_famerL_preview.md
     1478  logs/_famerL_rebar.md
     2285  logs/_famerL_valcheck.md
     7499  logs/_new_5_3.md
    21457  logs/_quant_mixed/_graph_probe.txt
     3610  logs/_quant_mixed/mixed_log.txt
     3331  logs/_quant_mixed/mixed_log_calib32.txt
     3268  logs/_quant_mixed/mixed_log_calib96.txt
     1648  logs/_quant_mixed/mixed_results.json
     1648  logs/_quant_mixed/mixed_results_calib32.json
     1412  logs/_quant_mixed/mixed_results_calib96.json
     1777  logs/_roi_preview.md
     7502  logs/_sec78_draft.md
     2695  logs/_wheelhouse_cleanup_20260922.md
     3790  logs/_zip_verify.md
     6074  logs/_开源收尾完成报告.md
    12649  logs/环境事故报告_20260921.md
     1515  requirements.txt
    17347  审查报告_交付物完整性与缺口.md
    12762  审查报告_赛题符合性大检查_20260922.md
    29263  自检报告_20260920.md
```

---

## 四、泄漏断言（实测，不是承诺）

对全量文件列表做正则筛选：

```python
re.search(r'\.(pt|onnx|mp4|avi|mov|pth|engine|weights)$', path, re.I)
```

- **匹配数：0** ⇒ ✅ **无任何权重或视频文件入库**

数据与权重未上传是**有意为之**（第三方数据集许可不允许我们代为再分发），
见 `LICENSE-NOTES.md` 与 README「数据与权重不在本仓库中」。

---

## 五、README 正文（实抓）

````markdown
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
|
````