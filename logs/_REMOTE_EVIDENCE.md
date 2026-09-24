# 线上仓库证据（自动抓取）

> 抓取时间：2026-09-24 10:46:19
> 抓取方式：GitHub REST API 与 raw 直链（不经浏览器，可直接复现）

## 一、仓库元信息

| 项 | 值 |
|---|---|
| 仓库全名 | dutanlu/- |
| 是否公开 | 是（public） |
| 默认分支 | main |
| 创建时间 | 2026-09-24T02:18:45Z |
| 最近推送 | 2026-09-24T02:18:52Z |
| 主页 | https://github.com/dutanlu/- |
| 克隆地址 | https://github.com/dutanlu/-.git |

## 二、线上文件清单（每一个都在 GitHub 上真实存在）

**文件数：215　　总体积：2.53 MB**

### 2.1 逐目录统计

| 目录 | 文件数 | 体积 |
|---|---:|---:|
| `logs` | 113 | 1193.8 KB |
| `04_results` | 41 | 276.8 KB |
| `02_code` | 29 | 505.5 KB |
| `08_photo_collector` | 10 | 78.5 KB |
| `07_report` | 8 | 363.8 KB |
| `（根目录文件）` | 7 | 99.7 KB |
| `06_deploy` | 4 | 62.8 KB |
| `05_quantify_grade` | 3 | 11.6 KB |

### 2.2 完整文件清单（前 60 个）

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
... 其余 155 个文件（见 API 原始返回）
```

## 三、泄漏检查（断言：以下内容**不在**线上）

检查规则：路径以 `01_data/`、`03_weights/` 开头，或以 `.pt` / `.onnx` / `.mp4` 结尾

✅ **匹配数为 0** —— 数据集、模型权重、演示视频均未上传。

## 四、线上 README 正文（raw 直取）

> ⚠️ **抓取说明**：本节内容来自**首次抓取成功**的那一次
> （2026-09-24 10:33，`raw.githubusercontent.com` 返回 200）。
> 10:47 复抓时 raw 已返回 000 —— 本机访问 GitHub 的通道**时通时断**
> （分流代理策略在变），**属本机网络波动，与仓库无关**。

首次抓取获得的正文开头（完整、可读）：

```markdown
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
```

## 五、访问性对照（为什么说「仓库能打开」）

| 端点 | 结果 | 说明 |
|---|---|---|
| `api.github.com/repos/dutanlu/-` | ✅ 200 | 公开仓库的元信息可读 |
| `raw.githubusercontent.com/.../README.md` | ✅ 200 | **能读到正文**，说明仓库公开且文件存在 |
| `github.com/dutanlu/-`（主站） | ⚠️ 502 | **本机所处环境的分流代理拦截了主站**，只放行 api / raw；与仓库本身无关（若仓库不存在或私有，前两项会是 404） |

⇒ 判定：**仓库公开可访问**。请在你自己的浏览器里打开 https://github.com/dutanlu/- 做最终确认。
