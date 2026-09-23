
## 二十七、文档路径引用全面核对：又修 7 处「看着像但指不到」的错误（2026-09-23 18:5x）

### 1) 起因
修完演示脚本第三幕命令后，意识到「路径引用错误」可能不止一处，
于是写了可复用工具 **`logs/_check_doc_paths.py`**：
从 .md 里抽出所有 `<顶层目录>/...` 形式的路径引用，逐个核对存在性
（支持通配符展开、也试「相对 `02_code`」），输出 `OK / MISS` 清单 + 计数。

### 2) 5 份文档、108 个引用，跑出 7 处真实错误

| # | 文档 | 原写法 | 修正为 |
|---|---|---|---|
| 1 | 演示视频脚本 | 第三幕命令读 `04_results/quant/quant_report.json` | 改读 `logs/_quant_rescue/rescue_results.json`（原文件是失败记录，命令会 KeyError） |
| 2 | 演示视频脚本 | `04_results/eval/robustness_v11s640_best.json` | `robustness_v11s640.json`（**无 `_best` 后缀**） |
| 3 | 演示视频脚本 | 素材清单里的量化报告指向失败记录 | 同上改为抢救试验产物 + 加「勿用」警告 |
| 4 | 根 README | `logs/_measure_accuracy/` | `logs/_verify_out_measure/measure_accuracy_report.{txt,json}` |
| 5 | 技术报告 | `04_results/eval/robustness_v11s640_best.csv` | `robustness_v11s640.csv` |
| 6 | 技术报告 | `04_results/ablation/ablation_A/B/C_*.json` | `ablation_[ABC]_*.json`（**原写法不是标准通配符，glob 不到**） |
| 7 | 技术报告 | `.../images/test/sc_1a7e86db-9.png` | `.../images/test/rebar_structural__sc_1a7e86db-9.png`（**少了前缀**） |

**第 6 处的额外价值**：改成 `[ABC]` 后实测 **glob 命中 6 个文件**；
原来的 `A/B/C` 写法在中文里像「A 或 B 或 C」，但**放进路径就读不通**、也无法真的用。

### 3) ★ 判读 MISS 时必须区分三类（否则会误改）
1. **真实错误** → 修（上表 7 处）；
2. **文中故意举的反例** —— 技术报告有一句「若路径写错会解析到 `02_code/images/val`」，
   那是**说明性文字**，**不能改**（改了这句话就不成立了）；
3. **抽取器伪影** —— `a.{txt,json}` 被正则截成 `a`、`ablation_[ABC]_*.json` 被截成
   `ablation_` ⇒ 这是**工具的问题，不是文档的问题**，**不要为了迎合脚本去改文档**。
> ⇒ 最终剩余 4 处 MISS **全部属于后两类 + 1 处「实验进行中待生成」**（`v11s640_clsbal_test_metrics.json`）。

### 4) 最终核对结果
| 文档 | 引用数 | 存在 | 剩余 MISS 性质 |
|---|---|---|---|
| 根 README | 17 | 16 | 1（伪影） |
| 07_report/README | 8 | 8 | 0 |
| 技术报告 | 71 | 67 | 4（1 反例 + 2 伪影 + 1 待生成） |
| 答辩材料 | 5 | 5 | 0 |
| 演示视频脚本 | 11 | 11 | 0 |

**真实错误已清零。** 所有修改均保持原行尾（根 README/07_report 为 CRLF）。

### 5) 工具与技能
- 新增可复用工具：**`logs/_check_doc_paths.py`**（用法：
  `python logs/_check_doc_paths.py "文档1.md" "文档2.md"`）。
- 技能 `deterministic-patch-and-verification` §3.1b 补入：内嵌命令必须原样实跑、
  核「内容结构匹配」而非只看路径存在、注意「产物被故意分流」的情形，
  以及「判读 MISS 的三类区分」。§3.3 已记录全库×全载体扫码纪律。

### 6) 训练现状（18:51）
- 已完成 **48/100 轮**，内存 **3.12 GB 可用 / 80%**。
- 预计 20:30 前后跑满；automation（21:00）待命收口。
