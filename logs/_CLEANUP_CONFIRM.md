# 空间清理确认单（待用户逐项拍板）

> 生成时间：2026-09-21
> 盘点脚本：`logs/_cleanup_audit.py` → `logs/_CLEANUP_AUDIT.txt`
> 明细来源：`logs/_trash_here/`（45 项）、`logs/_famerL_tmp/`（2 文件 + 1 空目录）
>
> ## ★ 状态声明
> **本单所列内容一项都未删除、未移动、未改名。**
> 全部为只读盘点。等你逐项确认后才动手。

---

## 一、总览

| # | 候选 | 占用 | 性质 | 建议 |
|---|---|---:|---|---|
| A | `logs/_trash_here/` | **4.573 GB** | 三个数据集的**下载段文件**（已全部合并完成） | ✅ 可删 |
| B | `logs/_famerL_tmp/val80.zip.copy.zip` | **889.9 MB** | 同一文件的**重复副本** | ✅ 可删 |
| C | `logs/_famerL_tmp/val80.zip` | **889.9 MB** | famerL 原始下载件（唯一一份） | ⚠️ 建议留 |
| D | 96 个空目录 | 0 B | 见 §四 | ✅ 可删（无害） |
| E | `02_code/runs/` | 33.2 MB | Ultralytics 运行目录 | ⚠️ 见 §五 |
| F | `04_results/eval/v8s640_final_test_metrics.json` | 2.3 KB | 与 `v8s640_test_metrics.json` **指标逐字相同** | ✅ 可归档 |
| **G** | **`logs/_bfdd_extract/`** | **538.7 MB** | BFDD 解压出的分析中间产物（4,192 文件） | ✅ 可删（可复现） |
| | **合计可回收（A+B+D+F+G）** | **≈ 6.00 GB** | | |

---

## 二、A 项明细：`logs/_trash_here/`（4.573 GB / 45 项）

**性质**：三个公开数据集在**分段下载 → 合并**过程中产生的段文件，改名加了时间戳后缀后暂存于此。

| 数据集 | 段文件 | 合并后成品（现状） |
|---|---|---|
| **BFDD** | 旧不完整件 `372,361,249` + `seg0 276,658,375` + `seg1 276,658,376` | `BFDD_dataset.tar.gz` **553,316,751 B** ✅ 与官方一致，可正常打开（4,198 成员） |
| **MBDD2025** | `seg0`–`seg6` 各 `320,159,847` + `seg7 290,398,413` | `mbdd2025-building-defects.zip` **2,531,517,342 B** ✅ `testzip=None` 零损坏 |
| **Urban** | `seg0`–`seg6` 各 `146,839,508` + `seg7.sub0/1/2` | `urban-infrastructure-anomalies.zip` **1,162,572,641 B** ✅ 34,632 条目，`testzip=None` |
| 另有 | 17 个 `.log`（多为 0–106 B） | 下载日志，无内容价值 |

**删除的前提条件（已逐条核实）**：

| 前提 | 核实结果 |
|---|---|
| 三个成品文件都存在且大小正确 | ✅ 见上表 |
| 成品能正常打开、无损坏 | ✅ BFDD 可 `tarfile` 打开；另两个 `testzip=None` |
| 段文件可由成品重建 | ❌ **不能** —— 段文件是下载原料，删后若成品损坏需**重新下载** |
| 是否还有别的程序依赖段文件 | ✅ 无（合并已完成，`_bfdd_finish.py` 等均已结束） |

> ⚠️ **唯一风险**：删段文件后，若日后发现某个成品损坏，需重新走一遍下载流程
> （urban / mbdd 走 Kaggle 签名 URL，BFDD 走 Mendeley）。三个源的链接都已记在
> 对应判读报告的「留痕」节，可复现。

---

## 三、B / C 项明细：`logs/_famerL_tmp/`（1.658 GB）

| 文件 | 大小 | 说明 | 建议 |
|---|---:|---|---|
| `val80.zip` | 889,900,983 | famerL 原始下载件，SHA256 `731f431b…` 已校验 | ⚠️ **建议保留** |
| `val80.zip.copy.zip` | 889,900,983 | **同名内容的第二份**（`.copy` 后缀） | ✅ 删除，省 889.9 MB |
| `ex/` | 空目录 | 解压中间目录 | ✅ 可删 |

> **为什么建议留 `val80.zip`**：该数据集来自 HuggingFace 上**匿名账号 `famerL`、
> 未声明许可、下载量仅 9 次**（见 `_FAMERL_VERDICT.md` §四）。
> 这类来源**随时可能被作者删除**，一旦删除就无法再获取。
> 它的 3 个可用类（crack / spalling / efflorescence）虽边际价值有限，
> 但保留原始件成本仅 890 MB，**建议留作证据**。

---

## 四、D 项明细：96 个空目录（0 B）

空目录不占空间，但会让目录树显得混乱。分布：

| 位置 | 数量 | 成因 |
|---|---:|---|
| `_archive_20260920/housekeeping_20260920/runs_ultralytics_val_plots/` | 44 | 已归档的旧 val 可视化目录 |
| `02_code/runs/detect/val-10 … val-*` | 38 | Ultralytics 反复跑 val 留下的空编号目录 |
| `01_data/raw/_archived_gitempty_20260920/` | 8 | ModelScope 下载残留的空 `.git` 骨架 |
| `01_data/raw/greybrick/src/{label,train,valid}/{images,labels}` | 5 | 曾 `--move` 搬空 |
| `logs/_famerL_tmp/ex/` | 1 | 解压中间目录 |

> 删除空目录**无数据风险**。但注意：`02_code/runs/detect/val-*` 若 Ultralytics
> 再次运行会重新创建 —— 属"可再生"目录。

---

## 五、E 项明细：`02_code/runs/`（72 文件 / 33.2 MB）

`runs/detect/` 下有历史 val 运行产物（含 38 个空目录 + 若干 `args.yaml`、
`predictions.json` 等）。

⚠️ **不建议整目录删除**，理由：技术报告附录 A 可能引用了其中的路径。
**建议**：只删其中的**空目录**（见 §四），保留有内容的运行记录。

---

## 六、F 项明细：`v8s640_final_test_metrics.json`（2.3 KB）

**这是本轮新查实的冗余**。逐键对比（`logs/_CLEANUP_AUDIT.txt` §四）：

| 键 | 结果 |
|---|---|
| `overall`（P / R / mAP50 / mAP50-95 / fitness） | **逐字相同** |
| `per_class`（7 类逐类 AP、instances、images） | **逐字相同** |
| `split` / `imgsz` / `weights_mb` | 相同 |
| `run` | `v8s640` vs `v8s640_final` |
| `weights` | `03_weights\v8s640_best.pt` vs `04_results\train\v8s640\weights\best.pt` |

⇒ 两个文件**只有「运行名」和「权重路径」两个元数据字段不同，指标完全一致**。
说明它们是同一次评估的两次落盘（或同一权重在两个位置的副本）。

**建议**：把 `v8s640_final_test_metrics.json` 归入 `_archive_20260920/`，
**不删除**（保留可追溯性），并在 `04_results/README` 注明
「`v8s640_test_metrics.json` 为唯一口径来源」。

---

## 六之二、G 项明细：`logs/_bfdd_extract/`（538.7 MB / 4,192 文件）

**本轮新增**。为做 BFDD 的确定性核查（数值↔颜色映射、编号空间比对、红外判别），
把 tar.gz 中需要的成员解压到了这里。

| 子项 | 内容 |
|---|---|
| `Label/` | 838 张灰度掩膜 |
| `Label_color/` | 838 张彩色掩膜 |
| `Label_backup_7classes_20260125/` | 838 张早期 6 类版掩膜 |
| `RGB/` | 838 张可见光 |
| `IR/` | 838 张红外 |
| `train.txt` / `test.txt` | 官方划分 |

**可删性**：✅ **完全可复现** —— 由 `logs/_bfdd_stats3.py` 与 `logs/_bfdd_ir_check.py`
自动解压（脚本幂等，重跑约 80 秒）。删除**不影响**任何结论与已落盘的分析产物
（`logs/_BFDD_STATS3.txt`、`logs/_BFDD_IR.txt`、`logs/_bfdd_mapcls/`、`logs/_BFDD_VERDICT.md`）。

> ⚠️ 注意：**不要连 tar.gz 一起删** —— 归档文件在
> `01_data/raw/_public_datasets/bfdd/BFDD_dataset.tar.gz`（553,316,751 B），
> 它与本目录是两回事，本目录只是解压出来的副本。

---

## 七、★ 需要你拍板的事项

- [ ] **A**：`logs/_trash_here/` 4.573 GB 段文件 —— 删？（我建议：**删**）
- [ ] **B**：`val80.zip.copy.zip` 889.9 MB 副本 —— 删？（我建议：**删**）
- [ ] **C**：`val80.zip` 889.9 MB 原始件 —— 删？（我建议：**留**）
- [ ] **D**：96 个空目录 —— 清？（我建议：**清**，零风险）
- [ ] **E**：`02_code/runs/` —— 只删空目录 / 全删 / 不动？（我建议：**只删空目录**）
- [ ] **F**：`v8s640_final_test_metrics.json` —— 归档 / 删除 / 不动？（我建议：**归档**）
- [ ] **G**：`logs/_bfdd_extract/` 538.7 MB 解压中间产物 —— 删？（我建议：**删**，脚本可复现）

> 关于删除方式：本项目所在磁盘的删除操作会走**系统回收站**（可还原），
> 但仍建议先确认上表再执行。**在你明确回复之前，我不会执行任何一项。**

---

## 八、留痕

| 文件 | 说明 |
|---|---|
| `logs/_cleanup_audit.py` | ★ 只读盘点脚本 |
| `logs/_CLEANUP_AUDIT.txt` | 盘点输出（含 `_trash_here` 与 eval 冗余逐键对比） |
| `logs/_bfdd_stats3.py` / `logs/_bfdd_ir_check.py` | ★ G 项（`_bfdd_extract/`）的生成脚本，可复现 |
| `logs/_CLEANUP_CONFIRM.md` | 本确认单 |
