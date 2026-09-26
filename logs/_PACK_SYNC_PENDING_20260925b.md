# 交付包待同步增补（2026-09-25 14:20 记）

> 本文件是 `MEMORY.md` 里「交付包待同步清单」的增补。训练（`v11s896_r20`）
> 结束后执行 `_copy_delivery_pack.py` 时，**除原有清单外，本次还多出以下项** ——
> 它们由 2026-09-25 13:30-14:20 的「弃权机制修复」与「批量入口新增」产生。

## 一、本次新增/修改的源文件（必须进包）

### 修改（内容不同）
| 源文件 | 变更 |
|---|---|
| `02_code/grade.py` | **新增** `classify_unjudgeable` / `apply_abstention`（共享弃权实现） |
| `02_code/pipeline.py` | `run_one` 接入共享弃权；imports 增补 |

### 新增（源有包无）
| 源文件 | 说明 |
|---|---|
| `02_code/batch_screen.py` | 批量/巡检入口（CLI 级） |

### 文档（内容不同）
| 源文件 | 变更 |
|---|---|
| `07_report/技术报告.md` | 新增 **§11.9**（弃权缺陷全过程）；**§2.3** 补「实现范围」说明；<br>★ **§7.13**（14:20 新增，分辨率两侧证伪 + 露筋重归因，含 7.13.1~7.13.7） |
| `07_report/README.md` | 加 `batch_screen.py` 用法 + 能力边界 |
| `07_report/弃权机制缺陷修复说明_20260925.md` | 弃权缺陷独立说明（已在既有清单） |

### 本轮新增的「证据类」产物（**是否进包按既有 logs/ 口径**）
| 源文件 | 说明 |
|---|---|
| `logs/_fix_imgsz_ab_confound.py` | 补回 confound/verdict（幂等） |
| `logs/_update_experiments_summary.py` | 实验汇总改为从 JSON 生成 |
| `logs/_splice_report_713.py` | §7.13 字节级拼接（含幂等保护） |
| `logs/_imgsz_ab/imgsz_ab_result.json` | 已补 confound + verdict |
| `logs/_EXPERIMENTS_SUMMARY.md` | 已刷新为含 896 负结论 |

> `logs/` 不进交付包（与既有口径一致）；但 §7.13 正文明写了这些产物名，
> 若评审要查证，源码目录内可见。

### 测试脚本（源有包无；**按 §02_code 整体进包口径**，`logs/` 不进包）
- `logs/_test_abstention.py`、`logs/_test_batch_screen.py`
  ⇒ 这两个留在 `logs/`（**不进交付包**，与既有 `logs/_*.py` 同口径）。
  ⚠️ 但它们是复现缺陷的**证据**，若评审需要，可在报告 §11.9 里给出脚本名。

## 二、执行顺序（不变）

1. 训练结束后：`python logs/_copy_delivery_pack.py`  ✅ 已完成（5857 文件 / 367.64 MB）
2. 再跑 `python logs/_pack_vs_src.py` ⇒ 判据1（内容不同）与「需复核」应同时归零  ✅ 归零
3. **★ 在包自身目录内**跑一次入口命令，验证 `_adapt_layout()` 生效：
   - `python code/batch_screen.py --dir=<某个图片目录>` （**新入口，必测**）
   - `python code/pipeline.py --help`
   - `python code/evaluate.py --help`
   - `python code/video_screen.py --help`
4. 验收存证追加到 `logs/_PACK_ENTRY_TEST.txt`

### ★★★ 第 3 步真的跑了 —— 抓到一个 P0（2026-09-25 14:46）

**只在源工程里跑是查不出来的**：包内 `results/train/*/weights/` 被刻意排除，
而 `pipeline.py` / `bench_pipeline.py` / `video_screen.py` / `batch_screen.py`
四个入口**各自硬写**该路径当默认值 ⇒ 包内一律「模型不存在」。
`pipeline.py` 更进一步，硬写的还是**已经不用的 v8s640**（主力 09-20 起是 v11s640）。

**修法**：`common.default_weight()` 单一实现 + 五处入口全部委托；
`app.py::_find_default_weight()` 一并改为委托（杜绝三处漂移）。
详见 **`logs/_PACK_WEIGHT_FIX_20260925.md`**。

**验收**：`logs/_pack_entry_test.py`（在包自身目录内 subprocess 跑全部入口）
→ `logs/_PACK_ENTRY_TEST.txt`。**全 PASS**，`batch_screen` 真跑 3 张 → 检出 16 处。

⚠️ **验收脚本会自动清理**包内 `results/vis/`（`batch_screen` 的 annotated 图**不跟随
`--json`** —— 第一次验收直接拿 568 张跑，往包里写了 568 张图，判据2 直接 FAIL）。

## 三、★ 特别提醒

- `batch_screen.py` 与 `video_screen.py` **此前从未进过包**（09-25 才新增），
  所以这次同步必须确认它们**真的被拷进去了**（只比「内容不同」会漏掉新文件 —— 见
  MEMORY.md 铁律 #17）。
- 包内目录被重命名（`02_code`→`code`），而 `batch_screen.py` 里
  `from common import (EVAL_DIR, RESULT_DIR, ...)` 依赖 `common._adapt_layout()`
  的探测式兜底 ⇒ **必须在包自身目录内实跑才敢说「开箱可运行」**。
- `app.py` **本次未改可见界面**（只改了内部权重解析函数 `_find_default_weight()`，
  界面/文案零改动）⇒ 已提交的实机运行视频仍然有效（无需因本次修复重录）。

## 四、本轮又新增待拷（14:50 更新：训练已结束）

`04_results/train/v11s640_merge/`（露筋碎片合并 A/B 处理组）—— 14:19 启动、14:42 完成（1338 s），
**属训练凭据类**，按既有 `TRAIN_RUN_ALLOW` 白名单口径处理（拷 `args.yaml`+`results.csv`，
`weights/` 与曲线图属噪声，不进包）。其配套
`04_results/eval/v11s640_merge_test_metrics.json` + `v11s640_merge_per_class.csv` 同样待拷。

★ **GPU 已空闲，可执行全量拷贝了**（纪律 #9 的前置条件已满足）。

## 五、本轮新增的「证据类」产物（logs/ 口径，不进包但报告点名）

| 源文件 | 说明 |
|---|---|
| `logs/_merge_ab/watchdog.out` / `merge_ab_result.json` | 合并 A/B 全过程与结论 |
| `logs/_fix_imgsz_ab_confound.py` | 补回 confound/verdict（幂等） |
| `logs/_update_experiments_summary.py` | 实验汇总改为从 JSON 生成 |
| `logs/_splice_report_713.py` / `_append_713_a_results.py` | §7.13 插入与结果回填 |
| `logs/_retire_20260925c~h.py` | MEMORY.md 第七次精简脚本 |

> 均留源码目录；§7.13 正文明写了这些产物名，评审需要时可查。


---

## 三、16:05 追记：本轮又新增的同步项（全部已完成）

### 3.1 新增的源改动（已在 16:00 的拷贝中进包）
| 源文件 | 变更 |
|---|---|
| `02_code/grade.py` | **§5.1 修复**：`LINE_CLASSES` 收窄为 `("crack",)`；新增 `BLOB_CLASSES`；`apply_abstention` 加 `_specific` 条件文本 |
| `06_deploy/app.py` | 删除 90 行内联弃权块（第二份实现）⇒ 改为委托共享实现；`_find_default_weight` 委托 |
| `07_report/技术报告.md` | §7.13 去重；新增 **§11.10**、**§11.11**；§11.9 结尾加补正指引；**§7.14**（TTA） |

### 3.2 文档（内容不同，已进包）
| 源文件 | 说明 |
|---|---|
| `07_report/技术报告.md` | 295608 B / CRLF=4716；`##`/`###` 重复标题 **0**；7.1–7.14 与 11.1–11.11 顺序干净 |

### 3.3 包一致性复验（16:00 实测）
```
【判据 1】内容不同 = 0
【判据 2】包有源无 = 0
【需复核】源有包无（不在白名单） = 0
结论：✅ PASS
```
- 唯一列项：`04_results/train/v8s640/results.csv`（「已识别·包内保留的复现凭据」，非野文件）。

### 3.4 包内入口验收（16:00 实测，`logs/_pack_entry_test.py`）
- `pipeline.py --help` / `evaluate.py --help` / `video_screen.py --help` / `batch_screen.py --help`
  → **4/4 EXIT=0**，且**全部解析到** `...外墙缺陷筛查工程包\weights\v11s640_best.pt`（P0 修复生效）。
- `batch_screen` 真跑 3 张验证图 → 16 处缺陷（efflorescence 14 / crack 2）→ 建筑级风险 **U**。
- 自动清理后**复跑判据2 仍 = 0** ⇒ 污染被彻底清掉。

### 3.5 仍不进包（与既有口径一致）
- `logs/**` 全部不进交付包。
- 但 §11.10 / §11.11 / §7.14 正文明写了这些证据脚本名，
  评审若需查证，**源码目录内可见**。

---

## 四、16:08 起在进行中的工作（尚未完成，届时需再同步）

### 4.1 ★ 露筋碎片合并 A/B 的 100 轮公平复跑
- **动机（方法学缺口）**：现行 §7.12 的合并 A/B 是 **20 轮**，对照基线 `v11s640_r20` 自身
  只有 0.70292，而报告的**主力基线 `v11s640` 是 100 轮**（0.72296）
  ⇒ **纯欠训练偏移 −0.02004** 污染了 A/B 差值。
  原实验脚本自己在 docstring 写了「若要写进报告的最终数字，须再用 100 轮复跑」。
- **设计**：基线 = 复用已存在的 100 轮 `v11s640`（不重训）；处理 = `v11s640_merge100`（新训 100 轮）。
  同配方 v11s640 / 同种子 42 / 同轮数 100，**唯一变量 = 标签**。
- **脚本**：`logs/_watch_merge_ab_100.py`；产出 `logs/_merge_ab100/merge_ab100_result.json`。
- **状态**：16:08 启动，训练中（`run_in_background=true`，task=XFyxCx）。预计 ~2.3 h。
- ⚠️ **完成后必须**：① 跑 `evaluate.py` 出 `v11s640_merge100_test_metrics.json`；
  ② 更新 §7.12 与 `_EXPERIMENTS_SUMMARY.md`；③ **重跑 `_copy_delivery_pack.py` + `_pack_vs_src.py`**。
- ⛔ **训练期间不跑其它重活**（内存硬约束）；`run_in_background=true` 不受 28 分钟限制。


---

## 五、19:10 用户要求暂停训练 —— 收尾状态与新增源改动

### 5.1 新增/改动的**源文件**（需进包）
| 源文件 | 变更 |
|---|---|
| `02_code/train.py` | **新增** `--imgsz=` / `--batch=` / `--workers=` 三个覆盖参数（不传时行为逐键相同） |
| `README.md` | 训练章节补「可覆盖的参数」清单 + `workers` 静默失败告警 |
| `logs/_update_experiments_summary.py` | 生成脚本新增 §8（imgsz batch 复跑未完成的诚实记录） |

### 5.2 imgsz 的 batch 单变量复跑：**尝试但未完成**
- 设计：`640@b16` vs `896@b16` 各 100 轮（`logs/_watch_imgsz_ab_100.py`）。
- **未完成**：本机内存不足（可用 1.1–7.6 GB / 16.88 GB），DataLoader worker 随机 OOM。
  三种 `workers` 实测：4 快但间歇崩、2 ≈16 h、0 ≈28 h，均不可行。
- ⇒ **不产出任何数字**；报告 §7.13.4 的 `strict_wording`（batch 混杂）**保持原样**，
  并在该节末尾如实补记了这次未完成的尝试。

### 5.3 失败实验的产物清理（已完成）
- `04_results/eval/train_summary.json`：移除 1 条 `status=failed` 的**幽灵记录**
  （`v11s640_b16`）。**8 条 → 7 条**；除 `results` 外的键逐键校验未变；CRLF 保持。
  备份：`logs/_backup_train_summary_before_cleanup_20260925_210618.json`。
- 4 个失败/空壳目录**移动**（未删除）到 `04_results/train/_failed_20260925/`。

### 5.4 同步后必核
1. `python logs/_copy_delivery_pack.py`
2. `python logs/_pack_vs_src.py` ⇒ 判据1/判据2/需复核 均应为 0
3. 包内 `report/技术报告.md` 与源 **md5 应一致**
