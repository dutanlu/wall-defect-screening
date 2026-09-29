# 全项目审计 · 进度与信任度分级（2026-09-28）

> ## ★ 状态更新（2026-09-29 收口）
> 下列事项在本文件生成后**已完成**，正文里的「未做 / 需斟酌 / 已知问题」以本注记为准：
> · **P3 文档一致性**：交叉引用 0 断链、产物 0 真实缺失、数字 96.2% 可追溯（修 §1.6 断链、
>   `_verify_ratelimit.py`→`_verify_phone_multiplex.py`、`v8s640/results.csv` 拷回）。
> · **边审边修**：A1（`window_ok` 默认值 `True`→`False` + **285 图零改数证明**）、
>   A2（三入口裸 flag 统一走 `common.argv_flag`）、A3（环境类别收口 `gsd.ENV_CLASSES`/`DEFAULT_ENV`）、
>   C1（字数 499/504）、E1/E2/E3（技能文件悬空引用/重编号/补两课）—— 全部完成。
> · **P1 接线**：`single_oom`→`pipeline.run_one`（`--single-ood`，默认关）；
>   `change_detect`→`pipeline.compare_pair()` + **Web「🔄 两次巡检对比」Tab**；
>   `stitch_facade`→`pipeline.stitch_facade()`（后端）。
> · **T3 真跑闭环**：`run_one` 在 285 张测试图上端到端跑通（2.8 MB 结果）。
> · **B1 包内权重**：处置为「报告/README 加权重口径说明」（不补权重，是有意设计）。
> · **仍待办**：P4（197 个结论文件比对）、P5（需求基线回溯）、UI 接线后的实机视频重录。


> **用途**：把本轮审计的结论按**信任度**分三档，并**完整列举尚未审查**的部分。
> 交给另一个模型做独立复核时，可按本文件给出的**范围 + 复跑命令**逐项验证，
> 不必依赖叙述。
>
> **阅读约定**：本文件只回答「**审了什么、信到什么程度**」，不重复结论细节；
> 每条都给出**证据载体**（脚本 / 台账 / `文件:行号`），细节去对应台账取。

---

## 零、本轮产出的脚本与台账（全部可复跑）

| 脚本 | 作用 | 台账 | 复跑命令（工作目录 = 工程根） |
|---|---|---|---|
| `logs/_audit_static.py` | 9 个静态断言 checker | `logs/_AUDIT_STATIC.txt`（CRLF） | `D:\下载\python.exe logs/_audit_static.py` |
| `logs/_selftest/fixtures/` | 10 个**反向验证**样本 | — | `... _audit_static.py --selftest` |
| `logs/_audit_imports.py` | 导入冒烟 + 接线缺口图 + 布局探测 | `logs/_AUDIT_IMPORTS.txt`（CRLF） | `... _audit_imports.py` |
| `logs/_audit_pack.py` | 交付包开箱（包内 import + `__file__` 断言 + 权重缺口 + 零污染） | `logs/_AUDIT_PACK.txt`（CRLF） | `... _audit_pack.py` |
| `logs/_verify_pack_entry.py`（**沿用**） | 包内 4 个 CLI + `app.build_ui()` + `queue` | `logs/_PACK_ENTRY_TEST.txt` | `... _verify_pack_entry.py` |
| `logs/_pack_vs_src.py`（**沿用**） | 交付包 vs 源 逐文件 md5 | `logs/_PACK_VS_SRC.txt` | `... _pack_vs_src.py` |

**解释器口径**：跑项目代码用 `D:\下载\python.exe`（实测 Python 3.13.15 / numpy 2.5.2 /
cv2 5.0.0 / gradio 6.28.0）。审计脚本本身只用标准库，用托管 Python 也能跑。

**扫描范围**：`02_code/`（58 个 `.py`）+ `06_deploy/`（3 个）+ `09_uav_realtime/`（6 个）
= **67 个生产 `.py`**。排除：`__pycache__` / `_selftest` / `_archive_20260920` /
`_verify_out` / `runs` / `_backup_before_private` / `logs`。

### 本轮结论数字速览

| 项 | 结果 |
|---|---|
| 静态断言 | **FAIL = 1，WARN = 145**（修 checker 自身缺陷前是 FAIL = 27，见 §1.3-8） |
| 编译 T0 | 67/67 通过；6 个文件含 UTF-8 BOM |
| 源目录导入 T1 | **35/35 成功**（core 19 / steps 7 / uav 6 / deploy 3），刻意跳过 1 个 |
| 包内导入 P2 | **20/20 成功**，且 `__file__` 全部落在包内 |
| 包 vs 源 | **PASS**（内容不同 = 0 / 包有源无 = 0 / 需复核 = 0） |
| 包内入口开箱 | **PASS**（4 个 CLI + `build_ui` + `queue`） |
| 接线缺口 | 2 个模块（`change_detect.py` 0 引用；`single_oom.py` 仅被脚手架引用） |

---

## 一、已审查过的内容

### 1.1 完全信任

判定标准：**机器可复跑 + 反向验证通过 + 结论有产物/字节级证据支撑**，且判据本身
不依赖我的主观口径。

| # | 结论 | 证据载体 |
|---|---|---|
| 1 | 67 个生产 `.py` **全部编译通过**（内存 `compile()`，不生成 `.pyc`） | `_AUDIT_STATIC.txt` 【T0】 |
| 2 | 源目录 **35 个模块全部 import 成功、0 失败**，含 `app.py` / `train.py` / `phone_live.py` | `_AUDIT_IMPORTS.txt` 【T1】 |
| 3 | 交付包内 **20 个模块全部 import 成功**，且每个模块的 `__file__` **断言落在包内**（排除"跑在源码上"的假 PASS） | `_AUDIT_PACK.txt` 【1】 |
| 4 | 11 个路径常量**全部落在工程根内**；`dataset/wall_defects.yaml`、`weights/v11s640_best.pt`、`grading_rules.json` 必需文件均存在 | `_AUDIT_IMPORTS.txt` 【A9】 |
| 5 | A9 断言**反向验证通过**：喂合成越界路径/缺失文件/坏权重，断言函数全部抓到 | `_AUDIT_IMPORTS.txt` 【A9-反向验证】 |
| 6 | 交付包**开箱可运行**：4 个 CLI 的 `--help` 全部 rc=0；`app.build_ui()` → `Blocks`；`queue` → `(8, 1)`；`batch_screen` 在包内可寻址 | `_PACK_ENTRY_TEST.txt` |
| 7 | 交付包与源**逐文件 md5 一致**（内容不同 = 0），无野文件（包有源无 = 0），**无未列入白名单的漏件** | `_PACK_VS_SRC.txt` |
| 8 | 跑审计**未污染交付包**：跑前跑后包内 `__pycache__` 均为 0 | `_AUDIT_PACK.txt` 【3】 |
| 9 | **所有 `return Measurement(...)` 路径都显式赋值 `window_ok`**（0 条 FAIL）—— 这是"A5 唯一命中是危险默认值本身"的直接证据 | `_AUDIT_STATIC.txt` 【A5】 |
| 10 | 生产核心 **0 处硬编码绝对路径**；20 处全在 `02_code/_*` 脚手架 | `_AUDIT_STATIC.txt` 【A2】 |
| 11 | 生产代码中唯一的**类别体系复本** = `02_code/bench_pipeline.py:165` 的 `_NAMES` | `_AUDIT_STATIC.txt` 【A8】 |
| 12 | 67 个模块中 **56 个被产品代码引用**；`change_detect.py` 零引用；`single_oom.py` 仅被脚手架 `_verify_single_ood.py` 引用 | `_AUDIT_IMPORTS.txt` 【A7】 |
| 13 | **反向验证套件本身 9/9 通过**，含"带 `# noqa` 声明 / 已知正确 ⇒ 必须**不** FAIL"的另一半 | `_audit_static.py --selftest` |
| 14 | 行尾逐文件实测（`read_bytes()`，**不用** `read_text()`）：`02_code`/`06_deploy`/`07_report` 核心 `.md` 为 CRLF；`07_report/录屏执行手册.md` 为**纯 LF**；`logs/*.py` 全部**纯 LF** | 本轮实测，见 §1.3-8 说明 |
| 15 | `07_report/项目介绍书.md` 字数实测：**正文 CJK 499 / 含标题 504** | 本轮实测 |

> **为什么这 15 条可以完全信任**：每一条要么是**字节级/进程级事实**（编译结果、
> rc、md5、行尾计数），要么**自带反向验证**（若判据恒真，喂已知错误时必露馅）。
> 复核者只需重跑上表命令即可独立得到同样结果。

### 1.2 比较信任

判定标准：**计算过程是机器做的、可复跑**，但**判据（阈值 / 分类规则 / 启发式）
是我本轮的设定**，换一个人可能给出不同但同样合理的判据。

| # | 结论 | 为什么只能"比较信任" | 证据载体 |
|---|---|---|---|
| 1 | 静默失败 **33 条 WARN、0 FAIL** | 三级定级规则（`# noqa` 声明 / 窄异常类型 / 盲捕获+写操作）是**我 2026-09-28 定的**；其中"写操作"清单与"窄异常"名单都带主观性 | 【A6】 |
| 2 | 环境类别字面量：「一类」5 处/5 文件、「三类」5 处/5 文件、「二类」**20 处/13 文件** | **计数是实测**；但"该不该收口、收成什么形态"是判断（§1.3-5） | 【A4】 |
| 3 | 占位标记 7 条（`TODO/占位/XXX` 正则命中） | 正则命中属实；**是否真有功能影响未逐条读源码确认** | 【A1】 |
| 4 | 「假设文件存在」14 条 | **设计上就是启发式、永不判错**（本项目大量读盘已在 `common.imread_u` / `read_yolo_labels` 里做了保护），误报率高 | 【A3】 |
| 5 | A8 的 11 条 WARN「类别子集/策略清单」 | "复本 vs 策略清单"的切分判据（**顺序 == `CLASSES` 前缀**）是我定的 | 【A8】 |
| 6 | 写死模型名等 **70 条 WARN** | 计数与位置可复跑；但"是否应改为走 `default_weight()`"需要按场景判断，且**我只核了前 20 条** | 【A2】 |
| 7 | `技术报告.md:5200` / `:5217` 已**主动披露**"拼接未接入 Web 界面 / 无 CLI-UI 入口" ⇒ 不构成夸大 | 属**语义判读**（我读了原文），不是机器结论 | `技术报告.md:5200,5217` |
| 8 | A6 的 5 处「已声明意图」（`# noqa: BLE001` 或 docstring「失败返回 None」） | 同上：读了源码做语义判断 | `common.py:140`、`change_detect.py:215`、`app.py:653`、`step6_quantize.py:257,389` |

### 1.3 还需要再斟酌

判定标准：**我做了判断，但证据链不完整**；或**存在未定选项**；或有**已知反例/边界**。

| # | 事项 | 缺口在哪 |
|---|---|---|
| 1 | 规划代理提出的「三个未接线模块 = 报告夸大」，我**证伪了 2 个**（`stitch_facade`、`change_detect` 都有主动披露） | **第 3 个 `single_oom.py` 的报告侧表述尚未核对** —— 它确实未被报告侧检查过 |
| 2 | 「把 `measure.py:64` 的 `window_ok` 默认值 `True` 改成 `False` 可证零行为变化」 | 依据只有**AST 静态断言**（所有 return 路径都显式赋值）。**运行时逐图比对（285 张测试图逐字节零差异）尚未做** —— 那才是能落地的证明 |
| 3 | `07_report/README.md:288` 写「正文 493 个中文汉字 ⇒ 全文 498 字」与实测 **499 / 504** 不符 | 差异属实；但**"该统一成哪个口径"未定**（正文？全文？含标题？），且需与另外 3 处声明一起对齐；另 `logs/_REMOTE_EVIDENCE.md:447` 的「493 字」是**历史证据，不该改** |
| 4 | 交付包内权重只有 2 个（`v11s640_best.pt`、`v11s640_clsbal_best.pt`），缺 `v8s640_best.pt` ⇒ **v8-vs-v11 版本对照在包内不可复现** | 结论结实；但**处置方式未定**（补权重进包 vs 在报告里明确降级表述） |
| 5 | 环境类别字面量散落（20 处/13 文件）+ 枚举三处独立维护 | **收口的正确形态未定**：收成 `common.DEFAULT_ENV` 常量？收成函数？还是保持字面量（因为 `rt_config.yaml` 必须写字面量） |
| 6 | 交付包完整性 | 我只核了 **3 个代码目录**（`code`/`deploy`/`uav_realtime`）的模块数与源一致；**其余 6 个目录**（`dataset`/`report`/`results`/`weights`/`grading_rules`/`photo_collector`）**未逐个核对**（`_pack_vs_src.py` 的 md5 覆盖了内容一致，但不覆盖"该收哪些"的设计判断） |
| 7 | T0 发现 6 个 `.py` 含 UTF-8 BOM | **影响未评估**：Python 3 本身接受 BOM，但需确认是否影响发布脚本/工具链/shebang |
| 8 | `_audit_static.py` 自身的判据设定（`len(names) >= 3`、`DEAD_MODEL_RE`、`ABS_PATH_RE`、`WRITE_OP_RE`、`NARROW_EXC`） | **未经第二人复核**。本轮已通过反向验证证明它们"不是恒真"，但**阈值取值是否合适**没有独立校验 |
| 9 | 本轮**反向验证当场逮到我自己脚本的 3 个 Bug**（扫描根被硬编码 ⇒ `--selftest` 全空跑；A8 无视脚手架降级且判据混三种语义；`class_order()` 只认 `Assign` 不认 `AnnAssign`） | 说明"脚本自己也可能系统性偏"，**同类偏差可能仍存在于未被反向验证覆盖的 checker 里** |

---

## 二、尚未审查过的内容（完整列举）

### 2.1 本轮计划里已排但**尚未全部完成**（P3 已在本轮完成，见各行）

| 项 | 计划内容 | 状态 |
|---|---|---|
| P3-a | `文件:行号` 引用校验（复用 `logs/_audit_line_refs.py`） | ✅ 已完成：BAD=0 / SHIFT=0；10 条"找不到源"全为 ultralytics 内部文件（合法外部引用） |
| P3-b | `§X.Y` 交叉引用断链（复用 `logs/_audit_xrefs.py`） | ✅ 已完成：断链 1 → 0（已修 §1.6 悬空引用） |
| P3-c | 报告引用的产物路径是否真实存在（复用 `logs/_check_claim_artifacts.py`） | ✅ 已完成：6 缺失 → 修 2 处真实缺失；剩 4 处为 checker 的 brace 展开误报 |
| P3-d | 报告数字能否从 `04_results` JSON 复算（复用 `logs/_audit_report_numbers.py`） | ✅ 已完成：96.2% 可追溯（1133/1178）；45 条为派生值（差值/SE/CI 半宽） |
| P3-e | P3 的反向验证（注入已知错误必报无法追溯） | **未跑**（遗留） |
| P4 | `logs/` **196 个 `_*.md` 结论文件**与当前主线结论是否矛盾（历史上多轮结论已翻盘） | **未做** |
| P5-a | 以 `创新题/外墙筛查_实施流程.md`（含「创新点 1~4」）+ `选题可行性验证报告.md` 为基线**逐条回溯**：已做 / 走偏 / 遗漏 / 边缘未覆盖 | **未做** |
| P5-b | 人判抽样：技术报告逐节抽 3 条可证伪断言（约 30~45 条）；4 份提交物（技术报告/答辩材料/PPT/介绍书）各抽 1 处同一数字做四方对照 | **未做** |
| T2 | 早退分支实跑：`pipeline.py` / `batch_screen.py` / `video_screen.py` / `evaluate.py` / `train.py` / `exp_*` 的最小实跑用例 | **未跑**（只做了 import） |
| T3 | 真跑闭环：`pipeline` 单图推理、`rt_infer` 合成视频 2 帧 | **未跑** |

### 2.2 完全没进计划的领域

**数据与标注**
1. `01_data/{raw,dedup,unified,dataset,audit}` 的图像/标注/yaml 一致性、类平衡、跨集泄漏复检
2. `01_data/dataset` 的 1995 / 568 / 285 划分与 `split_report.json` 的实际逐图核对
3. 数据集来源与许可（多源合并的素材出处核查）

**模型与产物**
4. 22 个 `.pt` 权重的**实际推理正确性**（未做任何一次真实前向）
5. `v11s640_best_v3.onnx` / `v11s640_best_v3_fp16.onnx` 能否加载，并复现 9.72 MB / mAP50 0.71093
6. `04_results/{train,eval,quant,ablation,gradcam,vis}` **全部产物**的完整性与口径
7. `04_results/eval` 的 **12 个 `bfdd_*.json`** 与报告 §9 的一致性
8. 边界声明类产物：`conformal_measure.json` / `decision_rule.json` / `collapse_detector*.json` / `image_quality_gate.json` / `multi_stream_measured.json` / `phone_*.json`（约 20 个）
9. `05_quantify_grade/grading_rules.json` 是否与 `gsd.py` / `grade.py` **同步**（该文件自称"由脚本导出、勿手工编辑"）
10. `requirements.txt` 声明版本 vs 实测环境是否一致

**运行期行为**
11. `06_deploy/app.py` Gradio 界面**实际交互**（上传 / 批量 Tab / 视频 Tab / 结果面板 / 拒答面板）
12. `06_deploy/live_stream.py`、`phone_live.py` 的运行时（含 7862 / 7863 端口、HTTPS 自签证书链）
13. `09_uav_realtime/rt_infer.py` 合成视频实跑、`rt_verify.py`、UDP 图传链路（已知设备侧未就绪）
14. `08_photo_collector/site`（网页采集站）与包内 `photo_collector/`

**交付物与合规**
15. 二进制交付物：23 页 PPT（`外墙缺陷筛查_答辩PPT.pptx`）、`项目介绍书.docx`、演示视频、实机录屏、`外墙缺陷筛查_开源仓库.bundle`
16. `LICENSE` / `LICENSE-NOTES.md` 的内容与第三方素材许可
17. 根目录 **4 份审查报告**（`审查报告_交付物完整性与缺口.md`、`审查报告_赛题符合性复核_20260924.md`、`审查报告_赛题符合性大检查_20260922.md`、`自检报告_20260920.md`，共约 110 KB）**自身**的准确性
18. 交付包其余 6 个目录的逐文件核对（见 §1.3-6）
19. 项目 GitHub 侧：`.gitignore`、本地仓库状态、远端 blob 行尾
20. `06_deploy/启动说明.txt`、`run_app.bat`、`_grun.txt` 等小文件

**工程卫生**
21. `requirements.txt` 之外的环境假设（如 `Arial.ttf` 必须在 `USER_CONFIG_DIR` 的约定）
22. `MEMORY.md` 与日记等**记忆文件**与项目实际状态的一致性
23. 技能文件 `deterministic-patch-and-verification/SKILL.md` 的已发现缺陷：
    §七 引用**不存在的 §五**（悬空引用）、**§1.9 重复编号**、§1.7/1.8 排在 §1.9/1.10 **之后**（编号与文件顺序不一致）、**缺两条实测教训**（CRLF 写回前必须归一化 / 有字数上限的交付物先量口径）
    —— **已发现、本轮未修**

### 2.3 刻意排除（有理由，不是遗漏）

| 排除对象 | 理由 |
|---|---|
| `logs/` 下 **1755 个历史 `.py`**（579805 行） | 是工地不是交付物；只抽其中的 `_*.md` 结论文件（见 §2.1 P4） |
| `_archive_20260920/`、`_archive_out_of_repo_20260924/`、`_archive_root_loose_20260924/` | 归档，语义就是"不再维护" |
| `runs/`、根 `weights/`、`_audit_tmp/` | 训练/临时目录 |
| 同工作区内的**其它三个工程**：`视觉赛道水果分拣/`、`yolo_fruit/`、`yolo_action_recognition/`、`gdino/` | **不属于本项目** |
| `种子杯项目报告/{YOLO项目报告, cv_fruit_measurement项目报告}/` | 同一比赛下的**另外两个独立项目** |
| `2026-09-1x/2x` 历史日记（约 1.2 MB） | 只在需要取证时按需读取 |

---

## 三、给复核者的建议入口

按**性价比**排序，建议先做这三件（都能纯机器复跑、且覆盖最关键的"夸大风险"）：

1. **`logs/_audit_line_refs.py` + `logs/_audit_xrefs.py` + `logs/_audit_report_numbers.py`**
   —— 直接检验维度 3「报告与代码/数据是否一致、有无夸大」。注意
   `_audit_report_numbers.py:119-134` 记录过一次**教训**：容差 5e-3 曾导致 524 处
   "全部可追溯" = **100% 假阳性**；判据必须是"精确相等 或 按报告位数四舍五入（≤5e-4）"，
   且**按 run 名绑定**（`v11s640` = 0.72296 与 `v11s640_merge_s2024` = 0.71236 是两个数）。
2. **`logs/_audit_static.py --selftest`** —— 先确认 checker 的判据不是恒真，再看
   `_AUDIT_STATIC.txt` 的 **1 条 FAIL** 与 145 条 WARN。
3. **T3 真跑闭环**（`pipeline` 单图 + `rt_infer` 合成 2 帧）—— 这是唯一能直接回答
   维度 1「能否**完整正确执行**它宣称的功能」的手段；本轮只做到 import 与早退，
   **尚未做过一次真实推理**。

**最容易踩的坑（本项目实测）**：
- `train.py` **无 argparse** ⇒ 用 `--help` 之类"探测"它会被**静默忽略并真开训**；
- `app.py` / `live_stream.py` / `phone_live.py` **无 `--help`** ⇒ 空参会**真起服务**；
- 交付包里目录被**重命名**过（`02_code`→`code` 等）⇒ 一致性核对 PASS **不等于**包内可运行；
- 「脚本报 ✅」不等于判据有效 ⇒ **必须喂一个已知错误，断言它报 ❌**。
