# P2 清理盘点（2026-09-24 夜）

> 只读盘点。**本文档生成时未移动、未删除任何文件。**
> 执行结果另见同目录 `_P2_CLEANUP_EXECUTED.md`。

## 一、总览

| 组 | 对象 | 占用 | 性质 | 处置 |
|---|---|---:|---|---|
| **1** | `_archive_20260920/junk_txt_20260920/` (2 文件) | **3024 MB** | famerL 数据集**结构转储**的两份巨大 txt（各 1.585 GB、349 万行），内容为文件名清单 | 移出仓库 → 项目外归档 |
| **2** | `logs/_famerL_tmp/val80.zip` | **849 MB** | HuggingFace 匿名账号 `famerL` 的原始下载件，SHA256 已校验 | **保留**（9/21 已判定：来源随时可能消失，成本低，留作证据） |
| **3** | `logs/_rec/` | 75 MB | 旧版演示视频 + 抽帧目录（`_previous_*` 系列已被新版替代） | 移出仓库 → 项目外归档 |
| **4** | `logs/_backup_lastpt_20260923/` | 73 MB | 9/23 类平衡训练中途的 `last.pt` 快照（该训练最终跑满 100 轮，快照已被成品取代） | 移出仓库 → 项目外归档 |
| **5** | `_archive_20260920/dataset_v1_20260920/` + `dataset_v2_20260921/` | 754 MB | V1/V2 数据集副本（V3 为现行版） | 保持原位（已归档目录，非交付物） |
| **6** | 两处 package quarantine（`_quarantine_20260921/`、`_quarantine_20260922_gradio/`） | 135 MB | 半损毁包的隔离副本（已用 wheel 全量解包修好） | 保持原位（回滚点） |
| **7** | 根目录散落练习文件 | <1 MB | `-A`/`_rcheck.txt`/`practice*.py`/`main.py`/`YOLO.py`/`predict.py`/`split_dataset.py`/`train.py`/`fruit.yaml`/`bus.jpg`/`yolov8n.pt`/`yolov8s.pt` | 移入 `_archive_root_loose_20260924/` |

## 二、逐项依据

### 组 1 `junk_txt_20260920`（3024 MB）—— 最需要处理的一项
- 内容核实（`head -c 300`）：`_famerL_struct.txt` 开头为「=== 顶层结构 ===" 后跟 `[DIR] images / [DIR] labels`；
  `_famerL_s2.txt` 为「=== 小体积 yaml/txt」清单。
  两者都是 2026-09-20 深夜为「摸清 famerL 数据集内部结构」而做的 `tree`/`ls -R` 类转储，
  **每文件 349 万行、1.585 GB**。
- 判据：目录名即 `junk`（垃圾），生成方 `_famerL_tmp` 已在 9/21 清理中处理；
  现行数据集结构已由 `01_data/raw/_public_datasets/construction_defect_yolo` 及其 `_AUDIT_*` 报告覆盖。
- ⇒ 结论：**移出项目仓库**（不是删除），归档到 `D:\pythonstudy 备份\_archive_out_of_repo_20260924\`，
  路径留在本盘点文档里可查回。

### 组 2 `val80.zip`（849 MB）—— **明确保留**
- 9/21 `_CLEANUP_CONFIRM.md` C 项已判定保留：来源匿名、未声明许可、下载量仅 9 次，随时可能消失。
- 本次不改判：成本 849 MB，证据价值 > 成本。

### 组 3 `logs/_rec/`（75 MB）
- 内含 `_previous_2scene_实机运行视频.mp4`（11 MB）、`_previous_3scene_oldUI_实机运行视频.mp4`（16 MB）、
  `_previous_3scene_oldcode_实机运行视频.mp4`（24 MB），以及多组抽帧目录与 3 个浏览器录制 `.webm`。
- 判据：文件名前缀 `_previous_` = 被新版替代；`06_deploy/app.py` 在 9/23 改版，
  演示视频需重录（任务 #69），旧片已作废。
- ⇒ **移出项目仓库**，与组 1 同归档根。

### 组 4 `logs/_backup_lastpt_20260923/`（73 MB）
- 内含 `last.pt`（76,011,938 B）+ `results.csv`，时间戳 9/23 17:35。
- 判据：`v11s640_clsbal` 最终 `results.csv` 已 100 轮、`03_weights/` 有正式 `best.pt`；
  该快照是**续训前的中间态**，成品已落盘。
- ⇒ 无需常驻项目内。**移出项目仓库**归档（仍是回滚材料，只是不放在仓库里）。

### 组 7 根目录散落文件
- 逐份读了内容（见下），确认全部是**入门练习**或**水果子任务**遗留，与「外墙缺陷筛查」主线无关：
  - `main.py`：`print("Hello World")` / `a=10;b=20`
  - `practice def.py`：`exec` 语法的练习稿
  - `practice1.py`：**0 字节**空文件
  - `YOLO.py`：`cv2.VideoCapture(0)` 摄像头实时推理练习
  - `predict.py` / `split_dataset.py` / `train.py` / `fruit.yaml`：均为 `D:/pythonsudy/yolo_fruit` 水果数据集脚本，
    路径写的是**旧盘符** `D:/pythonsudy`（无空格），与本工作区 `D:/pythonstudy 备份` 不同
  - `-A`：**0 字节**，文件名疑似某次命令 `-A` 被误当路径创建
  - `_rcheck.txt`：一份 9/23 的内存快照（带 BOM），内容是 `WorkBuddy 3566 MB` 等，已过期
  - `bus.jpg` / `yolov8n.pt` / `yolov8s.pt`：Ultralytics 官方示例资产（后者是预训练权重）
- ⇒ 全部移入 `_archive_root_loose_20260924/`，根目录只留项目级文件（README / LICENSE / 审查报告 / 子项目目录）。

## 三、执行方式

- **一律用「移动」而非删除**：本机删除会被 safe-delete 拦截（`SAFE_DELETE_FAIL_CLOSED`、`SHFileOperationW 0x78`）。
- 归档根：`D:\pythonstudy 备份\_archive_out_of_repo_20260924\`
- 根目录散件：`D:\pythonstudy 备份\_archive_root_loose_20260924\`
- 移动前对每组做**文件数 + 字节数**记录，移动后复核，保证 0 丢失。
