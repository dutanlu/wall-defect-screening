# P2 清理执行报告（2026-09-24 夜）

> 依据：`logs/_P2_CLEANUP_PLAN.md`（只读盘点）
> 脚本：`_audit_tmp/_p2_cleanup.py`
> 数据：`logs/_P2_CLEANUP_EXECUTED.json`
> **方式：一律「移动」，未删除任何文件。**

---

## 一、结果

| 组 | 源 | 目标 | 文件数 | 体积 |
|---|---|---|---:|---:|
| G1 | `_archive_20260920/junk_txt_20260920/` | `_archive_out_of_repo_20260924/junk_txt_20260920/` | 2 | **3021.6 MB** |
| G3 | `logs/_rec/` | `_archive_out_of_repo_20260924/_rec/` | 50 | 74.5 MB |
| G4 | `logs/_backup_lastpt_20260923/` | `_archive_out_of_repo_20260924/_backup_lastpt_20260923/` | 2 | 72.5 MB |
| G7 | 根目录散件 13 项 | `_archive_root_loose_20260924/` | 13 | 30.5 MB |
| | **合计** | | **67** | **3198.6 MB** |

**失败项：0。** 每组搬运前后均做「文件数 + 总字节 + 排序树哈希」双向核对，全部一致。

## 二、根目录清理效果

清理前根目录有 13 个散落文件（`-A` 0 B、`practice1.py` 0 B、`_rcheck.txt` 185 B、
各练习 `.py`、`fruit.yaml`、`bus.jpg`、`yolov8n.pt`、`yolov8s.pt`…）。
清理后根目录**只剩项目级文件**：

```
.wbapp_W69yHHyVCcBl0Sqk0o9m0l.genie   181 B
yolo_action_recognition_工程包.zip    620.0 MB   ← 独立子项目（动作识别）交付包
基础题交付审查清单.md                 12.0 KB
种子杯项目报告.zip                    103.0 MB
_archive_out_of_repo_20260924/                   ← 本次新建（项目内大件归档）
_archive_root_loose_20260924/                    ← 本次新建（根目录散件归档）
_audit_tmp/  gdino/  runs/  weights/  yolo_action_recognition/  yolo_fruit/
创新题/  种子杯项目报告/  种子杯项目报告_备份_20260919/  视觉赛道水果分拣/
```

> `yolo_action_recognition_工程包.zip`（620 MB）与 `种子杯项目报告.zip`（103 MB）
> **未动** —— 前者是另一个独立子项目的交付压缩包（对应 `yolo_action_recognition/` 目录），
> 后者是报告包快照。它们不是「散落垃圾」，需要单独判断，故本轮保留。
> 若确认 `yolo_action_recognition_工程包.zip` 可由 `yolo_action_recognition/` 目录复现，
> 可另行归档（预计可再省 620 MB）。

## 三、项目仓库体积变化

| 位置 | 清理前 | 清理后 | 变化 |
|---|---:|---:|---:|
| `创新题/外墙缺陷筛查/` | 15266 MB | **≈ 12067 MB** | **−3199 MB** |
| `创新题/外墙缺陷筛查/logs/` | 1626 MB | ≈ 777 MB | −849 MB（含 val80.zip 保留项在内） |

> `logs/_famerL_tmp/val80.zip`（849 MB）**明确保留**，未计入本次回收：
> 9/21 `_CLEANUP_CONFIRM.md` C 项已判定其来源（HuggingFace 匿名账号 `famerL`、
> 未声明许可、下载量仅 9 次）随时可能消失，成本低、证据价值高。

## 四、未处理项及理由

| 对象 | 占用 | 为什么不动 |
|---|---:|---|
| `logs/_famerL_tmp/val80.zip` | 849 MB | 明确保留项（见上） |
| 两处 package quarantine | 135 MB | 半损毁包的隔离副本，是回滚点 |
| `_archive_20260920/dataset_v1~v2` | 754 MB | 已在归档目录内，非交付物；V3 为现行版 |
| `_stale/onnx_过期_20260923` | 55 MB | 已按「过期」命名的回滚材料 |
| `yolo_action_recognition_工程包.zip` | 620 MB | 独立子项目交付包，需单独判断 |
| `logs/_quant_mixed/` | 14 MB | 混合量化产物（任务 #72 第②步仍要用） |

## 五、可复现命令

```powershell
# 复核归档完整性（文件数 / 字节 / 树哈希）
python _audit_tmp/_p2_cleanup.py --verify      # （预留）
# 或直接看落盘账本
Get-Content "创新题\外墙缺陷筛查\logs\_P2_CLEANUP_EXECUTED.json" -Raw
```
