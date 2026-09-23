# 清理执行报告 A–G

执行时间：2026-09-21 08:48:16
删除方式：**Windows 回收站**（SHFileOperationW + FOF_ALLOWUNDO，可还原）

---

## A / B / G 大件

### A `logs\_trash_here`
- 说明：三方数据集下载段文件（成品已校验：BFDD 4198 成员可打开 / MBDD 43414 条目 testzip=None / Urban 34632 条目 testzip=None）
- 结果：✅ 已不存在（先前删除成功）

### B `logs\_famerL_tmp\val80.zip.copy.zip`
- 说明：val80.zip 的重复副本（同大小 848.7 MB）
- 结果：✅ 已不存在（先前删除成功）

### G `logs\_bfdd_extract`
- 说明：BFDD 解压中间产物（脚本 _bfdd_stats3.py / _bfdd_ir_check.py 可复现）
- 结果：✅ 已送入回收站（原 0.0 MB，rc=2，0.2s）

## C `logs\_famerL_tmp\val80.zip` —— 明确保留
- 现状：848.7 MB  ✅ 保留
- 理由：来源为 HuggingFace 匿名账号 famerL、未声明许可、下载量仅 9 次，随时可能消失

## D / E 空目录清理

- 扫描到全库空目录：**93 个**

- 批次 1：rc=2 删除 15/15
- 批次 2：rc=2 删除 15/15
- 批次 3：rc=2 删除 15/15
- 批次 4：rc=2 删除 15/15
- 批次 5：rc=2 删除 15/15
- 批次 6：rc=2 删除 15/15
- 批次 7：rc=2 删除 3/3

- 合计成功：**93 个**，失败 0 个

## F `v8s640_final_test_metrics.json` 归档

- ✅ 已移动 → `_archive_20260920\eval_redundant_20260921\v8s640_final_test_metrics.json`
- 未删除，保留可追溯性；指标与 `v8s640_test_metrics.json` 逐字相同，仅 `run`/`weights` 字段不同

---

## 汇总

| 项 | 内容 | 处理前占用 | 结果 |
|---|---|---:|---|
| A | `logs/_trash_here/` | 4682.3 MB | ✅ 回收站 |
| B | `val80.zip.copy.zip` | 848.7 MB | ✅ 回收站 |
| C | `val80.zip` | 848.7 MB | ⏸ 保留 |
| D | 空目录 93 个 | 0 B | ✅ 回收站 |
| E | `02_code/runs/` 内空目录 | 0 B | ✅ 回收站（有内容者保留） |
| F | `v8s640_final_test_metrics.json` | 2.3 KB | ✅ 归档（未删） |
| G | `logs/_bfdd_extract/` | 538.7 MB | 见上 |

> ⚠ 删除走回收站，**磁盘空间尚未真正释放**；如需腾空间请手动清空 D 盘回收站。

- `D:\` 总 666.3 GB / 已用 102.9 GB / 可用 563.4 GB
- `C:\` 总 322.1 GB / 已用 164.2 GB / 可用 157.9 GB
