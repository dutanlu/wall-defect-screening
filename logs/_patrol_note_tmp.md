
### 2026-09-26 09:27–09:35（巡检第 10 次·自动化）
- **队列活着，无需重启。** `_watch_queue_0925.py`(29184) → `_watch_seed_merge100.py`(42576)
  → `train.py --tag=_merge_s2024`(12192，08:57:14 起) → 13 个 dataloader worker。**恰好 1 个训练**。
- GPU 连采 99/97/97/89 %、6264 MiB ⇒ 在跑（判据 = 「util 持续 0 且 15 min 行数不变」未触发）。
- item#1 完成态不变（`queue_state.json`: imgsz_ab100=done(rc_ok)）。
- **item#2 = 3/4 完成**：`v11s640_s123`(05:12:16) / `merge_s123`(07:03:22) / `s2024`(08:56:50)
  三个 run 各 **101 行**且三份 eval JSON 已落盘；第 4/4 `v11s640_merge_s2024` 进行中：
  `results.csv` **27 行**（26 轮完，末次落盘 09:26:36），≈69 s/轮（基准 63，正常）
  ⇒ 剩 ~74 轮 ≈85 min ⇒ 预计 **~10:52** 收口，item#2 整体 ~10:55（与「次日 11:00–11:30」预报一致）。
- ★ **s2024 是第二次「静默失败 + 看护器自动恢复」**：`seed_merge100.out` 07:42:02 记
  「第 1/14 次结束 **rc=0，落盘完成=False**」⇒ 自动 `--resume` 起第 2/14 次，08:57:03 跑满。
  **链路实测可靠，无需人工；未达「连续两次拉起即死」停手门。**
- 错误扫描：`logs/_seed_merge100/` 7 份日志（3 train 各 ~1.5 MB + train_merge_s2024 398 KB
  + 3 eval ~5.3 KB）正则 `Traceback|MemoryError|CUDA out of memory|RuntimeError` 命中 **0**。
- 完成判据未齐（缺 `merge_s2024` 的 eval JSON、`seed_merge100_result.json` 仍 **103 B 进度占位**、
  `tta_seeds_result.json` 缺）⇒ **未追加实验结论、未重生成 `_EXPERIMENTS_SUMMARY.md`**（按纪律）。
- 未改队列源码、未起第二个训练、未跑 eval/量化/TTA；全程只读 `stat`/`grep -c`，**未 tail 训练日志**。
