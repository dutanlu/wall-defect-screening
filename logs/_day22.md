
## 二十二、★★ 训练 OOM 事故（2026-09-23 17:32）——真因是宿主应用占内存

### 1) 事故现象
`v11s640_clsbal` 在 **epoch 9 的 59%** 崩了，**只跑了 18 分钟**（非 28 分钟线）：
```
Caught error in DataLoader worker process 2.
  load_image → cv2.imdecode
cv2.error: OpenCV(5.0.0) ... Failed to allocate 1228800 bytes
           in function 'cv::OutOfMemoryError'
v11s640_clsbal failed  mAP50=None mAP50-95=None size=NoneMB
```

### 2) ★ 真因不是 workers，是宿主应用 WorkBuddy 吃内存
用 `Get-Process | Sort WorkingSet64` 实测（17:38，训练进行中）：

| 进程 | 工作集 | 私有内存 |
|---|---|---|
| **WorkBuddy (PID 36984)** | **3982 MB** | **6180 MB** |
| python（训练，PID 6968） | 2685 MB | **10445 MB** |
| WorkBuddy 其它实例 ×5 | 197~288 MB 各 | — |

**WorkBuddy 合计约 5.2 GB**；训练私有内存 10.4 GB ⇒ **15.72 GB 总内存 96~97% 占满**，
仅剩 **0.44 GB** ⇒ worker 进程 `cv2.imdecode` 申请 1.2 MB 都失败。

**⇒ 结论：`workers=4` 只是「压垮骆驼的最后一根稻草」，真因是宿主应用 + 训练争内存。**
（`workers` 从 4 改 0 仍有价值：减少 4 个派生进程，但救不了根本。）

### 3) 处置（已执行）
1. **`workers` 4 → 0**（只改 `v11s640_clsbal` 配方的 `aug` 字段，不动全局）：
   - 降低内存压力的同时，**顺带消除了 A/B 的变量差异**
     —— 基线 `v11s640` 正是 `workers=0`（它 09-21 14:11 开跑，早于提速改动）。
2. **实测验证 resume 时覆盖真的生效**（`logs/_RESUME_WORKERS_VERIFY.txt`）：
   读 `trainer.py:966-997 check_resume` —— `workers` 在可覆盖白名单里，
   `cls_pw` 不在（靠 checkpoint 沿用）。实跑证实：
   ```
   Resuming training ... from epoch 9 to 100 total epochs
   Class weights: [1.349 1.622 0.804 0.748 0.24 1.058 1.178]
   trainer 实际生效：workers=0  cls_pw=0.5  fraction=1.0  scale=0.5   ✅
   ```
3. **接力器 v3 修正两处缺陷**：
   - ❌ v2 用 `subprocess.call` 直接继承 stdout ⇒ **续训输出没落进 `_train_clsbal.log`**，
     崩溃后无从查因（本次就是靠原任务残留日志才看到 OOM 栈）。
     ⇒ 改为**显式重定向**（`open(..., 'a')` + `stdout=lf, stderr=STDOUT`）。
   - ⇒ 新增 `detect_oom()`：本块未推进时先查是否 OOM，给出可操作建议。
   - 标题里的「v2」字样未同步改为 v3（**已知小瑕疵，功能无影响**，下轮统一修）。
4. 备份：`logs/_backup_lastpt_20260923/`（last.pt + results.csv）。

### 4) 复跑结果（17:41）
- **撑过 epoch 9，已完成 9/100 轮**（`results.csv` 10 行）。
- 内存回落到可用 1.43 GB / 90% —— 仍在危险区。
- 每轮约 **118–120 秒** ⇒ 剩余 91 轮 ≈ **3.0 小时**。
- `OutOfMemory` 在日志中出现 1 次（历史那次），本次会话未新增。

### 5) ★ 未解决的风险（需用户配合）
**WorkBuddy 独占 5.2 GB 是最大可控项**，但我不能替用户关窗口。
⇒ 若再次 OOM，可选手段（按侵入性排序）：
1. **用户关掉不需要的 WorkBuddy 窗口/标签**（最有效、零副作用）
2. 降 `batch` 24 → 16（会引入与基线的第三个变量，**不推荐**，除非反复崩）
3. 设 `PYTORCH_CUDA_ALLOC_CONF`（只影响显存，对系统内存 OOM 无效）
4. 加 `cache='disk'`（换页风险，本机已有 Memory Compression 249 MB）

### 6) 教训
> **「参数改了」和「参数生效」要分开验**（已固化技能），
> 而这次新增一条：**「训练崩了」要先量内存，不要先猜 workers**。
> 我第一反应是「workers=4 导致 OOM」，但实测证据指向宿主应用 ——
> **归因必须靠 `Get-Process` 实测，不能靠直觉。**
> 另：`--resume` 时 `results.csv` 的 `time` 列**从 0 重计**，
> 直接做差会得到负数 ⇒ 算每轮耗时要用相邻两轮的绝对差值，不能跨 resume 点算。
