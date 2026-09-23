
## 二十六、核「脚本/固定值同步」三项 + 修复第三幕命令崩溃（2026-09-23 18:2x–18:5x）

### 1) ★★★ 真问题：演示视频脚本「第三幕」的命令会当场崩溃

**现象**：`07_report/演示视频脚本.md` 第三幕（[3:05–4:00] 量化负结果）给的检查命令
引用了 `04_results/quant/quant_report.json`，而该文件的真实内容是：
```json
{ "results": [ { "name":"v11s640", "status":"quantize_failed",
                 "error":"MemoryError: bad allocation", "attempts":[150,128,96,64] } ] }
```
—— **根本没有** `onnx_int8_mb` / `compression_vs_onnx_fp32` / `latency_onnx_int8` / `verify_detects`
任何一个字段（那是 **v11s640 的量化失败记录**）。

**实测复现**：命令跑出 `FP32 ONNX: 36.18 MB` 后立刻
`KeyError: 'onnx_int8_mb'` **崩掉** ⇒ **第三幕根本拍不下去**。

**根因**：抢救试验的产物**故意不写** `04_results/quant/`（技术报告 §8 明写
「不写 `04_results/quant/`，正式产物零污染」），而是落在 `logs/_quant_rescue/`。
脚本却引用了被保留为「失败记录」的那份。

**正确数据源**：`logs/_quant_rescue/rescue_results.json`（4 档组合：
`calib_n` 32/16 × `method` MinMax/Entropy，全部 `int8_mb=9.54`、
`int8_maxconf=0.0`、`int8_boxes=0`、`usable=false`）—— 与旁白
「校准集 32 张和 16 张、MinMax 和 Entropy 四种组合，体积都是 9.54 兆，检出都是 0」**逐条对应**。

**修复**：`logs/_fix_demo_quant_cmd.py`（字节级替换、保持 CRLF），已备份。
新命令实测输出：
```
张数     算法        INT8体积     FP32最大置信      INT8最大置信      FP32框数    INT8框数    可用
32     MinMax    9.54 MB    0.9136        0.0           143       0         False
32     Entropy   9.54 MB    0.9136        0.0           143       0         False
16     MinMax    9.54 MB    0.9136        0.0           143       0         False
16     Entropy   9.54 MB    0.9136        0.0           143       0         False
```

**顺带加固编码**：本机 Python 已是 UTF-8 输出（`sys.flags.utf8_mode=1`），
但**终端代码页**不确定 ⇒ 命令前置 `chcp 65001 > $null` 并加 `-X utf8`，
使演示时中文表头不会乱码（PowerShell 重定向到文件时出现乱码正是这个原因）。
已验证 `chcp_ok=True py_ok=True`。

### 2) `_record_demo.py` 幕3 浮标：核对一致 ✅
`app.py:538-540` 的等级文字与录屏脚本的等待判据**逐条相同**：
`[A] 安全` / `[B] 个别危险点` / `[U] 无法判定`。
（三幕：幕1 默认→`[U]`；幕2 标定 420px→`[A]`；幕3 换空鼓图→`[B]`）

### 3) `_train_aug` 相关：产物链齐全 ✅
做了**全产物交叉核对**（RECIPE × train 目录 × 03_weights × test_metrics × 两个 summary）：

| run | RECIPE | train | weights | test_metrics |
|---|---|---|---|---|
| v11s640 / v11s640_aug / v8n640 / v8s1024 / v8s640 | Y | Y | Y | Y |
| v11s640_clsbal | Y | Y | —（训练中） | —（训练中） |

⇒ **5 个已完成模型产物链完整**。
两个 `*_summary.json` 只含 1 个 run，是**「本次批次覆盖式汇总」的设计**（非缺口）：
- `train_summary.json` 现在含 `v11s640_clsbal`（正在训练的批次）
- `test_eval_summary.json` 含 `v11s640_aug`（最后一次评估批次）

**增强实验的呈现**已覆盖三处：技术报告 §5.4 / 答辩材料 L160 / PPT 第 15 页；
仅两个 README 未提（README 是项目说明，不必列所有实验）。
> ⚠️ 我的核对脚本首版有个 bug：`p.parent.name` 应为 `p.name`，
> 导致 `train` 列全显 `-`（假缺口）。**核对脚本自身也要复核**。

### 4) ★★ 修正 automation 的一个路径隐患
`02_code/evaluate.py` 的 `--weights` 按**当前工作目录**解析相对路径。
我原先在 automation 里写的 `--weights=..\03_weights\...` 只有在 `02_code/` 下跑才对；
在项目根目录跑会指向 `创新题\03_weights`（实测确认该目录不存在）⇒ **会失败**。

**处置**：automation 已更新为「**绝对路径 + 整个参数用双引号包住**」，
实测从无关目录调用也能正确解析 ✅；并写入「先查 `03_weights/` 成品、
缺失则回退 `04_results/train/.../weights/best.pt`」的 fallback
（因为 train.py 只在训练**正常结束**时归档 best.pt，被中断则不归档）。

### 5) ★★★ 重要发现：接力器启动的训练**不受 28 分钟线限制**
实测：17:37 由 `subprocess.call` 启动的 `train.py --resume` **连续跑了 71 分钟未被杀**
（对比：我直接发起的工具任务约 28 分钟被终止）。
⇒ **28 分钟限制只作用于宿主直接发起的任务**，不影响接力器拉起的子进程。
⇒ 结论：本次训练**很可能一次跑完 100 轮**，不必反复接力。

### 6) 训练现状（18:48）
- 已完成 **46/100 轮**，内存 **4.01 GB 可用 / 74%**（用户关窗口后从 1.41 GB 大幅改善）。
- 速率约 **115 秒/轮** ⇒ 剩余 54 轮 ≈ 1.7 小时 ⇒ 预计 **20:30 前后完成**，
  与 automation 的 21:00 检查时间吻合。
