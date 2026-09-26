# 交付包「开箱即用」P0 修复：默认权重路径

**日期**：2026-09-25 14:46–14:54
**触发**：按 `_PACK_SYNC_PENDING_20260925b.md` §二 第 3 步，**在包自身目录内**跑入口命令
（这一步此前 4 轮审查都没做过 —— 历次只在源工程里跑，见 MEMORY §1.5b）。

---

## 一、发现的缺陷

在包目录 `外墙缺陷筛查工程包\` 内执行：

```
$ python code/evaluate.py --help          → EXIT=0（正常）
$ python code/video_screen.py --help
  [14:46:44] 模型: ...\外墙缺陷筛查工程包\results\train\v11s640\weights\best.pt
  [14:46:44] !! 模型不存在。请先训练（train.py）或用 --model 指定权重
```

**包内根本不存在** `results/train/v11s640/weights/`（实测 `No such file or directory`）。
根因两条，缺一不可：

1. **交付包刻意排除训练工作目录**。`_copy_delivery_pack.py` 按口径把
   `results/train/*/weights/` 判为训练副产物 → 不进包；成品权重被**扁平化**复制成
   `weights/v11s640_best.pt`。所以那条路径在包里**永远不存在**。
2. **入口脚本把这条件当默认值硬写**：
   - `02_code/video_screen.py:515` → `RESULT_DIR/train/v11s640/weights/best.pt`
   - `02_code/batch_screen.py:230` → 同上
   - `02_code/pipeline.py:343` → `RESULT_DIR/train/v8s640/weights/best.pt`
     ⚠️ **额外还错了模型**：主力 2026-09-20 起已切到 **v11s640**，这里还留着 **v8s640**。
   - `02_code/bench_pipeline.py:272` → 同 pipeline，也是 v8s640。

在源工程里这些都「看起来正常」，因为源侧 `04_results/train/v11s640/weights/best.pt`
**确实存在**（`best.pt` + `last.pt` 都在）。⇒ **只看源工程永远发现不了**。

**影响面**：任何按 README 克隆仓库的人，跑 `pipeline.py` / `video_screen.py` /
`batch_screen.py` / `bench_pipeline.py` 都会立刻「模型不存在」而失败。
`app.py`（Gradio）不受影响 —— 它本来就先试 `WEIGHTS_DIR/<run>_best.pt`。

---

## 二、修法（单一实现，杜绝各处漂移）

在 `02_code/common.py` 新增 **`default_weight(prefer="v11s640") -> str`**：
按优先级给一串候选，返回**第一个真实存在**的那条；全不存在时返回优先级最高者
（让调用方报「请先训练 / --model 指定」，而不是返回一个看似合理却永不存在的路径）。

四处入口**全部改为调用它**，不再各自维护候选表：

| 文件 | 改动 |
|---|---|
| `02_code/common.py` | 新增 `default_weight()`（唯一候选表） |
| `02_code/pipeline.py` | `argv_flag("model", default_weight("v11s640"))`（**同时修掉 v8s640 → v11s640**） |
| `02_code/bench_pipeline.py` | 同上（同样修掉 v8s640） |
| `02_code/video_screen.py` | `argv_flag("model", default_weight("v11s640"))` |
| `02_code/batch_screen.py` | 同上 |
| `06_deploy/app.py` | `_find_default_weight()` **改为委托** `common.default_weight()`，只做 Path 包装 |

> 为什么 app.py 也要改：三处入口各写一份候选 = 各自漂移。
> 这正是 `cross-entrypoint-claim-consistency` 的典型场景 ——
> 「系统级」的默认权重解析必须只有一个实现。

---

## 三、验收（写成一个脚本，一次跑完）

`logs/_pack_entry_test.py` —— **在包自身目录内** subprocess 跑全部入口，结论写
`logs/_PACK_ENTRY_TEST.txt`。

```
---- pipeline.py --help          EXIT=0  OK   模型: ...\外墙缺陷筛查工程包\weights\v11s640_best.pt
---- evaluate.py --help          EXIT=0  OK
---- video_screen.py --help      EXIT=0  OK   模型: ...\外墙缺陷筛查工程包\weights\v11s640_best.pt
---- batch_screen.py --help      EXIT=0  OK
---- batch_screen 真跑（3 张）    EXIT=0  OK   检出 16 处；建筑级风险粗判 U
总结论: ✅ 全部 PASS
```

**真跑结果**（不是 `--help`）：3 张 val 图 → 检出 16 处（efflorescence 14 / crack 2）
→ 建筑级汇总 GSD 中位 66.96429 mm/px（离散 0.0%, stable=True）。

### ★★ 验收自身踩的坑（已固化进脚本注释）

第一次验收**直接拿包内 `dataset/images/val`（568 张）跑** ⇒ `batch_screen` 往
**包内 `results/vis/` 写了 568 张 annotated jpg** + `results/eval/batch_screen_results.json`。
紧接着跑 `_pack_vs_src.py`：**判据2（包有源无）从 0 暴涨到 569 ⇒ FAIL**。

⇒ 两条纪律：
1. **真跑必须用临时目录**（搬 3 张进去），且跑完**显式清理**包内 `results/vis/`
   （`batch_screen` 的绘图路径**不跟随 `--json`**，光改 json 输出位置不够）。
2. **判据2 一 FAIL，第一个该怀疑的是「自己刚才的验收跑」，不是包本身。**

已在脚本内实现自动清理，并提示「清理后请再跑一次 `_pack_vs_src.py` 确认判据2=0」。

---

## 四、修复后的一致性状态

```
比对文件数：共用 5866 / 源独有 9930 / 包独有 0
【判据 1】内容不同                          = 0   ✅
【判据 2】包有源无（包内野文件）             = 0   ✅
【需复核】源有包无（不在 EXCLUDE 白名单内）   = 0   ✅
结论：✅ PASS
```
（`04_results/train/v8s640/results.csv` 属**已识别·包内保留的复现凭据**，非野文件。）

## 五、回归

- `logs/_test_abstention.py` → **9 案例 ALL PASS**
- `logs/_test_batch_screen.py` → **7 案例 ALL PASS**
- 源工程内 `pipeline.py` 真跑 → 正确取 `03_weights\v11s640_best.pt`，检出 4 处，风险 U
- `py_compile` common / pipeline / bench_pipeline / video_screen / batch_screen / app.py 全过
