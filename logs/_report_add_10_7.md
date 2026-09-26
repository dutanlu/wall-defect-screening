### 10.7 交付包入口的布局感知修复（2026-09-26 新增，**P0**）

#### 10.7.1 现象：交付包的入口**根本跑不起来**

在**交付包自身目录内**按 §10.4 的方式实跑：

```
cd <工程包>/deploy && python app.py
ModuleNotFoundError: No module named 'common'
```

这直接违反了本项目自定的 P0 纪律 ——
**交付包必须在「包自身目录内」跑一次入口命令**。

#### 10.7.2 根因：交付包的目录名与源码**不同名**

| 源码 | 交付包 |
|---|---|
| `02_code` | **`code`** |
| `06_deploy` | `deploy` |
| `04_results` | `results` |
| `05_quantify_grade` | `grading_rules` |
| `07_report` | `report` |
| `03_weights` | `weights` |
| `01_data/dataset` | `dataset` |
| `08_photo_collector` | `photo_collector` |

而 `app.py` 里写的是 `CODE_DIR = PROJ_ROOT / "02_code"`；
包内 `run_app.bat` 只执行 `"%PY%" app.py`，**没有设 `PYTHONPATH`** ⇒ 直接崩。

⇒ **包内任何硬编码源码目录名的相对路径都会断。**

#### 10.7.3 修法（最小、纯兼容）

```python
CODE_DIR = PROJ_ROOT / "02_code"
if not CODE_DIR.exists():
    CODE_DIR = PROJ_ROOT / "code"      # 交付包布局
```

**源码侧行为完全不变**（实测 `CODE_DIR = 02_code`）；
**包内侧实测**：`CODE_DIR = code`、导入成功、
`_i3_paths()`（§8.5）正确落到 `results/eval/{conformal_measure,decision_rule}.json` 并读到内容、
区间与决策计算正确。

> 这是纯**路径引导**、**不改可见界面** ⇒ **不触发**「必须重录实机视频」那条纪律。

#### 10.7.4 ★ 这处缺陷**一直存在**，此前没被发现的原因值得记录

- **不是本轮引入的**：改动前的两份备份
  （`logs/_opt_baseline_20260926/06_deploy__app.py.before_dedup` 与 `…before_i3`）
  里该行**都是** `02_code`；
- **为什么没被发现**：日常开发都在**源码工程**里跑（那里 `02_code` 存在），
  于是这个只在**交付包布局**下才暴露的问题，一直没被触发。
  ⇒ 这正是「**包内实跑入口命令**」这条纪律存在的理由 ——
  它是一条**只在"换个目录跑一遍"时才会亮**的检查，缺了它，问题可以一直潜伏。

#### 10.7.5 附：本轮新增实验脚本的路径边界（如实记录，未修）

本轮新增的探针 / 实验脚本（`_p1_train_probe.py`、`_p2_conformal_measure.py`、
`_p3_decision.py`、`_bfdd_*.py` 等）**硬编码了源码工程路径**，
因为它们依赖 `01_data/raw/_public_datasets/bfdd/_extract/`（548 MB，**不在交付包内**，
也不在开源仓库内）。

⇒ 这些脚本**只能在源码工作区运行**；
交付侧可独立复算的是 `_bfdd_repro_check.py`（§7.16.5）——
它**只读官方 tar.gz、自包含、不需要本工程**。
此边界在此写明，未强行"修"（修它需要把 548 MB 数据一起分发，既无必要也不符合许可）。
