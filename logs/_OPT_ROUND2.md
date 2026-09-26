# 优化收尾轮 · 记录（2026-09-26 晚）

> 承接本日已完成的：`_thin_skeleton` 活跃集版（`measure` 16.2×，逐像素等价）、
> `_skeleton_length_px` 向量化（位级等价）、bench 口径修正、视频去写盘往返、两处中文路径真 bug 修复。
> 本文记录**收尾三项**（1.1 / 1.2 / 1.3）与交付包收口（1.5）。

---

## 1.2 `segment_defect` 连通域过滤改查表 —— ✅ 逐像素等价

| 项 | 内容 |
|---|---|
| 原写法 | `for i in range(1, n): if stats[i, AREA] >= min_area: mask[labels == i] = 255`<br>⇒ 每个连通域做一次**全图**比较：O(连通域数 × ROI 像素数) |
| 新写法 | `lut = np.zeros(n, uint8); lut[keep_ids] = 255; mask = lut[labels]`<br>⇒ 单次 **O(N) 向量化** |
| **等价性要点** | 原循环是 `range(1, n)`，**标签 0（背景）从不参与** ⇒ 查表时**必须显式排除 0**（`keep_ids[keep_ids != 0]`） |
| **Gate 证据** | `logs/_lut_filter_equiv.txt`：**A/B 指纹比对**（旧版 measure.py 生成参考 → 新版比对）<br>**38 个 (ROI, 类别) 组合的掩膜 sha256 逐一相同**；语料含真实图、大图裁切、纯色、渐变、椒盐噪点、单大斑、80 个小连通域、32×32、8×1200 |
| **实测收益** | 真实 ROI 718×1404 / crack：**11.385 → 8.505 ms**（约省 2.9 ms）<br>同 ROI / spalling：26.254 → 26.547 ms（**无变化** —— 该路被 meddev+medianBlur 主导）<br>小 ROI：无变化 |
| 结论 | **保留**。收益小但为真，且逐像素等价、零风险 |

留痕：`logs/_patch_lut_filter.py`、`logs/_verify_lut_filter.py`、`logs/_lut_filter_ref.json`

---

## 1.3 `06_deploy/app.py` 去重 —— ✅ payload 逐字节一致

### 改了三处
1. `assess_interpretability` **4 次 → 2 次**：
   `:257/258` 算出 `interp_crack(0.30)` / `interp_blob(10.0)`，
   而 `:301/302` 又用**逐一相同**的参数算了 `crack_interp` / `blob_interp` ⇒ 纯重复计算。
2. `:289` 的 `risk = building_risk_level(grades)` 是**死代码**：
   `risk` 的所有读取（381/415/521/527/529/531/532/535/708）**全部在 `:348` 的
   `risk, n_void = apply_abstention(...)` 重新赋值之后**，289~348 之间零读取。
3. 随之 `building_risk_level` 在 app.py 中不再被调用 ⇒ 从 import 行移除（保留注释说明）。

### 双重证据
**(a) 按构造可证**（实测）：
- `assess_interpretability` 同参数两次调用 `to_dict()` **完全相同**，且调用后 `calib.mm_per_px` 未被改 ⇒ **纯函数**
- `building_risk_level` 两次调用结果相同，且调用后 `grades` **未被修改** ⇒ **纯函数、无副作用**

**(b) 实测 A/B（最硬）**：
用同一张测试图、同一组参数，分别在**旧版**与**新版** app.py 下调用 `analyze()`，
序列化 payload 逐字节比对 ⇒ **14,671 B 完全一致**（`logs/_app_dedup_ref.json` vs `_tmp_app_new.json`）。

> **做法上的一个改进（值得复用）**：第一次尝试用「换 app.py → 跑 → 换回」，中途被 SIGTERM
> 打断，`app.py` 一度停在旧版（已靠备份还原）。
> 第二次改为**把旧版复制成同目录下的临时模块 `06_deploy/_app_old_tmp.py` 再导入** ——
> 同目录 ⇒ `Path(__file__)` 解析行为一致，且**全程不动 app.py**，彻底消除"忘了换回"的风险。
> 跑完已删除该临时模块（`app.py` sha256 全程保持 `10e9f7e0…`）。

留痕：`logs/_patch_app_dedup.py`、`logs/_verify_app_dedup.py`、`logs/_app_dedup_ref.json`

---

## 1.1 标注图改可选落盘 —— ✅ 已实测验证

`02_code/pipeline.py` 新增 `--save-annotated`（**默认关**）；未落盘时结果里的 `annotated` 置 `None`
（避免下游 JSON 残留指向空文件的引用 —— `video_screen.py:498` 会原样透传该字段）。
`07_report/演示视频脚本.md` 的「可视化输出」行已补口径说明。

**冒烟测试（2 张真实测试图的对照）**：

| 模式 | 产物目录 | JSON 的 `annotated` | `n_detections` |
|---|---|---|---|
| 默认（无开关） | **空** | `None` | 6 / 14 |
| `--save-annotated` | 2 张 `*_annotated.jpg` | 真实路径 | 6 / 14 |

⇒ **默认不落盘生效**；且两模式的**检出数完全一致**（6 / 14）⇒ **零数值变化**。

> ⚠️ **冒烟测试必须用临时小目录**：本次用了 `logs/_tmp_annot_test/`（含 2 张图 + 独立 out 目录），
> 已清理。**教训来源**：本日早些时候用 `--limit=3`（该 flag 并不存在）跑 `pipeline.py --dir`，
> 结果把 `04_results/vis` 的 **165 张全跑并覆盖了 `pipeline_results.json`**（21 KB → 1.86 MB），
> 已从交付包还原。**跑前必须确认 flag 真实存在，并显式指定小输入目录与临时输出目录。**

留痕：`logs/_patch_save_annotated.py`

---

## 1.5 交付包收口 —— 见下节

`measure.py`（阶段 1 + 1.2）、`pipeline.py`（1.1）、`app.py`（1.3）、`stitch_facade.py` 与
`_survey_demo.py`（中文路径修复）、`技术报告.md` / `答辩材料.md` / `演示视频脚本.md`（文档同步）
均已变 ⇒ 重跑 `logs/_copy_delivery_pack.py` → `logs/_pack_vs_src.py`，目标 **PASS**。

---

## 一条反复出现的纪律（本轮第三次踩）

**多行替换串写成了 `\n`，而目标文件是纯 CRLF ⇒ 插入裸 LF、行尾被破坏。**
三次分别发生在 `bench_pipeline.py`、`技术报告.md`、`app.py` 相关补丁（均在补丁脚本自检里被拦下，
且都因"改前先备份"可无损还原）。

⇒ 落成硬规矩：**写多行替换串前先量目标文件行尾，并按该行尾逐行显式书写；补丁脚本必须自带行尾断言。**
