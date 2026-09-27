# -*- coding: utf-8 -*-
r"""单图 OOD 兜底：**一张图**进来看不出「模型在此数据上已失效」时，用什么判据把它拦下。

================================================================================
为什么单独一个模块（与 `grade.py` 平行的「单一共享实现」）
================================================================================
`batch_screen.py` 已实现**批级**门 `p_hit`（`p_hit = crack 检出中 conf≥0.25 的比例`，
分母在 conf≥0.05 上统计）。批级门的适用前提是「**一批同源图**」——
样本量足够，比例才稳定。

**单图没有这个前提**：n=1 时比例只能取 {0, 1} 两个值，**分母为 1 ⇒ 判据退化**。
这不是"实现不够好"，是数学上就没有信息量。故**不能**把批级门直接套到单图。

⇒ 本模块提供**单图可用的替代判据**，并**诚实标注每条判据的能力边界**。

================================================================================
单图到底还能看到什么（这才是设计的出发点）
================================================================================
用**一张图**能拿到的、与域外失效相关的量，只有「**这张图自身的检出统计**」：

  ① **有没有检出**（任一类别）          —— 域外可能完全无响应
  ② **检出的最高置信度** `conf_max`      —— 域外置信度整体塌陷
  ③ **检出置信度的中位数** `conf_med`    —— 同上（n≥1 时即可算）
  ④ **达标比例** `p_hit_local`          —— 单图版，**n 小时噪声大，须标注**
  ⑤ **检出数量** `n_det`                —— 域外产出率塌到 1/5

★ 关键设计（决定了"能不能叫 OOD 判据"）：

  **逐条判据都必须只在「本图内部」可计算，且不得依赖批次。**
  凡是需要"与同批其他图比"的量（如比例、分位），一律**降级为辅助证据**，
  并**显式标注 n 小时不可信**。

================================================================================
判据与阈值（**来自实测，不是拍的**）
================================================================================
来源：`logs/_BFDD_I1_结论.md` §二(3) 与 `logs/_bfdd_abstainer_ood.json`。
两侧在**同一 conf floor（0.05）**上测得：

| 域            | crack 检出数 | 中位 conf | conf≥0.25 比例 |
|---------------|-------------:|----------:|---------------:|
| V3 域内(285图)| 94           | **0.466** | **67.0%**      |
| BFDD 跨源(152)| 158          | **0.094** | **4.4%**       |

⇒ **中位置信度**是**单图可算**且两侧差 **5×** 的量 ⇒ 本模块的**主判据**。

⚠️ 但**阈值不能直接抄 0.094 / 0.466**：那是"整批中位数"，单图只有 1 个数。
故本模块的做法是：

  1. **主判据 `p_hit_local`**（本图内部比例）——阈值取**批级门同款 0.5957**
     （V3 bootstrap 5% 分位），因为它就是同一判据在 n=1 时的退化形式，
     阈值语义一致；**但当 n_det < 3 时不下判定**（样本太少，比例无意义）。
  2. **辅判据 `conf_med` / `conf_max`**——给出**参考带**（V3 观测带），
     落在带外 ⇒ 记一个"疑似域外"票，但**不单独据此拒答**。
  3. **兜底判据 `n_det == 0`**——**零检出**。这张图**什么都没检出**时，
     系统**必须**说「本图无检出」，**绝不能**顺推为「本立面安全」
     （这正是 §9 主线：把「判不了」说成「没问题」更危险）。

================================================================================
兜底行为（降级策略，三级）
================================================================================
  **level 0 NORMAL**    —— 未见异常，正常出结论。
  **level 1 SUSPECT**   —— 疑似域外：**保留**定量结论，但**强制追加警示**，
                            并把 `risk.level` **降一档置信**（附 `ood_warning`）。
  **level 2 ABSTAIN**   —— 强证据域外 / 零检出：**作废定量结论**，
                            风险等级输出 **U（无法判定）**，并**明确要求人工复核**。

★ 「降级」不是"静默改数字"，而是**把不确定性显式地加到输出上**。

★ **票分强弱两档**（这是实测校准出来的，不是设计美感）：
  为压制域内误触发，`conf_med` 单票**不足以**定 level 1；
  需 `conf_max` 弱票（两侧差得最干净：V3 2/60 vs BFDD 29/60）或两票互佐。
  校准确认见 `logs/_SINGLE_OOD_结论.md` §四。

================================================================================
本模块能说什么、不能说什么（**必须随结论声明**）
================================================================================
能说：
  - 单图**可以**给出「疑似域外」的**弱信号**，且**零检出**是**强信号**；
  - 阈值取自**实测的 V3 分布**，不是主观选择。

不能说：
  - 不能说这是"OOD 检测器"（没有 AUROC、没有留出集校准）；
  - 不能说单图判据与批级门等价（**粒度不同、信息量不同**）；
  - 不能说"level 0 ⇒ 一定在域内"（**只在域内校准过，没有域外真阴性集**）。
"""
from __future__ import annotations

import statistics
from typing import Any

# ---------------------------------------------------------------------------
# 阈值常量（**单一来源**：与 batch_screen.py 的批级门共享同一语义）
# ---------------------------------------------------------------------------

#: 与 `batch_screen.OOD_THRESHOLD_DEFAULT` 同源：V3 bootstrap 5% 分位。
#: 这里作为**单图版**的默认阈值（n 足够时才有判别力，见 `min_n_for_ratio`）。
OOD_THRESHOLD_SINGLE = 0.5957

#: 统计 p_hit 的工作阈值（与批级门的 C_HIT 一致）。
OOD_C_HIT = 0.25

#: 统计 p_hit 时用的低 conf floor（分母必须在 floor 上统计，否则恒为 1）。
OOD_CONF_FLOOR = 0.05

#: 单图 p_hit 的**最小检出数**：低于此不下判定（比例在 n<3 时噪声过大）。
OOD_MIN_N_FOR_RATIO = 3

#: 域内（V3）**逐图**中位置信度的下带。
#:
#: ⚠️ **不要误用批级数字**：I1 报的「V3 中位 conf = 0.466」是**整批 94 个检出的中位数**，
#:    与「**逐图**中位数」不是同一个量（逐图 n 小，天然更低、更散）。
#:    本阈值由**实测重标定**得到：在 V3 test 60 图 / BFDD val 60 图上扫描
#:    （`logs/_SINGLE_OOD_结论.md` §四 给出扫描表）：
#:
#:      阈值    V3 命中   BFDD 命中   净增益
#:      0.08     3.3%      13.7%     +10.4pp
#:      0.10    10.0%      37.3%     +27.3pp   ← 取此（误触发最低档位里拦截最高）
#:      0.12    15.0%      49.0%     +34.0pp
#:      0.15    23.3%      70.6%     +47.3pp
#:      0.20    36.7%      88.2%     +51.6pp   ← 第一版误用了这个，V3 误触发过高
#:
#:    ⇒ 取 **0.10**：V3 误触发 **10%**，同时拦住 **37%** 的 BFDD 图。
#:    （第一版取 0.20 ⇒ V3 误触发 36.7%，**过高**；已按实测改回。）
OOD_CONF_MED_SUSPECT = 0.10
#: 最高置信度都够不到工作阈值 ⇒ 记一票（这个量在两侧差得最干净：
#: V3 仅 2/60 命中，BFDD 29/60 命中 ⇒ 单票即可升 SUSPECT）。
OOD_CONF_MAX_SUSPECT = 0.25

#: 默认检测类别（与批级门一致）。
OOD_CLASS_DEFAULT = "crack"


def _conf_stats(boxes, cls_filter: str | None) -> dict:
    """从 ultralytics 的 `boxes` 里抽出（可选的）某一类的置信度列表。"""
    import numpy as np
    from common import CLASSES

    if boxes is None or len(boxes) == 0:
        return {"n": 0, "confs": []}

    clsid = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy().astype(float)
    out: list[float] = []
    for i in range(len(clsid)):
        name = CLASSES[clsid[i]] if clsid[i] < len(CLASSES) else str(clsid[i])
        if cls_filter and name != cls_filter:
            continue
        out.append(float(confs[i]))
    return {"n": len(out), "confs": out}


def single_image_ood(model, img, cls: str | None = OOD_CLASS_DEFAULT, *,
                     floor: float = OOD_CONF_FLOOR, c_hit: float = OOD_C_HIT,
                     imgsz: int = 640) -> dict:
    """对**单张图**跑一次低阈值检测，产出 OOD 判据所需的全部统计量。

    参数
    ----
    model : ultralytics YOLO
    img   : np.ndarray  (BGR/RGB 均可，由 model.predict 自行处理)
    cls   : 类别相关的判据（p_hit）用哪一类；None ⇒ 不分类
    floor : 低阈值，用于拿真实分母（默认 0.05）
    c_hit : 工作阈值，用于算达标比例（默认 0.25）

    返回
    ----
    dict：
        n_det      : 该 conf floor 下**指定类**的检出数
        n_hit      : 其中 conf ≥ c_hit 的个数
        p_hit_local: n_hit / n_det（**n_det == 0 时为 None**）
        all_n      : **不限类别**的检出总数
        conf_max   : **不限类别**的最高置信度（无检出为 None）
        conf_med   : **不限类别**的中位置信度（无检出为 None）

    ⚠️⚠️ **一个必须写清的实测教训（本轮实际踩到并修正）**：
      第一版把 `conf_max` / `conf_med` 也算成**指定类（crack）**的统计量。
      实测后果：BFDD 有 **43/60 张图 crack 类 0 检出** ⇒ 这两个量全是 `None`
      ⇒ 「置信度塌陷」这条判据**静默失效**，**34 张 BFDD 图被漏放过**（误判 NORMAL）。
      这是典型的「**指标选错 ⇒ 判据静默退化**」，比崩溃更隐蔽。
      ⇒ 修正：**置信度幅度类判据一律用「不限类别」统计**（模型对整张图的响应强度），
        只有**类别相关**的判据（p_hit）才限定到 crack。
    """
    res = model.predict(img, conf=float(floor), imgsz=int(imgsz), verbose=False)
    if not res:
        return {"n_det": 0, "n_hit": 0, "p_hit_local": None,
                "conf_max": None, "conf_med": None, "all_n": 0}

    r0 = res[0]
    boxes = getattr(r0, "boxes", None)

    # ① 类别相关的统计（p_hit 只对指定类有意义）
    st_cls = _conf_stats(boxes, cls)
    confs_cls = st_cls["confs"]
    n_det = len(confs_cls)
    n_hit = sum(1 for c in confs_cls if c >= float(c_hit))

    # ② ★ 不限类别的统计（置信度幅度判据必须用它，见 docstring 的实测教训）
    st_all = _conf_stats(boxes, None)
    confs_all = st_all["confs"]
    all_n = len(confs_all)

    return {
        "n_det": n_det,
        "n_hit": n_hit,
        "p_hit_local": (n_hit / n_det) if n_det > 0 else None,
        "conf_max": (max(confs_all) if all_n else None),
        "conf_med": (statistics.median(confs_all) if all_n else None),
        "all_n": all_n,
    }


def decide_single(stats: dict, *, n_measurements: int | None = None) -> dict:
    """把单图统计量判成三级降级结论。**纯函数**，便于单测。

    参数
    ----
    stats          : `single_image_ood(...)` 的返回
    n_measurements : 主链路最终产出的测量实例数（可选）。
                     用于区分「模型有响应但测量层无产出」这一种情形。

    返回
    ----
    dict：
        level      : 0 NORMAL / 1 SUSPECT / 2 ABSTAIN
        verdict    : 人类可读的一句话结论
        votes      : 命中的辅助判据列表（可审计）
        reasons    : 触发原因列表
        evidence   : 原始统计量（供写进 JSON，便于复核）
        caveats    : 本条结论**必须随附**的声明
    """
    n_det = int(stats.get("n_det") or 0)
    all_n = int(stats.get("all_n") or 0)
    p = stats.get("p_hit_local")
    cmax = stats.get("conf_max")
    cmed = stats.get("conf_med")

    votes: list[str] = []
    strong_votes: list[str] = []
    reasons: list[str] = []

    # ---- 强判据（单独即可 ABSTAIN）：本图**零检出** ----
    # 注意：用 all_n（不限类别）来判，避免"只是因为该类没检出"的假阳性。
    zero_det = (all_n == 0)

    # ---- 判据票 ----
    #
    # ★ 票分两档（**来自实测的校准结论**，不是拍脑袋）：
    #   `conf_max < 0.25`  —— **强票**：V3 仅 2/60 命中 vs BFDD 29/60 ⇒ 单票即可 SUSPECT；
    #   `conf_med < 0.10`  —— **弱票**：V3 10% vs BFDD 37% ⇒ **单票不足以 SUSPECT**
    #                        （否则 V3 误触发过高：实测单用它会误判 19/60 = 32%）。
    #   `p_hit_local`      —— 弱票（n 足够时才有判别力）。
    if cmax is not None and cmax < OOD_CONF_MAX_SUSPECT:
        strong_votes.append(f"conf_max {cmax:.4f} < {OOD_CONF_MAX_SUSPECT}")
    if cmed is not None and cmed < OOD_CONF_MED_SUSPECT:
        votes.append(f"conf_med {cmed:.4f} < {OOD_CONF_MED_SUSPECT}（弱票）")
    ratio_usable = (n_det >= OOD_MIN_N_FOR_RATIO)
    if ratio_usable and p is not None and p < OOD_THRESHOLD_SINGLE:
        votes.append(f"p_hit_local {p:.4f} < {OOD_THRESHOLD_SINGLE} (n={n_det})（弱票）")

    n_all_votes = len(votes) + len(strong_votes)

    # ---- 定级 ----
    if zero_det:
        level = 2
        reasons.append("本图在 conf≥%.2f 下**零检出**（不限类别）" % OOD_CONF_FLOOR)
        verdict = ("本图未检出任何目标 ⇒ **不能判读**。"
                   "注意：这**不等于**「该立面安全」——"
                   "零检出既可能是真的无缺陷，也可能是模型在该数据上失效，"
                   "单张图无法区分，**必须补充同源多图或改用已知合格的拍摄条件**。")
    elif strong_votes and n_all_votes >= 2:
        # 强票 + 任一其它票 ⇒ 强证据
        level = 2
        reasons.extend(strong_votes + votes)
        verdict = ("命中**强判据**且另有佐证 ⇒ 判定为**疑似分布外**，"
                   "定量结论**不予采信**，风险等级输出 **U（无法判定）**。")
    elif len(votes) >= 2:
        # 两条弱票互相佐证
        level = 2
        reasons.extend(votes)
        verdict = ("两条**弱判据**同时命中（互相佐证）⇒ 判定为**疑似分布外**，"
                   "定量结论**不予采信**，风险等级输出 **U（无法判定）**。")
    elif strong_votes:
        level = 1
        reasons.extend(strong_votes)
        verdict = ("命中**强判据**（最高置信度低于工作阈值）⇒ **疑似分布外**，"
                   "定量结论**仍给出但须附警示**，风险等级须按「降一档置信」理解。")
    elif len(votes) == 1:
        level = 1
        reasons.extend(votes)
        verdict = ("命中一条**弱判据** ⇒ **疑似分布外（弱）**，"
                   "定量结论**仍给出但须附警示**，风险等级须按「降一档置信」理解。")
    else:
        level = 0
        verdict = "未命中任何域外判据（**仅在域内校准过**，不构成「一定在域内」的证明）。"

    # ---- 与测量层产出的交叉检查（只增不减严格度）----
    if (n_measurements is not None and n_measurements == 0
            and level == 0 and not zero_det):
        level = 1
        reasons.append("检出存在但测量层产出为 0（可能几何/质量门拦下）")
        verdict += " 另：本图有检出但测量层无产出，已升为疑似。"

    caveats = [
        "单图判据**不是** OOD 检测器：无 AUROC、无留出集校准。",
        "阈值取自 V3 域内分布，**只在域内校准过**；无域外真阴性集 ⇒ "
        "level 0 **不能**读成「一定在域内」。",
        f"比例型判据在 n_det < {OOD_MIN_N_FOR_RATIO} 时**不可用**（单图 n 天然小）；"
        "已自动跳过，不代表该判据通过。",
        "本判据与批级门 `p_hit` **粒度不同、信息量不同**，不可互相替代。",
    ]

    return {
        "level": level,
        "level_name": {0: "NORMAL", 1: "SUSPECT", 2: "ABSTAIN"}[level],
        "verdict": verdict,
        "votes": votes,
        "strong_votes": strong_votes,
        "reasons": reasons,
        "ratio_usable": ratio_usable,
        "evidence": {
            "n_det": n_det,
            "n_hit": stats.get("n_hit"),
            "all_n": all_n,
            "p_hit_local": p,
            "conf_max": cmax,
            "conf_med": cmed,
            "conf_floor": OOD_CONF_FLOOR,
            "c_hit": OOD_C_HIT,
            "threshold_single": OOD_THRESHOLD_SINGLE,
        },
        "caveats": caveats,
    }


def summarize_ood_for_display(decision: dict) -> str:
    """把判决渲染成一段可直接贴进界面/CLI 的说明（Markdown）。"""
    d = decision
    emoji = {0: "✅", 1: "⚠️", 2: "⛔"}[d["level"]]
    lines = [
        "### %s 单图分布外（OOD）兜底：**%s**" % (emoji, d["level_name"]),
        "",
        d["verdict"],
        "",
    ]
    if d["reasons"]:
        lines.append("**命中原因**：")
        for r in d["reasons"]:
            lines.append("- %s" % r)
        lines.append("")
    ev = d["evidence"]
    lines.append("**证据**（conf floor %.2f / 工作阈值 %.2f）："
                 % (ev["conf_floor"], ev["c_hit"]))
    lines.append("")
    lines.append("| 量 | 值 |")
    lines.append("|---|---|")
    lines.append("| 检出总数（本类） | %s |" % ev["n_det"])
    lines.append("| 检出总数（不限类） | %s |" % ev["all_n"])
    lines.append("| 达标数（≥%.2f） | %s |" % (ev["c_hit"], ev["n_hit"]))
    lines.append("| 本图达标比例 | %s |"
                 % ("—（n 不足，不适用）" if not d["ratio_usable"]
                    else "%.4f" % ev["p_hit_local"]))
    lines.append("| 最高置信度 | %s |"
                 % ("—" if ev["conf_max"] is None else "%.4f" % ev["conf_max"]))
    lines.append("| 中位置信度 | %s |"
                 % ("—" if ev["conf_med"] is None else "%.4f" % ev["conf_med"]))
    lines.append("")
    if d["level"] >= 1:
        lines.append("> **边界声明（必须随结论呈现）**：")
        for c in d["caveats"]:
            lines.append("> - %s" % c)
    return "\n".join(lines)
