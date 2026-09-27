# -*- coding: utf-8 -*-
r"""同对象两次拍摄的**变化检测**（方向3 的可落地部分）。

（2026-09-27 新增）

--------------------------------------------------------------------------
为什么是「变化检测」而不是「演化预测」
--------------------------------------------------------------------------
报告 §9.4 已给出边界声明：本工程内**不存在任何跨时间的纵向重复观测**
（无拍摄时间字段、图片 0 EXIF、同源多图皆为同批连拍、收集站投稿 0 条），
而 BFDD 的 4 个航测日经 dHash 跨天配对实测（§9.4.1）证明
**拍的是 4 面不同的墙**。⇒ 以现有数据，**演化预测在物理上做不到**。

因此本模块**只做两件现有数据真正支撑得起的事**：

  1. **配对（pairing）**：给定两次拍摄，判断它们是否在拍**同一面墙的同一块区域**。
     判据是「跨时相可对齐」——不要求像素级相同（光照/曝光会变），
     只要求**结构上可对应**（用 dHash 汉明距离 + 可选的单应配准做粗判）。
  2. **变化度量（change metrics）**：配对成立后，给出**可复算**的变化量：
     同一缺陷的宽度变化、缺陷数量的增减、新增/消失的缺陷区域。

  ⚠️ **绝不输出**「扩展速率 / 剩余寿命 / 何时复检」——
     那需要 ≥3 个时间点 + 缺陷机理，本模块**没有**这个能力，
     代码里也不留任何看起来像预测的字段（防止下游误用）。

--------------------------------------------------------------------------
诚实性设计（关键）
--------------------------------------------------------------------------
· `pair_ok=False` 时，函数**返回 None 而不是硬算一个变化量** ——
  「配不上就不给结论」，与主线弃权机制同构。
· 所有输出都带 `evidence`（实际用到的距离/重叠率数值），可被第三方复算。
· 结果里**显式写入** `is_prediction=False` 与 `requires_timepoints`，
  防止有人把「变化检测」当成「预测」引用。

--------------------------------------------------------------------------
用法
--------------------------------------------------------------------------
  from change_detect import pair_and_diff, ChangePolicy
  r = pair_and_diff(img_t1, img_t2, dets_t1, dets_t2)
  if r["pair_ok"]:
      print(r["summary"])
  else:
      print("两次拍摄配不上：", r["pair_reason"])

  # 单应配准（可选，需要两侧有足够的可靠特征点）
  r = pair_and_diff(a, b, d1, d2, use_homography=True)

自检：`python logs/_verify_change_detect.py`（合成对 + 反例，见该脚本）
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ---------------------------------------------------------------------------
# 判据参数（全部集中在此，便于报告引用与第三方复核）
# ---------------------------------------------------------------------------
@dataclass
class ChangePolicy:
    """配对与变化判定的全部阈值。

    每个阈值都注明**来源**（实测 / 保守设定），不得凭感觉改。
    """

    # --- 配对判据 ---
    # dHash 汉明距离上界：超过即认为「不是同一面墙」。
    # 来源：§9.4.1 实测「同天连拍」最小距离 8~14、跨天 11~13；
    #   本值是「近重复」区间（≤6）的**放宽**版，用于容许光照/曝光差异，
    #   仍远低于「不同立面」实测的 11~13。
    max_dhash_hamming: int = 10
    # 单应配准的最小内点率（用 ORB 匹配时）。低于此值视为配准不可信。
    min_inlier_ratio: float = 0.30

    # --- 变化判据 ---
    # 区域重叠率（IoU）低于此值，视为「不同缺陷实例」，不配对。
    min_iou_match: float = 0.30
    # 像素宽度变化的**最小可辨幅度**（相对）。小于它只报「无明显变化」。
    # 来源：依赖 GSD；此处用相对量以便跨标定通用。
    min_rel_width_change: float = 0.20
    # 判读可用的最小 GSD（mm/px）。>此值只报「有/无变化」，不报毫米变化量。
    gsd_measurable_max: float = 0.10

    # --- 反预测护栏（写死，不提供修改入口）---
    is_prediction: bool = False
    requires_timepoints: int = 3


DEFAULT_POLICY = ChangePolicy()


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def dhash(img, hash_size: int = 8) -> int:
    """8×8 差分哈希，与 §9.4.1 的实测口径**一致**（同尺寸、同差分方向）。

    为什么自己实现而不引外部库：报告里的 11~13 / 8~14 就是用这套
    (resize → 灰度 → 水平相邻差分 → 摊平成 64 位) 算出来的，
    换实现会导致阈值不可比。
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (hash_size + 1, hash_size),
                       interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def hamming2(a: int, b: int) -> int:
    """两个整数的汉明距离（Python 3.8+ 的 int.bit_count 在 3.13 可用）。"""
    return int((a ^ b).bit_count())


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / ua if ua > 0 else 0.0


def _center(bx):
    return ((bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0)


# ---------------------------------------------------------------------------
# 配对
# ---------------------------------------------------------------------------
def check_pairing(img_t1, img_t2, policy: ChangePolicy = DEFAULT_POLICY,
                  *, use_homography: bool = False) -> dict:
    """判断两张图是否在拍**同一面墙的同一区域**。

    返回：
      {
        "pair_ok": bool,
        "dhash_hamming": int,          # 结构距离
        "hamming_ok": bool,
        "inlier_ratio": float | None,  # 仅 use_homography 时有值
        "reason": str,                 # 无论成败都给出人话理由
        "evidence": {...},             # 可复算的原始数值
      }

    ⚠️ **不做**任何「光照归一化后像素相减」—— 立面重拍的光照/白平衡
    差异极大，像素级相减会把光照当变化，是典型的假阳性来源。
    本函数只看**结构相似度**（dHash）与**可配准性**（单应内点率）。
    """
    ev: dict = {}
    h1 = dhash(img_t1)
    h2 = dhash(img_t2)
    d = hamming2(h1, h2)
    ev["dhash_h1"] = h1
    ev["dhash_h2"] = h2
    ev["dhash_hamming"] = d

    hamming_ok = d <= policy.max_dhash_hamming

    inlier_ratio = None
    if use_homography:
        inlier_ratio = _orb_homography_inlier_ratio(img_t1, img_t2)
        ev["inlier_ratio"] = inlier_ratio

    reasons = []
    if not hamming_ok:
        reasons.append(
            "结构距离 %d > 上界 %d ⇒ 大概率不是同一立面"
            % (d, policy.max_dhash_hamming))
    if use_homography and inlier_ratio is not None \
            and inlier_ratio < policy.min_inlier_ratio:
        reasons.append(
            "单应内点率 %.2f < %.2f ⇒ 无法可靠对齐"
            % (inlier_ratio, policy.min_inlier_ratio))

    ok = hamming_ok and (not use_homography or
                         (inlier_ratio is not None and
                          inlier_ratio >= policy.min_inlier_ratio))

    if ok:
        reason = ("结构距离 %d ≤ 上界 %d，判定为可配对的同一立面区域"
                  % (d, policy.max_dhash_hamming))
        if use_homography:
            reason += "（单应内点率 %.2f）" % (inlier_ratio or 0.0)
    else:
        reason = "；".join(reasons) or "未通过配对判据"

    return {
        "pair_ok": bool(ok),
        "dhash_hamming": int(d),
        "hamming_ok": bool(hamming_ok),
        "inlier_ratio": inlier_ratio,
        "reason": reason,
        "evidence": ev,
    }


def _orb_homography_inlier_ratio(a, b, *, max_features: int = 2000):
    """用 ORB + RANSAC 单应，返回内点率（内点数 / 匹配对数）。失败返回 None。"""
    import cv2
    import numpy as np
    try:
        g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
        g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
        orb = cv2.ORB_create(nfeatures=max_features)
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        if len(matches) < 8:
            return None
        matches = sorted(matches, key=lambda m: m.distance)
        src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return None
        return float(mask.sum()) / float(len(mask))
    except Exception:                                  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# 变化度量
# ---------------------------------------------------------------------------
def _match_detections(dets_t1, dets_t2, policy: ChangePolicy):
    """把 T1/T2 的缺陷按 IoU 贪心配对。返回 (pairs, only_t1, only_t2)。"""
    cands = []
    for i, a in enumerate(dets_t1):
        for j, b in enumerate(dets_t2):
            iou = _iou(a["bbox_xyxy"], b["bbox_xyxy"])
            if iou >= policy.min_iou_match:
                cands.append((iou, i, j))
    cands.sort(reverse=True)
    used1, used2 = set(), set()
    pairs = []
    for iou, i, j in cands:
        if i in used1 or j in used2:
            continue
        used1.add(i)
        used2.add(j)
        pairs.append((i, j, iou))
    only_t1 = [i for i in range(len(dets_t1)) if i not in used1]
    only_t2 = [j for j in range(len(dets_t2)) if j not in used2]
    return pairs, only_t1, only_t2


def _width_mm(d):
    """取缺陷的毫米宽度。优先 width_max_mm，退回 width_mean_mm，再退回 None。"""
    for k in ("width_max_mm", "width_mean_mm"):
        v = d.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v), k
    return None, None


def compare_defects(dets_t1, dets_t2, policy: ChangePolicy = DEFAULT_POLICY,
                    *, gsd_mm_per_px: float | None = None) -> dict:
    """对已配对的两个时相，逐缺陷给出变化量（**不是**速率，**不是**预测）。

    `dets_t1` / `dets_t2`：每项至少含 `bbox_xyxy`；含毫米字段则一并给变化。
    返回结构里 `changes` 每项说明：配对的两框、IoU、宽度前后值、
    变化量、以及**可判读性**（GSD 不够大时不给毫米变化，只给「有/无」）。
    """
    pairs, only1, only2 = _match_detections(dets_t1, dets_t2, policy)

    measurable = (gsd_mm_per_px is not None
                  and gsd_mm_per_px <= policy.gsd_measurable_max)

    changes = []
    for i, j, iou in pairs:
        a, b = dets_t1[i], dets_t2[j]
        wa, ka = _width_mm(a)
        wb, kb = _width_mm(b)
        item = {
            "t1_bbox": list(a["bbox_xyxy"]),
            "t2_bbox": list(b["bbox_xyxy"]),
            "iou": round(iou, 4),
            "cls_t1": a.get("cls_name"),
            "cls_t2": b.get("cls_name"),
            "width_mm_t1": wa,
            "width_mm_t2": wb,
            "width_key_t1": ka,
            "width_key_t2": kb,
            "measurable": measurable,
            "delta_mm": None,
            "rel_change": None,
            "verdict": "无法判定（GSD 不足）",
        }
        if wa and wb:
            d = wb - wa
            item["delta_mm"] = round(d, 4)
            item["rel_change"] = round(d / wa, 4) if wa else None
            if not measurable:
                item["verdict"] = "有变化（幅度不可信：GSD > %.2f）" \
                    % policy.gsd_measurable_max
            elif abs(d) / wa < policy.min_rel_width_change:
                item["verdict"] = "无明显变化（相对变化 < %.0f%%）" \
                    % (policy.min_rel_width_change * 100)
            elif d > 0:
                item["verdict"] = "增宽 %.3f mm（+%.0f%%）" % (d, 100 * abs(d) / wa)
            else:
                item["verdict"] = "变窄 %.3f mm（%.0f%%）" % (abs(d), 100 * abs(d) / wa)
        changes.append(item)

    # 未配上的：只能报「新增 / 消失」，不给毫米变化（没有对应物）
    new_defects = [{"t2_bbox": list(dets_t2[j]["bbox_xyxy"]),
                    "cls": dets_t2[j].get("cls_name")} for j in only2]
    gone_defects = [{"t1_bbox": list(dets_t1[i]["bbox_xyxy"]),
                     "cls": dets_t1[i].get("cls_name")} for i in only1]

    n_up = sum(1 for c in changes if c["delta_mm"] and c["delta_mm"] > 0)
    n_dn = sum(1 for c in changes if c["delta_mm"] and c["delta_mm"] < 0)
    n_flat = sum(1 for c in changes
                 if c["delta_mm"] is not None
                 and abs(c["delta_mm"]) / max(c["width_mm_t1"], 1e-9)
                 < policy.min_rel_width_change)

    return {
        "n_t1": len(dets_t1),
        "n_t2": len(dets_t2),
        "n_paired": len(pairs),
        "n_new": len(new_defects),
        "n_gone": len(gone_defects),
        "n_widened": n_up,
        "n_narrowed": n_dn,
        "n_no_change": n_flat,
        "measurable": measurable,
        "changes": changes,
        "new_defects": new_defects,
        "gone_defects": gone_defects,
        # ---- 反预测护栏：写死，绝不因为调用方想要而改变 ----
        "is_prediction": False,
        "requires_timepoints": policy.requires_timepoints,
        "note": ("本结果只描述 T1→T2 的**已发生**变化，不含速率、不含外推、"
                 "不含失效时间。要谈「演化」需 ≥ %d 个时间点。"
                 % policy.requires_timepoints),
    }


# ---------------------------------------------------------------------------
# 高层入口：配对失败就不给变化结论（与弃权同构）
# ---------------------------------------------------------------------------
def pair_and_diff(img_t1, img_t2, dets_t1, dets_t2,
                  policy: ChangePolicy = DEFAULT_POLICY, *,
                  use_homography: bool = False,
                  gsd_mm_per_px: float | None = None) -> dict:
    """配对 → 变化度量。**配对不成立时 changes 为 None**，防止误用。"""
    pr = check_pairing(img_t1, img_t2, policy, use_homography=use_homography)
    if not pr["pair_ok"]:
        return {
            "pair_ok": False,
            "pair_reason": pr["reason"],
            "evidence": pr["evidence"],
            "changes": None,
            "summary": "两次拍摄无法配对（%s）⇒ 按弃权原则**不给变化结论**。"
                       % pr["reason"],
            "is_prediction": False,
            "requires_timepoints": policy.requires_timepoints,
        }
    diff = compare_defects(dets_t1, dets_t2, policy,
                           gsd_mm_per_px=gsd_mm_per_px)
    diff.update({"pair_ok": True, "pair_reason": pr["reason"],
                 "evidence": pr["evidence"]})
    diff["summary"] = summarize(diff)
    return diff


def summarize(r: dict) -> str:
    """把 `compare_defects` 的结果压成一段人话（Markdown）。"""
    if not r.get("pair_ok", True):
        return r.get("summary") or "无法配对。"
    lines = [
        "**T1 → T2 变化检测**（配对成立：%s）" % r.get("pair_reason", ""),
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| T1 缺陷数 | %d |" % r["n_t1"],
        "| T2 缺陷数 | %d |" % r["n_t2"],
        "| 成功配对 | %d |" % r["n_paired"],
        "| 新增 | %d |" % r["n_new"],
        "| 消失 | %d |" % r["n_gone"],
        "| 增宽 | %d |" % r["n_widened"],
        "| 变窄 | %d |" % r["n_narrowed"],
        "| 无明显变化 | %d |" % r["n_no_change"],
        "| 毫米可判读 | %s |" % ("是" if r["measurable"] else "否（GSD 不足）"),
        "",
        "> %s" % r["note"],
    ]
    return "\n".join(lines)
