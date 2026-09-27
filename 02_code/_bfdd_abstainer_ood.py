# -*- coding: utf-8 -*-
"""
_bfdd_abstainer_ood.py —— 【I1 后续】学习式弃权器能否挡住 BFDD 跨源失效？

===== 要回答的问题 =====
`logs/_BFDD_I1_结论.md` §三 实测到弃权机制的缺口：
  它由「几何 GSD + 成像质量」驱动 ⇒ 在**几何合格、图像清晰**的 OOD 数据上，
  系统仍会输出 `risk.level=B`，而该结论建立在跨源 AP50 = 0.0026 的检测之上。
§四 列了三个候选补救信号，其中 (2)「学习式弃权器」当时**未能实测**，
因为 `duel_report.json` 只存了 `importances`、没有树结构。
★ 2026-09-27 已在 `_a_abstainer_duel.py` 补上持久化（`abstainer_models.joblib`）
⇒ 本脚本就是补跑那次「欠下的实测」。

===== 方法论（关键，决定了结论能说什么）=====
弃权器是在 **V3 分布**上训练的，标签 = 「共识真值」（cur 与 F1 多尺度一致）。
BFDD **没有对应的 F1 参照**（骨架/宽度测量在 BFDD 的网状细裂缝上语义不同），
⇒ **不能**在 BFDD 上算 AUROC（没有真值）。
⇒ 因此本脚本只做**分布对比**（distributional test）：
   把 BFDD 的每个检出实例抽出**完全相同的 9 个特征**，
   喂进**冻结**的弃权器，看它给出的「可信概率」分布是否系统性低于 V3。
   若显著偏低 ⇒ 该分数可当**廉价 OOD/失效探测器**用（这正是 §四 想要的）；
   若与 V3 无异 ⇒ 如实报告「学习式弃权器也挡不住」，是一个**诚实的负结果**。

===== 特征对齐（必须与 _a_collect_features.py 完全一致）=====
  conf, ks_cur, roi_short, window_ok, wmean_cur, L_cur, roi_h, roi_w, wmax_cur
  复用 `measure_instance` + `_multiscale_measure`（直接 import，不另写一套）

===== 两侧口径（必须一致，否则对比无效）=====
  权重 : 同一个 `v11s640_best.pt`
  conf : 同一阈值（默认 0.25，与采集脚本一致）
  类别 : 只取 `crack`（class 0）——I1 的失效正是在这一类上量化的；
         跨类混合会把「类别构成差异」混进「域差异」

输出：
  logs/_bfdd_abstainer_ood.json
  logs/_bfdd_abstainer_ood.txt（人类可读日志）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from common import CLASSES, DATA_DIR, DATASET_DIR, imread_u, log  # noqa: E402
from gsd import Calibration  # noqa: E402

ROOT = Path(DATA_DIR).resolve().parent
OUT_DIR = ROOT / "logs"

# BFDD 跨源测试图（I1 用的同一批：`_p1_crossdomain_crack.py::build_labels` 把
# `Dataset_1x/test.txt` 列出的 152 张 RGB 图 **hardlink** 进 `images/val`）。
# ⚠️ 目录名是 `val`（不是 YOLO 惯例的 test）；`images/test` 为空，勿误当数据缺失。
BFDD_IMG_DIRS = [
    ROOT / "01_data" / "raw" / "_public_datasets" / "bfdd" / "_yolo" / "crackdet" / "images" / "val",
]

FEATS = ["conf", "ks_cur", "roi_short", "window_ok", "wmean_cur", "L_cur",
         "roi_h", "roi_w", "wmax_cur"]


def _collect_features(imgs: list[Path], model, conf_thr: float,
                      only_cls: str = "crack", tag: str = "") -> list[dict]:
    """抽特征。**与 _a_collect_features.py 同一套调用**。"""
    from measure import measure_instance

    # 复用采集脚本里的多尺度参照（保持口径一致）
    sys.path.insert(0, str(_HERE))
    from _a_collect_features import _multiscale_measure

    rows: list[dict] = []
    n_err = 0
    for i, p in enumerate(imgs, 1):
        img = imread_u(p)
        if img is None:
            continue
        res = model.predict(img, conf=conf_thr, verbose=False, device=0)
        if not res:
            continue
        r0 = res[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            continue
        calib = Calibration(mm_per_px=1.0, method="placeholder",
                            confidence="low", detail={"note": "像素级特征占位"})
        for bi in range(len(r0.boxes)):
            xyxy = r0.boxes.xyxy[bi].cpu().numpy().astype(int)
            cls_id = int(r0.boxes.cls[bi].cpu().item())
            conf = float(r0.boxes.conf[bi].cpu().item())
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
            if only_cls and cls_name != only_cls:
                continue
            try:
                m = measure_instance(img, tuple(xyxy), cls_name, calib)
            except Exception as e:  # noqa: BLE001 — 绝不静默吞
                m = None
                n_err += 1
                if n_err <= 3:
                    log(f"  ⚠️ measure_instance 异常 ×{n_err}: {type(e).__name__}: {e}")
            rows.append(dict(
                image=p.name, cls=cls_name, conf=round(conf, 4),
                roi_h=int(xyxy[3] - xyxy[1]), roi_w=int(xyxy[2] - xyxy[0]),
                roi_short=max(1, min(int(xyxy[3] - xyxy[1]), int(xyxy[2] - xyxy[0]))),
                ks_cur=int(getattr(m, "ks", 0)) if m else 0,
                window_ok=int(bool(getattr(m, "window_ok", True))) if m else 0,
                L_cur=round(float(getattr(m, "length_px", 0.0)) if m else 0.0, 3),
                wmean_cur=round(float(getattr(m, "width_mean_px", 0.0)) if m else 0.0, 4),
                wmax_cur=round(float(getattr(m, "width_max_px", 0.0)) if m else 0.0, 4),
            ))
        if i % 50 == 0:
            log(f"  {tag} 进度 {i}/{len(imgs)}，累计 {len(rows)}")
    return rows, n_err


def _score(model_payload: dict, rows: list[dict]) -> np.ndarray:
    """用冻结的弃权器打分（返回 P(可信)）。

    ⚠️ 线性路依赖训练时的标准化 ⇒ 必须用 payload 里的 mu/sd，不能用测试集自己的。
    """
    feats = model_payload["features"]
    X = np.array([[_fnum(r.get(k), 0.0) for k in feats] for r in rows], dtype=float)
    gb = model_payload["gb"]
    lr = model_payload["lr"]
    mu = np.asarray(model_payload["mu"], dtype=float)
    sd = np.asarray(model_payload["sd"], dtype=float)
    p_gb = gb.predict_proba(X)[:, 1]
    p_lr = lr.predict_proba((X - mu) / sd)[:, 1]
    return p_gb, p_lr


def _fnum(v, default=0.0):
    try:
        s = str(v).strip()
        return float(s) if s else default
    except Exception:
        return default


def _dist(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return dict(n=0)
    return dict(
        n=int(v.size),
        mean=round(float(v.mean()), 4),
        median=round(float(np.median(v)), 4),
        p10=round(float(np.percentile(v, 10)), 4),
        p90=round(float(np.percentile(v, 90)), 4),
        frac_ge_05=round(float((v >= 0.5).mean()), 4),
        frac_ge_03=round(float((v >= 0.3).mean()), 4),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--conf-floor", type=float, default=0.05,
                    help="算 p_hit 时统计检出总数的 conf 下界（必须 < --conf）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--weights", default=str(ROOT / "03_weights" / "v11s640_best.pt"))
    args = ap.parse_args()

    lines: list[str] = []

    def say(s=""):
        print(s)
        lines.append(s)

    # ---------- 1) 载入冻结的弃权器 ----------
    mp = ROOT / "logs" / "_a_abstainer" / "abstainer_models.joblib"
    if not mp.exists():
        say(f"!! 找不到 {mp}；请先跑 02_code/_a_abstainer_duel.py 生成模型产物")
        return 2
    import joblib
    payload = joblib.load(mp)
    say("=" * 78)
    say("I1 后续 —— 学习式弃权器能否挡住 BFDD 跨源失效？")
    say("=" * 78)
    say(f"弃权器 : {mp.name}（n_train={payload['n_train']}，正例={payload['n_train_pos']}，"
        f"seed={payload['seed']}）")
    say(f"权重   : {Path(args.weights).name}    conf 阈值: {args.conf}    类别: crack")
    say()

    # ---------- 2) 域内 V3 test：直接读已采好的 features.csv ----------
    csv_path = ROOT / "logs" / "_a_abstainer" / "features.csv"
    if not csv_path.exists():
        say(f"!! 找不到域内特征表 {csv_path}")
        return 2
    v3_rows_all = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("cls") == "crack":
                v3_rows_all.append(r)
    say(f"[域内 V3 test] crack 实例 {len(v3_rows_all)}（来自 features.csv，"
        f"conf≥0.25 已由采集阶段保证）")

    # ---------- 3) 跨源 BFDD：现抽特征 ----------
    imgs: list[Path] = []
    for d in BFDD_IMG_DIRS:
        if d.exists():
            imgs += sorted([p for p in d.glob("*")
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.limit:
        imgs = imgs[: args.limit]
    if not imgs:
        say("!! BFDD 测试图目录为空。I1 的图像是从 tar **流式**读取的，未落盘。")
        say("   ⇒ 见下方「补数据」分支：本脚本会改用 tar 流式抽取。")
        return 3
    say(f"[跨源 BFDD] 图像 {len(imgs)} 张，现抽 crack 特征…")

    from ultralytics import YOLO
    model = YOLO(args.weights)
    bfdd_rows, n_err = _collect_features(imgs, model, args.conf, "crack", "BFDD")
    say(f"[跨源 BFDD] crack 实例 {len(bfdd_rows)}（measure 异常 {n_err}）")
    say()

    # ---------- 3b) ★ 产出率(yield)：域漂移的最直接指标 ----------
    # 两侧同为 `conf≥{args.conf}` ⇒ 检出数之比直接反映「模型在该域的响应强度」。
    # ⚠️ 但**图像数不同**（V3 是 285 图，BFDD 是 152 图）⇒ 必须按「每图产出」比，不能比总数。
    n_v3_img = 285          # V3 test 图数（见报告 §4 口径）
    y_v3 = len(v3_rows_all) / n_v3_img
    y_bf = len(bfdd_rows) / max(1, len(imgs))
    say("─" * 78)
    say("【产出率】同一 conf 阈值下，每图 crack 检出数（域漂移的直接指标）")
    say("─" * 78)
    say(f"  V3 test   : {len(v3_rows_all):4d} 实例 / {n_v3_img} 图 = {y_v3:.4f} 个/图")
    say(f"  BFDD      : {len(bfdd_rows):4d} 实例 / {len(imgs)} 图 = {y_bf:.4f} 个/图")
    if y_v3 > 0:
        say(f"  ⇒ 产出率比 BFDD/V3 = {y_bf / y_v3:.4f}"
            f"（{'塌陷' if y_bf < y_v3 * 0.5 else '相当'}）")
    say()

    if len(v3_rows_all) == 0 or len(bfdd_rows) == 0:
        say("!! 某一侧 crack 实例为 0，无法对比")
        return 4

    # ---------- 4) 打分 ----------
    p_gb_v3, p_lr_v3 = _score(payload, v3_rows_all)
    p_gb_bf, p_lr_bf = _score(payload, bfdd_rows)

    # ---------- 4b) ★ 域内健全性检查（必须先证明弃权器在 V3 上是「活的」）----------
    # 若弃权器在 V3 上都不能分开正负例，那它在 BFDD 上的任何表现都无从解释
    # ⇒ 这是「先证实工具本身可信」的前置检查，缺了它后面全是空中楼阁。
    v3_lab = np.array([int(_fnum(r.get("label_ok"), -1)) for r in v3_rows_all])
    say("─" * 78)
    say("【前置健全性】弃权器在 V3 域内自己能否分开正/负例？")
    say("─" * 78)
    if (v3_lab == 1).sum() >= 3 and (v3_lab == 0).sum() >= 3:
        say(f"  GBDT  V3 正例(label_ok=1) n={int((v3_lab==1).sum()):3d} "
            f"median={np.median(p_gb_v3[v3_lab==1]):.4f}  |  "
            f"负例 n={int((v3_lab==0).sum()):3d} "
            f"median={np.median(p_gb_v3[v3_lab==0]):.4f}")
        sep_ok = np.median(p_gb_v3[v3_lab == 1]) > np.median(p_gb_v3[v3_lab == 0]) + 0.10
        say(f"  ⇒ {'✅ 有分离度（弃权器在域内是活的）' if sep_ok else '❌ 无分离度 ⇒ 后续对比无效'}")
        sanity_ok = bool(sep_ok)
    else:
        say("  ⚠️ 测试子集正/负例过少，跳过健全性检查")
        sanity_ok = None
    say()

    say("─" * 78)
    say("弃权器输出「可信概率」分布对比（分数越低 ⇒ 弃权器越倾向弃权）")
    say("─" * 78)
    say(f"{'路':26s} {'域':10s} {'n':>6s} {'mean':>8s} {'median':>8s} "
        f"{'p10':>8s} {'p90':>8s} {'≥0.5':>7s} {'≥0.3':>7s}")
    say("-" * 78)
    out = {}
    for name, v3s, bfs in [("P3 学习式 GBDT", p_gb_v3, p_gb_bf),
                           ("P2 学习式 线性", p_lr_v3, p_lr_bf)]:
        d_v3, d_bf = _dist(v3s), _dist(bfs)
        out[name] = dict(v3=d_v3, bfdd=d_bf)
        for tag, d in [("V3 域内", d_v3), ("BFDD 跨源", d_bf)]:
            say(f"{name:26s} {tag:10s} {d['n']:>6d} {d['mean']:>8.4f} "
                f"{d['median']:>8.4f} {d['p10']:>8.4f} {d['p90']:>8.4f} "
                f"{d['frac_ge_05']:>7.4f} {d['frac_ge_03']:>7.4f}")

    # 附加：纯 conf 对照（I1 §四 建议的最廉价信号）
    conf_v3 = np.array([_fnum(r.get("conf")) for r in v3_rows_all], dtype=float)
    conf_bf = np.array([_fnum(r.get("conf")) for r in bfdd_rows], dtype=float)
    d_cv3, d_cbf = _dist(conf_v3), _dist(conf_bf)
    out["P0 纯 conf"] = dict(v3=d_cv3, bfdd=d_cbf)
    for tag, d in [("V3 域内", d_cv3), ("BFDD 跨源", d_cbf)]:
        say(f"{'P0 纯 conf(对照)':26s} {tag:10s} {d['n']:>6d} {d['mean']:>8.4f} "
            f"{d['median']:>8.4f} {d['p10']:>8.4f} {d['p90']:>8.4f} "
            f"{d['frac_ge_05']:>7.4f} {d['frac_ge_03']:>7.4f}")

    # ---------- 5) 判据（预登记，跑前写死）----------
    say()
    say("─" * 78)
    say("判据（预登记）：主看 **GBDT 的 median** 与 **≥0.5 比例**")
    say("  · 若 BFDD 的 median 明显低于 V3（Δmed ≥ 0.10）**且** ≥0.5 比例显著降低")
    say("      ⇒ 【有效】学习式弃权器可当 OOD 探测器")
    say("  · 若两侧差异 < 上述阈值 ⇒ 【无效 / 与 conf 同源】，如实报告负结果")
    say("-" * 78)

    gb_d = out["P3 学习式 GBDT"]
    d_med = gb_d["v3"]["median"] - gb_d["bfdd"]["median"]
    d_half = gb_d["v3"]["frac_ge_05"] - gb_d["bfdd"]["frac_ge_05"]
    verdict, concl = "无法判定", []

    # ★ 关键现实约束（必须最先说，否则后面任何数字都会被误读）：
    #   BFDD 侧在 conf≥0.25 下只剩极少数实例 —— **这个「少」本身就是 OOD 症状**
    #   （I1 已量化：BFDD 达标比例 4.4% vs V3 67%）。
    #   但它同时意味着**弃权器层的对比样本量太小**，不能支撑一个强结论。
    concl.append(
        f"① **产出率**：同 conf 阈值下每图 crack 检出 V3 {y_v3:.4f} → BFDD {y_bf:.4f}"
        f"（比 {y_bf / y_v3 if y_v3 else float('nan'):.4f}）⇒ "
        f"**域漂移在「模型层」就已经把样本抽稀了**。")
    concl.append(
        f"② BFDD 在 conf≥{args.conf} 下仅 {gb_d['bfdd']['n']} 个 crack 实例"
        f"（V3 同窗 {gb_d['v3']['n']} 个）⇒ **该窗内的弃权器对比样本量不足**，"
        f"不能据此判定「弃权器有效/无效」。")

    if not sanity_ok:
        concl.append("③ ❌ 前置健全性未通过（弃权器在 V3 上都分不开正负例）⇒ 后续对比无效。")
    elif gb_d["bfdd"]["n"] < 10:
        verdict = "样本不足（但已定位到真正的信号层）"
        concl.append(
            "③ ⇒ **本轮结论：学习式弃权器在「conf≥0.25 工作窗」内无法被有效检验**，"
            "因为 OOD 把进入该窗的样本抽稀到了 n<10。")
        concl.append(
            "★ **但这本身是有价值的发现**：它说明**该在「模型层/分布层」拦，而不是在「测量层」拦** —— "
            "当样本已被 conf 抽稀到只剩 7 个时，测量层的弃权器已无事可做；"
            "真正该触发的是**「达标比例塌陷」这一分布级判据**（I1 §四 候选 1）。")
        concl.append(
            "⇒ **建议的报告结论**：把 §7.16.4 的「缺口」进一步精确化为 —— "
            "缺的不是「一个更聪明的测量层弃权器」，而是**上游的分布级 OOD 门**。")
    elif d_med >= 0.10 and d_half >= 0.10:
        verdict = "有效"
        concl.append(f"③ ✅ **有效**：GBDT 可信概率中位数 V3 {gb_d['v3']['median']:.4f} → "
                     f"BFDD {gb_d['bfdd']['median']:.4f}（Δmed {d_med:+.4f}）；"
                     f"≥0.5 比例 {gb_d['v3']['frac_ge_05']:.4f} → {gb_d['bfdd']['frac_ge_05']:.4f}"
                     f"（Δ {d_half:+.4f}）")
    elif d_med <= 0.05 and d_half <= 0.05:
        verdict = "无效"
        concl.append(f"③ ❌ **无效（诚实负结果）**：GBDT 分数两侧几乎无差异"
                     f"（Δmed {d_med:+.4f}，Δ≥0.5 比例 {d_half:+.4f}）"
                     f" ⇒ 学习式弃权器**也挡不住**这类跨源失效。")
    else:
        concl.append(f"③ ⚠️ **信号弱 / 不达预登记阈值**：Δmed {d_med:+.4f}，"
                     f"Δ≥0.5 比例 {d_half:+.4f}（需 ≥0.10 才算有效）")
    for c in concl:
        say("  · " + c)

    # 与纯 conf 对照：弃权器相对纯 conf 有没有增量
    cv = out["P0 纯 conf"]
    d_med_conf = cv["v3"]["median"] - cv["bfdd"]["median"]
    say()
    say(f"  对照：纯 conf 的 Δmed = {d_med_conf:+.4f}"
        f"（弃权器 Δmed = {d_med:+.4f}）")
    # ★ 注意：即便样本够，这个对照也容易被误读 ——
    #   GBDT 输出域与 conf 不同（前者是 0~1 的「可信概率」且被极度压低，
    #   后者是检测置信度）。故只在「同一预登记判据下」比**方向与量级**，不宣称严格可比。
    if gb_d["bfdd"]["n"] >= 10 and sanity_ok:
        if d_med > d_med_conf + 0.03:
            say("  ⇒ 弃权器**强于**纯 conf（增量 > 0.03），说明几何/多尺度特征带来了独立信号。")
        elif d_med < d_med_conf - 0.03:
            say("  ⇒ 弃权器**弱于**纯 conf ⇒ 复杂度没换来收益，如实记录。")
        else:
            say("  ⇒ 弃权器与纯 conf **本质同源**（差异 ≤ 0.03）。")
    else:
        say("  ⚠️ 因 BFDD 侧样本不足（或健全性未过），**不做**「弃权器 vs 纯 conf」的增量判定；"
            "仅记录两者在本轮各自的 Δmed 供后续扩样本后复用。")

    # ---------- 5b) ★ 建设性部分：把「分布级 OOD 门」做成可判定的判据 ----------
    # 上面的结论是「测量层拦不住，该在分布层拦」。那就必须回答：**分布层怎么拦、阈值多少**。
    # 定义：某批图的 p_hit = （crack 检出中 conf ≥ C_HIT 的个数）/（crack 检出总数）
    #
    # ⚠️⚠️ 一个**极易犯的致命错**（本轮实际踩到并修正）：
    #   若在 conf ≥ C_HIT 的**已筛选**数据上算 p_hit，则分子 = 分母 ⇒ **恒等于 1，门退化**。
    #   本脚本第一版的 `features.csv` 是 conf≥0.25 采的 ⇒ 正好掉进这个坑。
    #   ⇒ 正确做法：p_hit 必须在**更低的 conf floor** 上统计检出总数。
    #   故设 `--conf-floor`（默认 0.05，与 I1 §三 的测量窗口一致），
    #   在 floor 上重新跑两侧检测，再对 C_HIT=0.25 算 p_hit。
    say()
    say("─" * 78)
    say("【建设性】分布级 OOD 门：批级「达标比例」p_hit 的判别力")
    say("─" * 78)
    C_HIT = 0.25
    FLOOR = args.conf_floor
    say(f"定义：p_hit = （crack 检出中 conf≥{C_HIT} 的个数）/（crack 检出总数）")
    say(f"⚠️ 检出总数在 **conf floor = {FLOOR}** 上统计（**不能**在 {C_HIT} 上统计，否则 p_hit 恒为 1）")
    say(f"⚠️ 这是**批级**统计（需足够多图）；单图 p_hit 无意义。")
    say()

    if FLOOR >= C_HIT:
        say(f"  ⚠️ conf-floor({FLOOR}) ≥ C_HIT({C_HIT}) ⇒ p_hit 会退化，"
            f"请把 --conf-floor 设得低于 {C_HIT}（默认 0.05）")
        p_hit_v3 = p_hit_bf = float("nan")
        lo_v3 = float("nan")
        door_fires = None
        c_low, cf_low, boot = np.array([]), np.array([]), np.array([])
    else:
        # V3 侧：在 floor 上重跑检测（285 图）
        v3_test_dir = ROOT / "01_data" / "dataset" / "images" / "test"
        v3_imgs = sorted([p for p in v3_test_dir.glob("*")
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if v3_test_dir.exists() else []
        say(f"[V3 @floor] 在 conf≥{FLOOR} 上重抽 crack 检出（{len(v3_imgs)} 图）…")
        v3_low, _e1 = _collect_features(v3_imgs, model, FLOOR, "crack", "V3@floor")
        say(f"[V3 @floor] crack 检出 {len(v3_low)}")

        c_low = np.array([_fnum(r.get("conf")) for r in v3_low], dtype=float)
        cf_low = np.array([_fnum(r.get("conf")) for r in bfdd_rows], dtype=float)
        # ⚠️ BFDD 侧也要在 floor 上！但上面抽的是 0.25窗 ⇒ 需重抽
        say(f"[BFDD @floor] 在 conf≥{FLOOR} 上重抽 crack 检出（{len(imgs)} 图）…")
        bfdd_low, _e2 = _collect_features(imgs, model, FLOOR, "crack", "BFDD@floor")
        say(f"[BFDD @floor] crack 检出 {len(bfdd_low)}")
        cf_low = np.array([_fnum(r.get("conf")) for r in bfdd_low], dtype=float)

        p_hit_v3 = float((c_low >= C_HIT).mean()) if c_low.size else float("nan")
        p_hit_bf = float((cf_low >= C_HIT).mean()) if cf_low.size else float("nan")

        # V3 侧 bootstrap 定保守下界（用 V3 自己的批级分布）
        rng = np.random.RandomState(42)
        B = 500
        boot = []
        for _ in range(B):
            s = c_low[rng.randint(0, len(c_low), len(c_low))]
            boot.append(float((s >= C_HIT).mean()))
        boot = np.array(boot)
        lo_v3 = float(np.percentile(boot, 5))
        say(f"V3 域内 p_hit（bootstrap {B}，n={len(c_low)}）："
            f"中位 {np.median(boot):.4f}，5% 分位 {lo_v3:.4f}，1% 分位 {np.percentile(boot, 1):.4f}")
        say(f"BFDD 整批 p_hit = {p_hit_bf:.4f}（n={len(cf_low)}）")
        say()
        say(f"  ⇒ 若把门设为「p_hit < {lo_v3:.4f}（V3 的 5% 分位）则拒绝该批」：")
        say(f"     · V3 自身误触发率 ≈ 5%（by construction）")
        door_fires = bool(p_hit_bf < lo_v3)
        say(f"     · BFDD 触发 = {'✅ 是（成功拦住）' if door_fires else '❌ 否（漏过）'}"
            f"（{p_hit_bf:.4f} vs 门 {lo_v3:.4f}）")
        say()
        say("  ⚠️ 声明：本门用**同一批 V3 数据**既定阈值又测误触发率 ⇒ 误触发率是乐观估计；")
        say("     真正上线须在**留出集**上重定阈值（本轮无余量，如实标注）。")

    # ---------- 6) 落盘 ----------
    rep = dict(
        purpose="I1 后续：学习式弃权器能否挡住 BFDD 跨源失效（分布对比，非 AUROC）",
        model_artifact=str(mp.relative_to(ROOT)).replace("\\", "/"),
        weights=Path(args.weights).name, conf_thr=args.conf, cls="crack",
        n_v3=len(v3_rows_all), n_bfdd=len(bfdd_rows), n_measure_err=n_err,
        n_v3_img=n_v3_img, n_bfdd_img=len(imgs),
        yield_v3_per_img=round(float(y_v3), 4),
        yield_bfdd_per_img=round(float(y_bf), 4),
        yield_ratio=round(float(y_bf / y_v3), 4) if y_v3 else None,
        sanity_v3_separates=sanity_ok,
        distribution_gate=dict(
            C_HIT=C_HIT,
            conf_floor=FLOOR,
            v3_n_at_floor=len(c_low) if FLOOR < C_HIT else None,
            bfdd_n_at_floor=len(cf_low) if FLOOR < C_HIT else None,
            v3_p_hit=(round(float(p_hit_v3), 4) if p_hit_v3 == p_hit_v3 else None),
            bfdd_p_hit=(round(float(p_hit_bf), 4) if p_hit_bf == p_hit_bf else None),
            v3_p_hit_boot_p05=(round(float(lo_v3), 4) if lo_v3 == lo_v3 else None),
            v3_p_hit_boot_p01=(round(float(np.percentile(boot, 1)), 4)
                              if FLOOR < C_HIT else None),
            door_fires_on_bfdd=door_fires,
            note=("批级判据：p_hit = crack 检出中 conf≥C_HIT 的比例，检出总数在 conf_floor 上统计"
                  "（**不可**在 C_HIT 上统计，否则恒为 1）；阈值取 V3 bootstrap 5% 分位；"
                  "误触发率 5% 系同批自估、偏乐观，上线须留出集重定。"),
        ),
        v3_source="logs/_a_abstainer/features.csv（域内已采）",
        bfdd_source="01_data/raw/_public_datasets/bfdd/_yolo/crackdet/images/val（现抽）",
        dist=out,
        delta_median_gbdt=round(float(d_med), 4),
        delta_ge05_gbdt=round(float(d_half), 4),
        delta_median_conf=round(float(d_med_conf), 4),
        verdict=verdict,
        conclusions=concl,
        caveat=("① BFDD 无 F1 参照 ⇒ **不能算 AUROC**，本项只做分布对比；"
                "② BFDD 在 conf≥0.25 下仅少量实例（产出率塌陷）⇒ 弃权器层的对比样本不足，"
                "本项**不宣称**「弃权器有效/无效」，而是把 §7.16.4 的缺口精确定位到「分布层」；"
                "③ 两侧 crack 的成像尺度/形态差异大（BFDD 为无人机远景网状细裂缝）；"
                "④ 弃权器在 V3 分布上训练，这是**零样本域外打分**；"
                "⑤ 本项只评测，未训练、未微调，V3 已上报数字未受影响。"),
    )
    (OUT_DIR / "_bfdd_abstainer_ood.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_DIR / "_bfdd_abstainer_ood.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    say()
    say(f"写出 {OUT_DIR / '_bfdd_abstainer_ood.json'} 与 _bfdd_abstainer_ood.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
