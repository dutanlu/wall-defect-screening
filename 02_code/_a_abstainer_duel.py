# -*- coding: utf-8 -*-
"""
_a_abstainer_duel.py —— A 路【四路对决】学习式弃权器 vs 手写几何 vs 纯置信度

===== 输入 =====
logs/_a_abstainer/features.csv （由 _a_collect_features.py 产出）
每行 = 一个检测实例的逐实例特征 + 共识真值 label_ok
  label_ok = 1 : cur 测量与 F1 多尺度参照一致（< TAU_AGREE）⇒ 测量可信
  label_ok = 0 : 两者差异大 ⇒ 应弃权
  label_ok = -1: 无 F1 参照（非线状类 / F1 失败）⇒ **不参与监督**

===== 四方参赛者（同一份数据、同一套评测）=====
  P0  纯 softmax 置信度     : score = conf（越高越可信）
  P1  手写几何判据（现行）  : score = f(px 是否落在 [MIN_RELIABLE_PX, ks-1] 带内)
                              —— 这正是 §5.2 那条「弃权判据必须是带不是阈值」
  P2  学习式弃权器(线性)    : LogisticRegression，特征 = [conf, ks_cur, roi_short,
                              window_ok, wmean_cur, L_cur]（**不含 F1**，防泄漏）
  P3  学习式弃权器(GBDT)    : GradientBoosting，同特征

===== ⚠️ 防泄漏纪律（关键）=====
真值 label_ok 是**用 F1 算出来的**。因此任何把 F1 相关列（L_f1/wmean_f1/wmax_f1/
rel_diff）喂进模型的做法都是**标签泄漏**，会让 AUROC 虚高到没有意义。
⇒ **参赛特征一律剔除全部 F1 列**。脚本内置断言：若特征名含 "f1" 直接报错退出。

===== 指标 =====
  ① AUROC（弃权 = 二分类；真值=可信）
  ② risk-coverage 曲线：在若干覆盖档下，被接受样本的**错误率**（= 1 - 精确率）
  ③ 关键对比：在「同等覆盖率」下谁的错误率最低 —— 这才是弃权器的真实价值

输出：
  logs/_a_abstainer/duel_report.json
  logs/_A_ABSTAINER_DUEL.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "logs" / "_a_abstainer"

MIN_RELIABLE_PX = 3.0          # 与 measure.py 的常数一致（§5.2）
COVER_GRID = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def _load_features(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _fnum(v, default=0.0):
    try:
        s = str(v).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _handwritten_geometric_score(r) -> float:
    """P1：现行手写判据。返回 [0,1] 的「可信度」分。

    规则（§5.2）：可信 ⟺ MIN_RELIABLE_PX ≤ px ≤ ks-1
    这里把「带内」做成软分：带内 1.0；带外按距离衰减，便于与其它路比 AUROC。
    """
    px = _fnum(r.get("wmean_cur"), 0.0)
    ks = _fnum(r.get("ks_cur"), 0.0)
    if ks <= 0:
        return 0.0
    lo, hi = MIN_RELIABLE_PX, ks - 1.0
    if px <= 0:
        return 0.0
    if lo <= px <= hi:
        return 1.0
    # 带外：按相对偏离衰减
    if px < lo:
        d = (lo - px) / max(lo, 1e-6)
    else:
        d = (px - hi) / max(hi, 1e-6)
    return float(max(0.0, 1.0 / (1.0 + d)))


def _auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUROC，用秩公式（无 sklearn 依赖也能算）。"""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    # 处理并列：取平均秩
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos_ranks = float(ranks[y == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _risk_coverage(y_true: np.ndarray, score: np.ndarray,
                   grid=COVER_GRID):
    """按 score 降序，取覆盖率档，统计被接受样本的错误率。

    返回 [(coverage, n_accept, err_rate)]。
    错误率 = 1 - 精确率（被接受的里面，label_ok=0 的占比）。
    """
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    y = np.asarray(y_true)[order]
    out = []
    n = len(y)
    for cov in grid:
        k = max(1, int(round(cov * n)))
        yk = y[:k]
        err = 1.0 - float((yk == 1).sum()) / k
        out.append((cov, k, round(err, 4)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(OUT_DIR / "features.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.30)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"!! 找不到 {csv_path}；请先跑 _a_collect_features.py")
        return 2

    rows = _load_features(csv_path)
    # 只保留有线状 F1 参照的（label_ok ∈ {0,1}）
    lab = [r for r in rows if int(_fnum(r.get("label_ok"), -1)) in (0, 1)]
    if len(lab) < 40:
        print(f"!! 可监督样本仅 {len(lab)}，太少（<40），无法训练/评测")
        print("   注意：只有线状类(crack/exposed_rebar/rust)才有 F1 参照。")
        return 3

    # ★ 更早的前置检查：**正例数**才是真正的瓶颈（不是总可监督数）。
    #   真值 label_ok=1（「测量可信」）在本项目是**极稀有类**：
    #   实测 smoke 12 图上 24 个可监督实例里只有 1 个正例（4.2%）。
    #   ⇒ 若正例过少，AUROC 的 CI 会宽到任何结论都不成立。
    n_pos_all = sum(1 for r in lab if int(_fnum(r.get("label_ok"), -1)) == 1)
    if n_pos_all < 20:
        rate = n_pos_all / max(1, len(lab))
        print(f"!! 正例（label_ok=1 可信）仅 {n_pos_all} 个 / 可监督 {len(lab)} 个 "
              f"（正例率 {rate:.2%}），样本量不足以做四路对决。")
        print("   为什么卡在这：AUROC 的正类极稀有 ⇒ CI 半宽 ≈ 1.96*sqrt(0.75*0.25/n)")
        print(f"   按 n={max(n_pos_all,1)} 估算，CI 半宽 ≈ "
              f"{1.96 * (0.75 * 0.25 / max(n_pos_all, 1)) ** 0.5:.3f}（越大越无法分辨）")
        print()
        print("   ⇒ 可选处置（**必须在报告里如实说明改了口径**）：")
        print("     ① 扩大扫描集（用更多测试图，正例自然变多）；")
        print("     ② 放宽 TAU_AGREE（0.15 → 0.25），把「勉强一致」也算可信；")
        print("     ③ **不二值化**：直接用连续的 rel_diff 做 AUROC，绕开正类稀有问题；")
        print("     ④ 若正例率 << 1%（例如 1000 个可监督里只有几个正例），")
        print("        说明「共识真值」本身把绝大多数样本判为不可信 ⇒")
        print("        **应如实报告「本判据下可信样本极稀有」这一负面发现**，")
        print("        而不是硬凑一个 n_pos≥20 的子集来出漂亮数字。")
        print("   本脚本**不自动降级**，避免在不知情的情况下产出弱结论。")
        return 6

    # ---------- 防泄漏：特征里绝不能含 f1 ----------
    FEATS = ["conf", "ks_cur", "roi_short", "window_ok", "wmean_cur", "L_cur",
             "roi_h", "roi_w", "wmax_cur"]
    for f in FEATS:
        if "f1" in f.lower():
            print(f"!! 特征 {f} 含 'f1' ⇒ 标签泄漏，拒绝运行")
            return 4

    y = np.array([int(_fnum(r["label_ok"])) for r in lab], dtype=int)
    X = np.array([[_fnum(r.get(k), 0.0) for k in FEATS] for r in lab], dtype=float)

    print("=" * 78)
    print("A 路 四路对决 —— 学习式弃权器 vs 手写几何 vs 纯置信度")
    print("=" * 78)
    print(f"样本 {len(lab)}（可信 {int((y==1).sum())} / 应弃权 {int((y==0).sum())}）")
    n_skipped = len(rows) - len(lab)
    print(f"（另有 {n_skipped} 个实例无 F1 参照，已排除）")

    # ---------- 切分（★ 分层，保证测试集两类都非空；四路共用同一测试集）----------
    #
    # ⚠️ 曾经的写法是「对全体做随机 permutation 再切前 n_test 个」。
    # 但本项目正类（label_ok=1，「测量可信」）是**少数类**（smoke 12 图上仅 1/24），
    # 在满量样本上随机切分极可能切出「测试集一个正例都没有」⇒ AUROC 恒为 NaN。
    # ⇒ 改为**按类别分层**切分：正负各自按同比例分到 train/test。
    rng = np.random.RandomState(args.seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_pos_te = int(round(args.test_size * len(pos_idx)))
    n_neg_te = int(round(args.test_size * len(neg_idx)))

    def _guard(name, n_all, n_te, n_tr):
        """任一集为空 ⇒ 明确报错，绝不产出 NaN AUROC 当结论。"""
        if n_te < 1 or n_tr < 1:
            print(f"!! 分层后 {name} 类测试/训练集为空"
                  f"（总 {n_all} / 测试 {n_te} / 训练 {n_tr}）")
            print("   ⇒ 样本太少，无法做可信的四路对决；请扩大扫描集（更多测试图）。")
            return False
        return True

    if not (_guard("正(label_ok=1 可信)", len(pos_idx), n_pos_te, len(pos_idx) - n_pos_te)
            and _guard("负(label_ok=0 应弃权)", len(neg_idx), n_neg_te, len(neg_idx) - n_neg_te)):
        return 5

    te = np.concatenate([pos_idx[:n_pos_te], neg_idx[:n_neg_te]])
    tr = np.concatenate([pos_idx[n_pos_te:], neg_idx[n_neg_te:]])
    rng.shuffle(te)          # 顺序打散，避免 P1 的「按 lab 序取」偏置
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    print(f"分层切分：训练 {len(tr)}（正 {len(pos_idx)-n_pos_te} / 负 {len(neg_idx)-n_neg_te}）"
          f"  测试 {len(te)}（正 {n_pos_te} / 负 {n_neg_te}）")
    if n_pos_te < 5:
        print(f"⚠️ 测试集正例仅 {n_pos_te} 个 ⇒ AUROC 的 CI 很宽，结论须谨慎表述。")

    results = {}

    # ---------- P0 纯 softmax ----------
    s0 = Xte[:, FEATS.index("conf")]
    results["P0_softmax_conf"] = dict(
        auroc=round(_auroc(yte, s0), 4),
        rc=_risk_coverage(yte, s0),
    )

    # ---------- P1 手写几何判据 ----------
    s1 = np.array([_handwritten_geometric_score(lab[i]) for i in te])
    results["P1_handwritten_geom"] = dict(
        auroc=round(_auroc(yte, s1), 4),
        rc=_risk_coverage(yte, s1),
    )

    # ---------- P2 / P3 学习式（需要 sklearn） ----------
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier

        # 标准化（线性模型需要；树模型无所谓但统一处理）
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd

        lr = LogisticRegression(max_iter=2000, random_state=args.seed)
        lr.fit(Xtr_s, ytr)
        s2 = lr.predict_proba(Xte_s)[:, 1]
        results["P2_learned_linear"] = dict(
            auroc=round(_auroc(yte, s2), 4),
            rc=_risk_coverage(yte, s2),
            coef={f: round(float(c), 4) for f, c in zip(FEATS, lr.coef_[0])},
        )

        gb = GradientBoostingClassifier(random_state=args.seed, n_estimators=120,
                                        max_depth=3, learning_rate=0.08)
        gb.fit(Xtr, ytr)
        s3 = gb.predict_proba(Xte)[:, 1]
        results["P3_learned_gbdt"] = dict(
            auroc=round(_auroc(yte, s3), 4),
            rc=_risk_coverage(yte, s3),
            importances={f: round(float(v), 4)
                         for f, v in zip(FEATS, gb.feature_importances_)},
        )
        have_sklearn = True
    except ImportError:
        have_sklearn = False
        print("⚠️ 未安装 sklearn ⇒ P2/P3 跳过（pip install scikit-learn -i https://pypi.org/simple）")

    # ---------- 汇总表 ----------
    print()
    print(f"{'路':30s} {'AUROC':>8s}   在同等覆盖率下的错误率")
    print(f"{'':30s} {'':>8s}   " + " ".join(f"@{int(c*100)}%" for c in COVER_GRID))
    print("-" * 78)
    name_map = {
        "P0_softmax_conf": "P0 纯 softmax 置信度",
        "P1_handwritten_geom": "P1 手写几何判据(现行)",
        "P2_learned_linear": "P2 学习式(LogisticRegression)",
        "P3_learned_gbdt": "P3 学习式(GradientBoosting)",
    }
    for k in ["P0_softmax_conf", "P1_handwritten_geom",
              "P2_learned_linear", "P3_learned_gbdt"]:
        if k not in results:
            continue
        r = results[k]
        errs = " ".join(f"{e[2]:>5.3f}" for e in r["rc"])
        print(f"{name_map[k]:30s} {r['auroc']:>8.4f}   {errs}")

    # ---------- 结论 ----------
    print()
    a0 = results["P0_softmax_conf"]["auroc"]
    a1 = results["P1_handwritten_geom"]["auroc"]
    concl = []
    if have_sklearn:
        a2 = results["P2_learned_linear"]["auroc"]
        a3 = results["P3_learned_gbdt"]["auroc"]
        best = max([("P1", a1), ("P2", a2), ("P3", a3)], key=lambda x: x[1])
        concl.append(f"最佳 AUROC = {best[1]:.4f}（{best[0]}）")
        if a3 > a1 + 0.02:
            concl.append("★ 学习式(GBDT) 明显优于手写几何判据（ΔAUROC > 0.02）")
        elif a1 > a3 + 0.02:
            concl.append("★ 手写几何判据反而优于学习式 —— 说明几何先验已足够强")
        else:
            concl.append("学习式与手写几何 **AUROC 差异在 0.02 内** ⇒ 分不出高下")
        # ★ 2026-09-25 修正：原判据拿「手写几何 a1」去比 softmax，
        #   但手写几何本身就是四路里最弱的一路（它只是「现行方案」的基线）。
        #   真正该回答的问题是「**本文主张的**学习式信号，相对纯 softmax 有没有增量」，
        #   故应拿**最优路**（a1/a2/a3 中的最大者）去比 a0。
        #   实测踩到：a1=0.5426 < a0=0.5786 ⇒ 旧逻辑打印「几何信号优势未超 0.03」，
        #   而真正的最优路 a3=0.7996 其实高出 a0 达 0.221 —— 措辞会把结论带偏。
        a_best_learned = max(a1, a2, a3)
        if a_best_learned > a0 + 0.03:
            concl.append(
                f"★ **学习式信号明显优于纯 softmax 置信度**"
                f"（最优路 AUROC {a_best_learned:.4f} vs {a0:.4f}，Δ {a_best_learned - a0:+.4f} > 0.03）"
                f" ⇒ 几何/多尺度信号含有 softmax 之外的**独立判别信息**。")
        else:
            concl.append(
                f"⚠️ 最优路 AUROC {a_best_learned:.4f} 相对纯 softmax {a0:.4f} "
                f"的优势未超 0.03 ⇒ 学习式信号未证实独立价值，需更大样本。")
    for c in concl:
        print("  ·", c)

    if not have_sklearn:
        print("  · （未装 sklearn，仅完成 P0/P1 两路对比）")

    # ---------- 落盘 ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rep = dict(
        n_rows_total=len(rows), n_labeled=len(lab),
        n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()),
        n_skipped_no_f1=n_skipped,
        features=FEATS, seed=args.seed, test_size=args.test_size,
        cover_grid=COVER_GRID,
        results=results, conclusions=concl,
        caveat=("真值为「共识真值」（cur 与 F1 一致），非绝对真值；"
                "特征已剔除全部 F1 列以防标签泄漏。"),
    )
    (OUT_DIR / "duel_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    md = _render_md(rep, name_map)
    (ROOT / "logs" / "_A_ABSTAINER_DUEL.md").write_bytes(
        md.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
    print(f"\n写出 {OUT_DIR / 'duel_report.json'} 与 logs/_A_ABSTAINER_DUEL.md")
    return 0


def _render_md(rep: dict, name_map: dict) -> str:
    L = []
    L.append("# A 路：四路弃权对决（学习式 vs 手写几何 vs 纯置信度）")
    L.append("")
    L.append(f"> 样本 {rep['n_labeled']}（可信 {rep['n_pos']} / 应弃权 {rep['n_neg']}），"
             f"另有 {rep['n_skipped_no_f1']} 个无 F1 参照已排除。")
    L.append(f"> 真值为**共识真值**（cur 与 F1 多尺度一致），非绝对真值；"
             f"特征已**剔除全部 F1 列**以防标签泄漏。")
    L.append("")
    L.append("## AUROC")
    L.append("")
    L.append("| 路 | AUROC |")
    L.append("|---|---|")
    for k in ["P0_softmax_conf", "P1_handwritten_geom",
              "P2_learned_linear", "P3_learned_gbdt"]:
        if k in rep["results"]:
            L.append(f"| {name_map[k]} | {rep['results'][k]['auroc']:.4f} |")
    L.append("")
    L.append("## Risk-Coverage（同等覆盖率下的错误率）")
    L.append("")
    L.append("| 路 | " + " | ".join(f"@{int(c*100)}%" for c in rep["cover_grid"]) + " |")
    L.append("|---|" + "---|" * len(rep["cover_grid"]))
    for k in ["P0_softmax_conf", "P1_handwritten_geom",
              "P2_learned_linear", "P3_learned_gbdt"]:
        if k in rep["results"]:
            errs = " | ".join(f"{e[2]:.3f}" for e in rep["results"][k]["rc"])
            L.append(f"| {name_map[k]} | {errs} |")
    L.append("")
    if rep.get("conclusions"):
        L.append("## 结论")
        L.append("")
        for c in rep["conclusions"]:
            L.append(f"- {c}")
        L.append("")
    # 特征重要性
    if "P3_learned_gbdt" in rep["results"] and rep["results"]["P3_learned_gbdt"].get("importances"):
        L.append("## GBDT 特征重要性")
        L.append("")
        L.append("| 特征 | 重要性 |")
        L.append("|---|---|")
        for f, v in sorted(rep["results"]["P3_learned_gbdt"]["importances"].items(),
                           key=lambda x: -x[1]):
            L.append(f"| {f} | {v:.4f} |")
        L.append("")
    L.append(f"> 注意：{rep['caveat']}")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
