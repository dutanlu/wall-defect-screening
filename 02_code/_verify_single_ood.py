# -*- coding: utf-8 -*-
r"""验证「单图 OOD 兜底」判据在**真实数据**上是否真的能分开域内/域外。

**为什么要跑这个**：判据写在纸上都好看；只有当它在 V3（域内）与 BFDD（跨源）
两侧产生**系统性差异**，才配叫判据。否则就是自欺。

设计：
  - 对 V3 test（285 图）与 BFDD val（152 图）**逐图**跑 `single_image_ood`；
  - 对每图跑 `decide_single` 得到 level；
  - 统计两侧的 **level 分布** 与 **各判据的命中率**。

预登记判据（跑前写死，避免事后挑）：
  - **主判据**：level ≥ 1 的比例，BFDD 应显著高于 V3；
  - **强判据**：level == 2（零检出 / 多票）比例，BFDD 应显著高于 V3；
  - **若两侧无差异** ⇒ 结论就是「单图判据不成立」，**如实写**，不调参硬凑。

用法：
  python 02_code/_verify_single_ood.py [--n 60]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(r"D:\pythonstudy 备份\创新题\外墙缺陷筛查")
V3_TEST = ROOT / "01_data" / "dataset" / "images" / "test"
BFDD_VAL = (ROOT / "01_data" / "raw" / "_public_datasets" / "bfdd"
            / "_yolo" / "crackdet" / "images" / "val")
WEIGHTS = ROOT / "03_weights" / "v11s640_best.pt"
OUT = ROOT / "04_results" / "eval" / "single_ood_verify.json"


def collect(d: Path, n: int, seed: int = 0) -> list[Path]:
    import random
    fs = sorted([p for p in d.iterdir()
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    random.seed(seed)
    if n and len(fs) > n:
        fs = sorted(random.sample(fs, n))
    return fs


def run_side(model, files: list[Path], label: str) -> dict:
    from common import imread_u
    from single_oom import single_image_ood, decide_single

    rows = []
    for i, p in enumerate(files):
        img = imread_u(p)
        if img is None:
            rows.append({"file": p.name, "read_fail": True})
            continue
        st = single_image_ood(model, img)
        dec = decide_single(st)
        rows.append({
            "file": p.name,
            "level": dec["level"],
            "all_n": st["all_n"],
            "n_det": st["n_det"],
            "p_hit_local": st["p_hit_local"],
            "conf_max": st["conf_max"],
            "conf_med": st["conf_med"],
            "votes": dec["votes"],
        })
        if (i + 1) % 20 == 0:
            print("   %s %d/%d" % (label, i + 1, len(files)), flush=True)

    ok = [r for r in rows if not r.get("read_fail")]
    n = len(ok)
    dist = {0: 0, 1: 0, 2: 0}
    for r in ok:
        dist[r["level"]] += 1

    with_det = [r for r in ok if (r["all_n"] or 0) > 0]
    cmeds = [r["conf_med"] for r in with_det if r["conf_med"] is not None]

    summary = {
        "label": label,
        "n_images": n,
        "n_read_fail": len(rows) - n,
        "level_dist": dist,
        "frac_level_ge1": (dist[1] + dist[2]) / n if n else None,
        "frac_level_2": dist[2] / n if n else None,
        "frac_zero_det": (sum(1 for r in ok if (r["all_n"] or 0) == 0) / n) if n else None,
        "n_with_det": len(with_det),
        "conf_med_mean_of_images": (sum(cmeds) / len(cmeds)) if cmeds else None,
        "rows": rows,
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="每侧抽样张数（0=全部）")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    from ultralytics import YOLO
    if not WEIGHTS.exists():
        print("!! 权重不存在:", WEIGHTS)
        return 1

    print("加载权重:", WEIGHTS)
    model = YOLO(str(WEIGHTS))
    try:
        model.to("cuda:%s" % a.device if str(a.device).isdigit() else a.device)
    except Exception as e:
        print("  (device 设置失败，用默认:", e, ")")

    v3 = collect(V3_TEST, a.n, seed=0)
    bd = collect(BFDD_VAL, a.n, seed=0)
    print("V3 test 抽样 %d 图；BFDD val 抽样 %d 图" % (len(v3), len(bd)))

    print("\n--- V3（域内）---")
    s_v3 = run_side(model, v3, "V3")
    print("\n--- BFDD（跨源）---")
    s_bd = run_side(model, bd, "BFDD")

    # ---- 判读（预登记判据）----
    f1_v3, f1_bd = s_v3["frac_level_ge1"], s_bd["frac_level_ge1"]
    f2_v3, f2_bd = s_v3["frac_level_2"], s_bd["frac_level_2"]
    fz_v3, fz_bd = s_v3["frac_zero_det"], s_bd["frac_zero_det"]

    print("\n" + "=" * 70)
    print("逐图 level 分布")
    print("%-8s %6s %8s %8s %8s" % ("域", "n", "NORMAL", "SUSPECT", "ABSTAIN"))
    for s in (s_v3, s_bd):
        d = s["level_dist"]
        print("%-8s %6d %8d %8d %8d" % (s["label"], s["n_images"],
                                        d[0], d[1], d[2]))
    print()
    print("%-22s %10s %10s %8s" % ("指标", "V3(域内)", "BFDD(跨源)", "差"))
    print("%-22s %9.1f%% %9.1f%% %+7.1fpp"
          % ("level≥1 比例", 100 * f1_v3, 100 * f1_bd, 100 * (f1_bd - f1_v3)))
    print("%-22s %9.1f%% %9.1f%% %+7.1fpp"
          % ("level==2 比例", 100 * f2_v3, 100 * f2_bd, 100 * (f2_bd - f2_v3)))
    print("%-22s %9.1f%% %9.1f%% %+7.1fpp"
          % ("零检出比例", 100 * fz_v3, 100 * fz_bd, 100 * (fz_bd - fz_v3)))
    if s_v3["conf_med_mean_of_images"] is not None:
        print("%-22s %9.4f  %9.4f  %+7.4f"
              % ("有检出图的 conf_med 均值",
                 s_v3["conf_med_mean_of_images"],
                 s_bd["conf_med_mean_of_images"],
                 s_bd["conf_med_mean_of_images"] - s_v3["conf_med_mean_of_images"]))

    verdict = {
        "main_criterion_delta_pp": 100 * ((f1_bd or 0) - (f1_v3 or 0)),
        "strong_criterion_delta_pp": 100 * ((f2_bd or 0) - (f2_v3 or 0)),
        "zero_det_delta_pp": 100 * ((fz_bd or 0) - (fz_v3 or 0)),
    }
    verdict["main_works"] = bool((f1_bd or 0) - (f1_v3 or 0) > 0.30)
    verdict["strong_works"] = bool((f2_bd or 0) - (f2_v3 or 0) > 0.20)
    print()
    print("主判据(level≥1) 差 %.1f pp ⇒ %s"
          % (verdict["main_criterion_delta_pp"],
             "✅ 成立" if verdict["main_works"] else "❌ 不成立"))
    print("强判据(level=2) 差 %.1f pp ⇒ %s"
          % (verdict["strong_criterion_delta_pp"],
             "✅ 成立" if verdict["strong_works"] else "❌ 不成立"))

    out = {
        "purpose": "验证单图 OOD 兜底判据能否分开 V3(域内) 与 BFDD(跨源)",
        "weights": str(WEIGHTS.relative_to(ROOT)),
        "n_per_side": a.n,
        "v3": {k: v for k, v in s_v3.items() if k != "rows"},
        "bfdd": {k: v for k, v in s_bd.items() if k != "rows"},
        "verdict": verdict,
        "note": ("预登记判据：level≥1 差 > 30pp 判主判据成立；level==2 差 > 20pp 判强判据成立。"
                 "若两侧无差异 ⇒ 如实写「单图判据不成立」，不调参硬凑。"),
        "v3_rows": s_v3["rows"],
        "bfdd_rows": s_bd["rows"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n-> %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
