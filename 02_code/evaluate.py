# -*- coding: utf-8 -*-
"""
evaluate.py —— 在独立测试集上评估，并产出「每类 AP」明细表

为什么不能只看训练时的 mAP：
  Ultralytics 训练过程中报的 mAP 是 **val** 集上的，而 val 集参与了
  「早停 / 选 best.pt」的模型选择过程，属于**间接参与调参**的数据。
  报告里要引用的是**测试集**指标 —— 测试集在整个训练流程中被完全隔离，
  只在最后评估一次。本脚本就是那个「最后评估一次」的执行者。

产出：
  04_results/eval/<run>_test_metrics.json   总指标 + 每类 AP
  04_results/eval/<run>_per_class.csv       可直接贴进报告的表格
  04_results/eval/<run>_confusion.png       混淆矩阵

用法：
  python evaluate.py --weights=..\\03_weights\\v8s640_best.pt --name=v8s640
  python evaluate.py --all                  # 评估 03_weights/ 下所有 *_best.pt
"""

from __future__ import annotations

import os

# ---- ★ onnxruntime 三道防线（必须在**任何 ultralytics 导入之前**设置）----
#
# 背景：加载 .onnx 做 predict/val 时，ultralytics 的 AutoBackend 会在
# device 未指定（或为 GPU）时判定 cuda=True，进而执行
#   check_requirements(("onnx", "onnxruntime-gpu"))
# 自动 `pip install onnxruntime-gpu`。而 onnxruntime-gpu 与 CPU 版
# **共用同一个 onnxruntime/ 包目录**，pip 会直接覆盖其中的 .py/.dll，
# 把环境弄成「Python 层是新版、dist-info 仍是旧版」的半损毁状态。
# 这是本项目真实发生过的事故，详见 logs/环境事故报告_20260921.md。
#
# 三道防线：
#   ① YOLO_AUTOINSTALL=false —— 总闸；但它在 ultralytics **import 时**
#      就被读入模块常量，所以**必须**在任何 ultralytics 导入之前设置。
#      本模块把 `from ultralytics import YOLO` 放在函数内延迟导入，
#      因此在这里设置是有效的。
#   ② ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1 —— 每次调用时求值、且在
#      包解析之前就 return，与导入顺序无关，**最可靠**。
#   ③ device="cpu" —— 由调用方显式传（见 --device 参数）；
#      另见 main() 里对 .onnx 权重的自动降级保护。
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

import csv
import json
from pathlib import Path

from common import (
    CLASSES,
    CLASS_CN,
    DATASET_DIR,
    DATASET_YAML_NAME,
    EVAL_DIR,
    WEIGHTS_DIR,
    argv_flag,
    dump_json,
    ensure_dirs,
    log,
)


def _fmt(v, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def evaluate_one(weights: Path, name: str, data: str, imgsz: int,
                 split: str, device: str) -> dict:
    """对单个权重在指定 split 上评估，返回结构化结果。"""
    from ultralytics import YOLO

    log(f"[{name}] 评估 {weights.name} @ imgsz={imgsz} split={split}")
    model = YOLO(str(weights))

    # ⚠️ 坑：ultralytics 的 plots=True 会在「模型类别数与标签体系对不上」时直接崩掉，
    # 而且崩在整条流程的最后一步 —— mAP 数值其实已经算完了，却因为画 PR 曲线失败
    # 导致整份测试集结果被丢弃。
    #
    # 【归因更正 2026-09-20】此前本项目把根因写成「ultralytics 绘图缺陷、names 只有 4 项」，
    # 自检复核后确认**这个归因是错的**，真实原因是**类别错配**：
    #   当时 04_results/train/v8n640/args.yaml 的 data 指向冒烟集（6 类、缺 moss），
    #   训出的 best.pt 是 6 类模型（输出 shape (1,10,8400) = 4 + 6），
    #   而评估用的 wall_defects.yaml 是 7 类（多出 moss），
    #   模型的类别 id 空间与数据集的 names 表对不上，索引时抛 KeyError。
    #   ultralytics 只是「把既有错误暴露出来」，它本身不是错误的来源。
    # 本项目自身另有一处独立缺陷（np_per_class 的取键方式）也已修，见下方注释。
    #
    # 处理策略（防御性保留）：先尝试 plots=True（正常情形要 PR 曲线图）；
    # 若失败，则退化为 plots=False 重跑一次，保住指标。
    # 宁可少一张 PR 曲线，也不能因为画图失败丢掉整个测试集结果。
    # 注：V2 数据集中 exposed_rebar / rust / delamination 三类实例数为 0，
    #     属「有类名无样本」的退化情形，仍会走到这条回落逻辑。
    #     （V3 起 7 类均有实例，该退化情形不再出现，此处保留作防御。）
    def _run_val(plots_on: bool):
        return model.val(
            data=str(data),
            split=split,
            imgsz=imgsz,
            batch=16,
            device=device,
            workers=0,          # Windows + 中文路径：多进程 dataloader 易卡死
            plots=plots_on,
            save_json=False,
            conf=0.001,         # 低 conf 才能算准 PR 曲线（AP 需要全阈值扫描）
            iou=0.6,
            verbose=False,
        )

    plot_fallback = None
    try:
        metrics = _run_val(True)
    except (KeyError, IndexError) as e:
        # 这里同时接 KeyError 与 IndexError：类别 id 空间错配时，越界既可表现为
        # 字典取键失败（KeyError），也可表现为数组下标越界（IndexError）。
        plot_fallback = (
            f"ultralytics 绘图阶段报错（{type(e).__name__}: {e}），已自动改用 plots=False 重跑。"
            f"最常见原因是「模型的类别 id 空间与数据集 names 表不一致」"
            f"（例如模型类别数与 wall_defects.yaml 的 7 类不符，见自检报告 P0-2），"
            f"其次是测试集存在 0 实例的类别导致的退化情形。指标本身有效。"
        )
        log(f"  !! {plot_fallback}")
        metrics = _run_val(False)

    rd = getattr(metrics, "results_dict", None) or {}
    overall = {k: (round(float(v), 5) if isinstance(v, (int, float)) else v)
               for k, v in rd.items()}

    # ---- 每类 AP ----
    per_class: list[dict] = []
    try:
        # metrics.box.ap_class_index 给出「本次评估中实际出现过的类 id」
        idx = list(getattr(metrics.box, "ap_class_index", []))
        p_arr = list(getattr(metrics.box, "p", []))
        r_arr = list(getattr(metrics.box, "r", []))
        ap50 = list(getattr(metrics.box, "ap50", []))
        ap = list(getattr(metrics.box, "ap", []))

        # ⚠️⚠️ 关于「每类实例数」的取值口径（自检 P2-9，2026-09-20 修正）：
        #   ultralytics 8.4.138 的 `Metric` 类里**没有 np_per_class 这个属性**，
        #   早期版本按它取值 → npc_map 恒为空字典 → 每类 instances 恒为 None。
        #   真正可用的是 **DetMetrics 上的 nt_per_class / nt_per_image**
        #   （np.ndarray，长度 = len(names)，按类 id 索引），由 DetMetrics.process()
        #   在 metrics.py:1174 从 stats["target_cls"] / stats["target_img"] 统计得出。
        #   为免「模型 id 顺序 vs 数据集 names 顺序」错配（本项目踩过，见 P0-2），
        #   这里**按类名对齐**，而不是按 id 直接对号入座。
        nt_pc = getattr(metrics, "nt_per_class", None)
        nt_pi = getattr(metrics, "nt_per_image", None)
        m_names = getattr(metrics, "names", None) or {}
        cnt_by_name: dict = {}
        img_by_name: dict = {}
        for _k, _v in (m_names.items() if hasattr(m_names, "items") else []):
            try:
                _ki = int(_k)
            except Exception:
                continue
            if nt_pc is not None and _ki < len(nt_pc):
                cnt_by_name[str(_v)] = int(nt_pc[_ki])
            if nt_pi is not None and _ki < len(nt_pi):
                img_by_name[str(_v)] = int(nt_pi[_ki])
        # 兜底：万一上游改了字段名，仍尝试旧的 dict 形态（按 id → 名字再对齐）
        if not cnt_by_name:
            npc = getattr(metrics.box, "np_per_class", None)
            if isinstance(npc, dict):
                cnt_by_name = {str(m_names.get(int(k), k)): int(v) for k, v in npc.items()}
        log(f"  每类实例统计: nt_per_class={'ok' if nt_pc is not None else '缺失'} / "
            f"nt_per_image={'ok' if nt_pi is not None else '缺失'} / "
            f"names={len(m_names)} 类 / 命中小计 {len(cnt_by_name)} 类")

        for n, ci in enumerate(idx):
            ci = int(ci)
            cls_name = CLASSES[ci] if 0 <= ci < len(CLASSES) else f"cls{ci}"
            per_class.append({
                "class_id": ci,
                "class": cls_name,
                "class_cn": CLASS_CN.get(cls_name, cls_name),
                "precision": round(float(p_arr[n]), 4) if n < len(p_arr) else None,
                "recall": round(float(r_arr[n]), 4) if n < len(r_arr) else None,
                "ap50": round(float(ap50[n]), 4) if n < len(ap50) else None,
                "ap50_95": round(float(ap[n]), 4) if n < len(ap) else None,
                "instances": cnt_by_name.get(cls_name),
                "images": img_by_name.get(cls_name),
            })
    except Exception as e:
        log(f"  !! 每类 AP 解析失败（总指标仍有效）: {e!r}")

    # 本次评估集里「完全没出现的类」也要列出来，
    # 否则报告表格会看起来像「该类 AP=1.0」或直接被误读成漏统计。
    appeared = {r["class_id"] for r in per_class}
    for ci, cname in enumerate(CLASSES):
        if ci not in appeared:
            per_class.append({
                "class_id": ci, "class": cname,
                "class_cn": CLASS_CN.get(cname, cname),
                "precision": None, "recall": None,
                "ap50": None, "ap50_95": None, "instances": 0, "images": 0,
                "note": "测试集中无该类实例，无法评估",
            })
    per_class.sort(key=lambda r: r["class_id"])

    rec = {
        "run": name,
        "weights": str(weights),
        "weights_mb": round(weights.stat().st_size / 1024 / 1024, 2),
        "split": split,
        "imgsz": imgsz,
        "overall": overall,
        "per_class": per_class,
    }
    if plot_fallback:
        rec["plot_fallback"] = plot_fallback
    return rec


def main() -> None:
    # 兼容工程包被拷贝到别的机器：若 yaml 里的 path 仍指向训练机旧路径，就地自愈
    from common import ensure_dataset_yaml, DATASET_YAML_NAME
    data = str(ensure_dataset_yaml(str(DATASET_DIR / DATASET_YAML_NAME)))
    split = argv_flag("split", "test")
    device = argv_flag("device", "0")
    imgsz = int(argv_flag("imgsz", "640"))
    single = argv_flag("weights")
    name = argv_flag("name")
    do_all = argv_flag("all") is not None

    ensure_dirs(EVAL_DIR)

    if not Path(data).exists():
        log(f"!! 数据集配置不存在: {data}")
        return

    jobs: list[tuple[Path, str]] = []
    if single:
        w = Path(single)
        if not w.exists():
            log(f"!! 权重不存在: {w}")
            return
        jobs.append((w, name or w.stem))
    elif do_all:
        for w in sorted(WEIGHTS_DIR.glob("*_best.pt")):
            # 文件名形如 v8s640_best.pt -> run 名 v8s640
            jobs.append((w, w.stem.replace("_best", "")))
    else:
        log("用法: python evaluate.py --weights=<pt> [--name=<run>]")
        log("      python evaluate.py --all")
        return

    if not jobs:
        log(f"!! 没有可评估的权重（{WEIGHTS_DIR}/*_best.pt 为空）")
        log("   请先跑训练：python train.py --runs=v8s640,v11s640")
        return

    # ---- ★ ONNX 权重的 device 自动降级（fail-safe）----
    # 若权重是 .onnx 而 device 不是 cpu，AutoBackend 会判 cuda=True
    # → 触发 onnxruntime-gpu 自动安装 → 弄坏环境（见文件头三道防线注释）。
    # 这里兜底：只要有一个 ONNX 任务，就把 device 强制为 cpu。
    if any(w.suffix.lower() == ".onnx" for w, _ in jobs) and device != "cpu":
        log(f"⚠ 检测到 ONNX 权重，device 自动从 {device!r} 降级为 'cpu'"
            f"（避免 ultralytics 自动安装 onnxruntime-gpu 损坏环境）")
        device = "cpu"

    results = []
    for weights, run_name in jobs:
        try:
            rec = evaluate_one(weights, run_name, data, imgsz, split, device)
            results.append(rec)

            # ---- 每类明细落盘（JSON + CSV）----
            dump_json(rec, EVAL_DIR / f"{run_name}_test_metrics.json")

            csv_path = EVAL_DIR / f"{run_name}_per_class.csv"
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["class_id", "class", "class_cn",
                             "precision", "recall", "ap50", "ap50_95",
                             "images", "instances"])
                for r in rec["per_class"]:
                    wr.writerow([r["class_id"], r["class"], r["class_cn"],
                                 _fmt(r.get("precision")), _fmt(r.get("recall")),
                                 _fmt(r.get("ap50")), _fmt(r.get("ap50_95")),
                                 r.get("images"), r.get("instances")])
            log(f"  已写: {csv_path.name}")

            # ---- 控制台汇总 ----
            m = rec["overall"]
            log(f"  总指标: P={_fmt(m.get('metrics/precision(B)'))} "
                f"R={_fmt(m.get('metrics/recall(B)'))} "
                f"mAP50={_fmt(m.get('metrics/mAP50(B)'))} "
                f"mAP50-95={_fmt(m.get('metrics/mAP50-95(B)'))} "
                f"({rec['weights_mb']} MB)")
            log("  每类 AP50:")
            for r in rec["per_class"]:
                if r.get("note"):
                    log(f"    {r['class']:15s} {r['class_cn']:6s} —  {r['note']}")
                else:
                    log(f"    {r['class']:15s} {r['class_cn']:6s} "
                        f"AP50={_fmt(r.get('ap50'))} AP50-95={_fmt(r.get('ap50_95'))} "
                        f"n={r.get('instances')} 图={r.get('images')}")
        except Exception as e:
            import traceback
            log(f"  !! {run_name} 评估失败: {e!r}")
            log(traceback.format_exc())
            results.append({"run": run_name, "weights": str(weights),
                            "status": "failed", "error": repr(e),
                            "traceback": traceback.format_exc()})

    dump_json({
        "data_yaml": str(data),
        "split": split,
        "imgsz": imgsz,
        "note": "测试集指标：测试集全程未参与训练与早停选择，是报告中唯一可引用的泛化指标",
        "results": results,
    }, EVAL_DIR / "test_eval_summary.json")

    log("=" * 72)
    log("测试集评估汇总（可直接引用进报告）")
    log("=" * 72)
    for r in results:
        if r.get("status") == "failed":
            log(f"{r['run']:12s} 失败")
            continue
        m = r["overall"]
        log(f"{r['run']:12s} mAP50={_fmt(m.get('metrics/mAP50(B)'))} "
            f"mAP50-95={_fmt(m.get('metrics/mAP50-95(B)'))} "
            f"P={_fmt(m.get('metrics/precision(B)'))} R={_fmt(m.get('metrics/recall(B)'))} "
            f"{r['weights_mb']}MB")
    log("=" * 72)
    log(f"明细: {EVAL_DIR}")


if __name__ == "__main__":
    main()
