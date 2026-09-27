# -*- coding: utf-8 -*-
"""
一次无人机/相机巡检的**多图批量筛查**入口。

与 `pipeline.py`（单图）和 `video_screen.py`（单段视频）并列：
本脚本面向「一个立面拍了几十张照片」的典型巡检场景 —— 无人机绕飞、
或者人工分格拍摄 —— 把整个目录当作**一个建筑（或一个立面）**来出一份
聚合报告。

为什么不放进 `06_deploy/app.py`：
    按本项目纪律（见 MEMORY.md 铁律 #20），任何对 app.py **可见界面**的
    改动都必须重录实机运行视频。批量筛查是工程/命令行能力，放进 CLI 即可，
    避免无谓地让已录视频作废。

设计要点
--------
1. **复用主链路**：每张图都走 `pipeline.run_one`，不另写检测/测量逻辑，
   保证与单图、视频三者的口径完全一致。
2. **聚合到建筑级**：把每张图 `grade_all` 得到的 `DefectGrade` 汇到一起，
   再调用 `grade.building_risk_level` 得到 A/B/C/D 粗判 —— 这与单图报告
   用的是同一个函数，避免出现「两套分级口径」。
3. **校准一致性诊断**：与 `video_screen.py` 同款，输出 GSD 相对离散度；
   跨图标定若离散过大，聚合结论的可信度下降，必须显式告警。
4. **不虚构**：本脚本**不**做 SLAM/正射拼接/多图几何配准。若一个立面的
   多张照片来自不同拍摄距离，GSD 不一致是必然的，脚本如实报出来而不是
   偷偷用平均值糊过去。
5. **★ 分布级 OOD 门（2026-09-27 新增）**：批量筛查**天然**是「一批同源图」
   ⇒ 正好是报告 §7.16.6 那个**批级**判据 `p_hit` 的适用场景。
   动机：报告 §7.16.4 实测到「系统知道自己**几何上**什么时候不该给结论，
   但不知道**模型层面**什么时候不该给」—— 在跨源 OOD 数据上，几何合格 +
   图像清晰时，系统仍会输出风险等级，而该结论建立在近乎失效的检测之上。
   ⇒ 本脚本在**入口**加一道**分布级**门：统计整批图的
   `p_hit = (crack 检出中 conf ≥ C_HIT 的比例)`，低于阈值就**拒绝出结论**。

   ⚠️ **为什么必须单独跑一次低阈值检测**：主链路是以 `--conf`（默认 0.25）
   调的，若直接用主链路的检出算 `p_hit`，则**分子 = 分母 ⇒ p_hit 恒为 1**，
   门**数学上退化为常数、静默失效**（本轮实测踩到过）。
   ⇒ 门的**分母**必须来自一次**更低 conf floor**（默认 0.05）的检测。

用法
----
    python 02_code/batch_screen.py --dir=<图片目录> [--json=out.json]
        [--conf=0.25] [--imgsz=640] [--distance=20] [--focal=24]
        [--device=cpu] [--model=<权重>] [--facade=<名称>]
        [--ood-gate=1] [--ood-threshold=0.5957] [--ood-conf-floor=0.05]
        [--no-ood-gate]

输出
----
- 控制台：逐图一行 + 建筑级汇总 + 分布级 OOD 门判读。
- JSON：`<EVAL_DIR>/batch_screen_results.json`（或 --json 指定）。
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from common import (EVAL_DIR, RESULT_DIR, argv_flag, default_weight, dump_json,
                    log, resolve_weight)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ---------- 分布级 OOD 门（报告 §7.16.6）----------
# 判据：p_hit = （某类检出中 conf ≥ C_HIT 的个数）/（该类检出总数）
#   分母在 **conf floor**（默认 0.05）上统计 —— 见文件头「为什么必须单独跑一次」。
OOD_C_HIT = 0.25            # 达标阈值（与报告 §7.16.4 的测量窗口一致）
OOD_CONF_FLOOR = 0.05       # 分母的采样下界（必须严格 < C_HIT）
# 默认阈值取报告 §7.16.6 实测的 V3 bootstrap 5% 分位（crack 类，285 图）。
# ★ 这是**当时那一批数据**的保守下界；换数据/换类别必须重定（见 _ood_gate 的 warning）。
OOD_THRESHOLD_DEFAULT = 0.5957
OOD_CLASS_DEFAULT = "crack"   # 报告实测用的类别（class 0）


def list_images(folder: str | Path) -> list[Path]:
    """列出目录下的图片（不递归，按文件名排序以便复现）。"""
    p = Path(folder)
    if not p.is_dir():
        return []
    out = [q for q in sorted(p.iterdir())
           if q.is_file() and q.suffix.lower() in IMG_EXTS]
    return out


def _gsd_of(rec: dict) -> float | None:
    """从单图结果里取 GSD（mm/px）。取不到返回 None。"""
    calib = (rec.get("calibration") or {})
    v = calib.get("mm_per_px")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _ood_gate(p_hit: float | None, n_total: int, threshold: float,
              c_hit: float, floor: float, cls: str) -> dict:
    """分布级 OOD 门（纯函数，便于单测）。

    判据（报告 §7.16.6）：
        门 = 「p_hit < threshold ⇒ 拒绝该批出结论」
    返回 dict，含 fire（是否触发）/ verdict（文字）/ 以及复算所需的全部门槛。

    ⚠️ 三条必须随结论声明的边界（写进返回的 caveat，也写进控制台）：
      ① 阈值来自**另一批数据**（V3 285 图）的 bootstrap 5% 分位 ⇒
         误触发率 5% 是**同批自估、偏乐观**；换数据须重定；
      ② 这是**批级**判据：它只覆盖「一批同源图」的入口筛查，
         **不覆盖单图兜底**（单图 p_hit 分母可为 0~1，无意义）；
      ③ 分母必须在 floor 上统计，否则 p_hit 恒为 1（见文件头）。
    """
    out = {
        "cls": cls, "c_hit": c_hit, "conf_floor": floor,
        "threshold": threshold, "n_detections_at_floor": int(n_total),
        "p_hit": (round(float(p_hit), 4) if p_hit is not None else None),
        "fire": None, "verdict": "",
        "note": ("p_hit = 该类检出中 conf≥c_hit 的比例；分母在 conf_floor 上统计"
                 "（不可在 c_hit 上统计，否则恒为 1）。"),
        "caveat": ("阈值取自 V3 285 图的 bootstrap 5% 分位 ⇒ 误触发率系同批自估、偏乐观，"
                   "换数据/换类别须重定；本门为**批级**判据，不覆盖单图兜底。"),
    }
    if p_hit is None or n_total <= 0:
        out["verdict"] = f"未计算（该批在 conf≥{floor} 下没有 {cls} 检出）"
        out["fire"] = None
        return out
    if n_total < 10:
        # 样本太少时门本身不可靠 —— 如实声明，不硬给一个 fire 值。
        out["fire"] = None
        out["verdict"] = (f"⚠️ 样本不足（该批仅 {n_total} 个 {cls} 检出，<10）"
                          f"⇒ 门不可靠，**不下判定**")
        return out
    fire = bool(p_hit < threshold)
    out["fire"] = fire
    if fire:
        out["verdict"] = (f"⛔ **触发**：p_hit {p_hit:.4f} < 阈值 {threshold:.4f}"
                          f" ⇒ 该批的检测响应明显弱于域内基线，"
                          f"**拒绝据此批出危险性结论**（疑似分布外数据 / 拍摄条件异常）")
    else:
        out["verdict"] = (f"✅ 未触发：p_hit {p_hit:.4f} ≥ 阈值 {threshold:.4f}"
                          f" ⇒ 该批检测响应与域内基线相当，可继续出结论")
    return out


def compute_ood_p_hit(model, images: list, floor: float, c_hit: float,
                      cls: str = OOD_CLASS_DEFAULT, imgsz: int = 640) -> dict:
    """跑一次**低阈值**检测，只为统计分布级判据 p_hit。

    为什么单独跑：主链路以 `--conf`（0.25）调模型，其检出**全部** ≥0.25 ⇒
    直接拿主链路检出算 p_hit 会把分子=分母、恒为 1（门静默退化）。
    ⇒ 必须在**更低的 floor**（默认 0.05）上重跑，得到真实分母。

    返回 dict：{n_total, n_hit, p_hit}；读图失败的图跳过并计数。
    """
    n_total = 0
    n_hit = 0
    n_read_fail = 0
    from common import imread_u
    for p in images:
        img = imread_u(p)
        if img is None:
            n_read_fail += 1
            continue
        res = model.predict(img, conf=float(floor), imgsz=int(imgsz), verbose=False)
        if not res:
            continue
        r0 = res[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            continue
        clsid = r0.boxes.cls.cpu().numpy().astype(int)
        confs = r0.boxes.conf.cpu().numpy()
        from common import CLASSES
        for i in range(len(clsid)):
            name = CLASSES[clsid[i]] if clsid[i] < len(CLASSES) else str(clsid[i])
            if cls and name != cls:
                continue
            n_total += 1
            if float(confs[i]) >= float(c_hit):
                n_hit += 1
    p_hit = (n_hit / n_total) if n_total > 0 else None
    return {"n_total": n_total, "n_hit": n_hit, "p_hit": p_hit,
            "n_read_fail": n_read_fail}


def summarize_batch(records: list[dict], facade: str = "") -> dict:
    """把逐图结果聚合成建筑级摘要（纯函数，便于单测）。

    参数
    ----
    records : list[dict]
        每项为 `pipeline.run_one` 的返回（含 status / measurements / grades 等）。
    facade : str
        立面名称，仅用于报告展示。

    返回
    ----
    dict，键：
        n_images / n_ok / n_failed
        n_defects_total        —— 全部图加起来的缺陷实例数
        per_class              —— {类名: 实例数}
        gsd                    —— 校准一致性诊断块
        risk                   —— grade.building_risk_level 的建筑级粗判
        images                 —— 逐图轻量摘要（不含大字段）
    """
    ok = [r for r in records if r.get("status") == "ok"]
    failed = [r for r in records if r.get("status") != "ok"]

    # ---- 逐类计数 ----
    per_class: dict[str, int] = {}
    for r in ok:
        for m in (r.get("measurements") or []):
            cls = m.get("cls_name") or m.get("cls") or "unknown"
            per_class[cls] = per_class.get(cls, 0) + 1

    n_defects_total = sum(per_class.values())

    # ---- 校准一致性诊断（与 video_screen 同款判据）----
    gsds = [g for g in (_gsd_of(r) for r in ok) if g is not None]
    gsd_block: dict = {"n": len(gsds)}
    if gsds:
        med = statistics.median(gsds)
        spread = (max(gsds) - min(gsds)) / med * 100.0 if med > 0 else 0.0
        gsd_block.update({
            "median_mm_per_px": round(med, 5),
            "min_mm_per_px": round(min(gsds), 5),
            "max_mm_per_px": round(max(gsds), 5),
            "rel_spread_pct": round(spread, 1),
            # 与 video_screen.CV_INCONSISTENT=0.35 对齐：跨图 GSD 相对离散
            # 超过 35% 视为不一致（跨图比跨帧更宽松地看待，取同一阈值）。
            "stable": bool(spread <= 35.0),
        })
        if spread > 35.0:
            gsd_block["warning"] = (
                f"跨图标定尺度离散度 {spread:.1f}% 偏大：这些照片很可能来自"
                "不同拍摄距离/不同镜头。此时「毫米级」定量结论只在单图内可比，"
                "跨图聚合的尺寸统计不可直接比较；建议按拍摄距离分批，或改用"
                "同一距离采集。"
            )

    # ---- 建筑级风险粗判（复用 grade.building_risk_level，单一口径）----
    # ★ 关键：**必须**用真正的 DefectGrade 对象去调 building_risk_level，
    #   不要自己另写一套「数 severity」的等价逻辑 —— 那会导致两个 calibre：
    #   building_risk_level 会先把非结构类（苔藓）剔除再加权，自写逻辑极易
    #   漏掉这一步（满墙苔藓被误升到 C 级）。此处由 JSON 逐字段重建对象，
    #   保证与单图报告用的是**同一个**函数、同一套结构类判定。
    from grade import DefectGrade, building_risk_level

    grades_obj: list = []
    for r in ok:
        for g in (r.get("grades") or []):
            try:
                grades_obj.append(DefectGrade(
                    cls_name=g.get("cls_name", "unknown"),
                    bbox_xyxy=tuple(g.get("bbox_xyxy") or (0, 0, 0, 0)),
                    metric_name=g.get("metric_name", ""),
                    metric_value=g.get("metric_value", 0.0),
                    limit_mm=g.get("limit_mm"),
                    exceeded=bool(g.get("exceeded", False)),
                    severity=g.get("severity", "ok"),
                    reason=g.get("reason", ""),
                    confidence_note=g.get("confidence_note", ""),
                    extra=g.get("extra") or {},
                ))
            except TypeError:
                # 字段不兼容（升级/降级过 dataclass）时如实报错，不静默吞。
                raise
    risk = building_risk_level(grades_obj) if grades_obj else {
        "level": "A", "n_defects": 0, "n_danger": 0, "n_attention": 0,
        "level_desc": "（本批无缺陷，或全部未达分级门槛）",
    }

    # ★ 逐图弃权条数**必须逐图累加**，不能从聚合后的 risk 里读 ——
    #   聚合时的 building_risk_level 只看到「未弃权」的 grades，
    #   它并不知道有多少条被上游作废了。这里从每张图的 risk 里汇总。
    n_unjudgeable = 0
    n_abstained_images = 0
    for r in ok:
        rr = r.get("risk") or {}
        n_unjudgeable += int(rr.get("n_unjudgeable", 0) or 0)
        if rr.get("abstained"):
            n_abstained_images += 1
    risk["n_unjudgeable"] = n_unjudgeable
    risk["n_abstained_images"] = n_abstained_images

    # ★ 与单图口径一致：**全部条目都弃权时输出 U，而不是 A**。
    #   否则一批「一张都测不了」的照片会被报成 A（安全）—— 正是报告 §2.3
    #   反复强调的「把『判不了』说成『没问题』是最危险的误报」。
    #   判据用「有效条目数」而非 n_defects（后者含作废条）：
    #   有效条目 = 未弃权条目 = 总条数 − n_unjudgeable。
    n_total_grades = sum(len(r.get("grades") or []) for r in ok)
    n_effective = n_total_grades - n_unjudgeable
    if n_unjudgeable > 0 and n_effective <= 0:
        risk["level"] = "U"
        risk["level_desc"] = ("本批所有检出条目均因不可判读而作废，"
                              "无法给出危险性等级。")
        risk["need_professional_inspection"] = True
    risk["abstained"] = bool(n_unjudgeable)

    images_light = []
    for r in records:
        images_light.append({
            "image": Path(r.get("image", "")).name,
            "status": r.get("status"),
            "n_measurements": len(r.get("measurements") or []),
            "gsd_mm_per_px": _gsd_of(r),
            "risk_level": (r.get("risk") or {}).get("level"),
        })

    return {
        "facade": facade,
        "n_images": len(records),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "failed_images": [Path(r.get("image", "")).name for r in failed],
        "n_defects_total": n_defects_total,
        "per_class": per_class,
        "gsd": gsd_block,
        "risk": risk,
        # ★ 弃权可见性：run_one 现在会把不可判读条目降为 unjudgeable 并
        #   计入 risk.n_unjudgeable。聚合层必须把它抬到顶层，否则使用者
        #   只看到一个 A/B/C 级、看不到「这个结论建立在多少条作废数据上」。
        "n_unjudgeable": int(risk.get("n_unjudgeable", 0) or 0),
        "abstained": bool(risk.get("abstained", False)),
        "images": images_light,
    }


def main() -> None:
    folder = argv_flag("dir")
    if not folder:
        log("请指定 --dir=<图片目录>（一次巡检/一个立面的多张照片）")
        return
    imgs = list_images(folder)
    if not imgs:
        log(f"目录 {folder} 下没有图片（支持 {'/'.join(sorted(IMG_EXTS))}）")
        return

    save_json = argv_flag("json", "")
    facade = argv_flag("facade", "")
    model_path = argv_flag("model", default_weight("v11s640"))

    args = {
        "conf": argv_flag("conf", "0.25"),
        "imgsz": argv_flag("imgsz", "640"),
        "distance": argv_flag("distance", "20"),
        "focal": argv_flag("focal", "24"),
        "calib_object_px": argv_flag("calib-object-px"),
        "calib_object_mm": argv_flag("calib-object-mm", "210"),
        "calib_object_name": argv_flag("calib-object-name", "A4短边210mm"),
        "calib_brick": argv_flag("calib-brick") is not None,
        "brick_pitch_mm": argv_flag("brick-pitch-mm", "250"),
        "env": argv_flag("env", "二类环境（露天/潮湿）"),
        "env_class": argv_flag("env-class", "二类环境（露天/潮湿）"),
        "jgj125": argv_flag("jgj125", ""),
    }
    args["rectify"] = ("--rectify" in sys.argv[1:]
                       or str(argv_flag("rectify", "")).strip().lower()
                       not in ("", "0", "none", "false"))

    resolved = resolve_weight(model_path)
    log(f"模型: {resolved}")
    if not Path(resolved).exists():
        log("!! 模型不存在。请先训练（train.py）或用 --model 指定权重")
        return

    from ultralytics import YOLO
    from pipeline import run_one

    model = YOLO(resolved)
    dev = str(argv_flag("device", "") or "").strip()
    if dev:
        try:
            model.overrides["device"] = dev
            log(f"推理设备: {dev}")
        except Exception as e:      # noqa: BLE001
            log(f"!! 设置 device={dev} 失败，沿用默认: {e}")

    log(f"待处理 {len(imgs)} 张图片"
        + (f"（立面：{facade}）" if facade else ""))
    records: list[dict] = []
    for i, p in enumerate(imgs, 1):
        try:
            rec = run_one(p, model, args)
        except Exception as e:      # noqa: BLE001
            # 不吞异常语义：明确标成 error 并保留原因，绝不当成「无缺陷」。
            log(f"  [{i}/{len(imgs)}] {p.name} 处理异常: {type(e).__name__}: {e}")
            rec = {"image": str(p), "status": "error", "error": f"{type(e).__name__}: {e}"}
        records.append(rec)
        if rec.get("status") == "ok":
            n = len(rec.get("measurements") or [])
            g = _gsd_of(rec)
            log(f"  [{i}/{len(imgs)}] {p.name}: 检出 {n} 处"
                + (f"，GSD {g:.3f} mm/px" if g else ""))

    summary = summarize_batch(records, facade=facade)

    # ---------- ★ 分布级 OOD 门（报告 §7.16.6）----------
    # 默认开。`--no-ood-gate` 关；`--ood-gate=0` 同义。
    gate_on = True
    if "--no-ood-gate" in sys.argv[1:]:
        gate_on = False
    _og = str(argv_flag("ood-gate", "1")).strip().lower()
    if _og in ("0", "false", "no", "off", "none"):
        gate_on = False

    ood_block: dict | None = None
    if gate_on:
        try:
            floor = float(argv_flag("ood-conf-floor", str(OOD_CONF_FLOOR)))
            c_hit = float(argv_flag("ood-c-hit", str(OOD_C_HIT)))
            thr = float(argv_flag("ood-threshold", str(OOD_THRESHOLD_DEFAULT)))
            cls = str(argv_flag("ood-class", OOD_CLASS_DEFAULT) or OOD_CLASS_DEFAULT)
            imgsz = int(args.get("imgsz", 640))
            if floor >= c_hit:
                log(f"⚠️ --ood-conf-floor({floor}) ≥ c_hit({c_hit}) ⇒ p_hit 会退化，"
                    f"已强制用 {OOD_CONF_FLOOR}")
                floor = OOD_CONF_FLOOR
            log(f"分布级 OOD 门：在 conf≥{floor} 上重跑一次检测以统计 p_hit"
                f"（阈值 {thr}，类别 {cls}）…")
            raw = compute_ood_p_hit(model, imgs, floor, c_hit, cls, imgsz)
            ood_block = _ood_gate(raw["p_hit"], raw["n_total"], thr, c_hit, floor, cls)
            ood_block["n_hit"] = raw["n_hit"]
            if raw.get("n_read_fail"):
                ood_block["n_read_fail"] = raw["n_read_fail"]
        except Exception as e:      # noqa: BLE001
            # 不静默吞：门算不出来时如实报错，且**不**谎报「未触发」。
            log(f"⚠️ OOD 门计算失败（{type(e).__name__}: {e}）⇒ 本轮跳过门判定")
            ood_block = {"fire": None,
                         "verdict": f"未计算（异常：{type(e).__name__}）"}
        summary["ood_gate"] = ood_block

    log("=" * 64)
    log(f"建筑级汇总（{summary['n_ok']}/{summary['n_images']} 张成功）")
    log(f"  缺陷实例合计: {summary['n_defects_total']}")
    if summary["per_class"]:
        parts = ", ".join(f"{k} {v}" for k, v in
                          sorted(summary["per_class"].items(),
                                 key=lambda x: -x[1]))
        log(f"  逐类: {parts}")
    g = summary["gsd"]
    if g.get("median_mm_per_px"):
        log(f"  GSD 中位 {g['median_mm_per_px']} mm/px "
            f"(离散 {g['rel_spread_pct']}%, stable={g['stable']})")
        if g.get("warning"):
            log(f"  ⚠️ {g['warning']}")
    r = summary["risk"]
    log(f"  建筑级风险粗判: {r.get('level')}"
        f"（danger {r.get('n_danger', '?')} / attention {r.get('n_attention', '?')}）")
    if summary.get("n_unjudgeable"):
        log(f"  ⚠️ 其中 {summary['n_unjudgeable']} 条因不可判读已作废、"
            f"未参与等级判定（拒答机制生效）")
    log("  ⚠️ 本结论为筛查层面的风险排序，不构成法定鉴定。")

    # ---------- 分布级 OOD 门结果 ----------
    og = summary.get("ood_gate")
    if og:
        log("-" * 64)
        log("分布级 OOD 门（报告 §7.16.6）")
        if og.get("p_hit") is not None:
            log(f"  类别 {og.get('cls')}：{og.get('n_hit')}/{og.get('n_detections_at_floor')} "
                f"个检出达到 conf≥{og.get('c_hit')} ⇒ p_hit = {og.get('p_hit')}"
                f"（阈值 {og.get('threshold')}）")
        log(f"  {og.get('verdict')}")
        if og.get("fire") is True:
            # 门触发时，把上面那个风险等级**明确标为不可采信** —— 这正是本门的目的。
            log("  ⇒ **上面的建筑级风险等级请勿采信**：它建立在检测响应已明显塌陷的批次上。"
                "建议先排查拍摄条件/数据来源（是否分布外），或改由人工/专业鉴定。")
        if og.get("caveat"):
            log(f"  边界：{og['caveat']}")

    out_json = Path(save_json) if save_json else (
        EVAL_DIR / "batch_screen_results.json")
    # 落盘时剔除大字段（原图/可视化 base64 若存在），只留结构化摘要 + 逐图轻量信息。
    dump_json({
        "model": resolved,
        "args": args,
        "summary": summary,
    }, out_json)
    log(f"结果已写入 {out_json}")


if __name__ == "__main__":
    main()
