# -*- coding: utf-8 -*-
"""deploy_args.py —— 部署层参数 → `pipeline.run_one` args 的**单一映射 + 契约校验**。

为什么要有它（2026-09-29 代码审查 #4）
----------------------------------------
四个部署入口（单图/视频/批量/实时流）原先**各自手写** run_one 的 args 键名，
而且用的是 `analyze()` 的**形参名**（`distance_m`/`focal_mm`/`env_class`/`conf_thr`/
`rectify_mode`/`jgj125_parts`），而 `run_one` 实际读取的是**另一套键名**
（`distance`/`focal`/`env`/`conf`/`rectify`/`jgj125`）。
⇒ `live_stream.py` 与 `phone_live.py` 喂给 run_one 的 args **11 键错 7**，
多数靠 run_one 默认值恰好等于期望值才"巧合成立"，但 `jgj125` 恒失效。

本模块把「UI/部署参数 → run_one args」收口为**一个函数** `build_run_one_args`，
并用 `_RUN_ONE_KEYS`（run_one 实际 `args.get` 键集）做**白名单校验**：
  · 输出键若不在白名单 ⇒ 立刻报错（不再静默落空）；
  · `contract_selfcheck()` 对照 `pipeline.py` 源码重取键集 ——
    pipeline 哪天新增键，这个契约测试会抓到（防漂移）。

用法：
    from deploy_args import build_run_one_args, jgj125_to_ids
    args = build_run_one_args(distance_m=20.0, focal_mm=24.0,
                              env_class="二类环境（露天/潮湿）",
                              jgj125_parts=["梁板受力主筋处（0.50mm 危险点）"])
"""
from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# run_one 实际读取的 args.get 键集（与 02_code/pipeline.py 一致；
# 由 contract_selfcheck() 对照源码自动校验，防漂移）。
# ---------------------------------------------------------------------------
_RUN_ONE_KEYS = {
    "brick_pitch_mm", "calib_brick", "calib_object_mm", "calib_object_name",
    "calib_object_px", "conf", "distance", "env", "focal", "imgsz",
    "jgj125", "out", "rectify", "rectify_force", "save_annotated", "single_ood",
}

# UI 勾选的中文部位 → run_one 的 jgj125 英文标识（**单一映射**，
# 单图入口 analyze() 与视频入口 analyze_video() 都走这里 —— 视频入口历史上
# 直接把中文串 join 进 args["jgj125"]，run_one 集合匹配恒空 ⇒ 100% 静默失效）。
_JGJ125_ID_MAP = {
    "梁板受力主筋处（0.50mm 危险点）": "main-rebar",
    "板受拉区（1.00mm 危险点）": "slab-tension",
}


def jgj125_to_ids(parts):
    """UI 勾选的中文部位 → run_one 的 jgj125 英文标识。"""
    return [_JGJ125_ID_MAP.get(p, p) for p in (parts or [])]


def build_run_one_args(*, calib_mode="相机参数估算（最粗）",
                       calib_object_px=0.0, calib_object_mm=210.0,
                       calib_object_name="A4短边210mm", brick_pitch_mm=250.0,
                       distance_m=20.0, focal_mm=24.0,
                       env_class="二类环境（露天/潮湿）",
                       conf_thr=0.25, imgsz=640,
                       rectify_mode="关闭（正对拍摄）", jgj125_parts=None,
                       single_ood=False, out=None, save_annotated=False):
    """把部署层参数统一映射成 `run_one` 的 args 字典，并做白名单校验。

    所有入口（单图/视频/批量/实时流）都应调这一个函数，而不是各自拼键名。
    输出键若不在 `_RUN_ONE_KEYS` 里 ⇒ 立刻 `ValueError`（不再静默落空）。
    """
    args = {
        "conf": float(conf_thr),
        "imgsz": int(imgsz),
        "distance": float(distance_m),
        "focal": float(focal_mm),
        "calib_object_px": float(calib_object_px),
        "calib_object_mm": float(calib_object_mm),
        "calib_object_name": str(calib_object_name),
        "calib_brick": bool(str(calib_mode).startswith("砖缝")),
        "brick_pitch_mm": float(brick_pitch_mm),
        "env": str(env_class),
        "jgj125": (",".join(jgj125_to_ids(jgj125_parts)) if jgj125_parts else ""),
        "rectify": bool(rectify_mode and not str(rectify_mode).startswith("关闭")),
        "rectify_force": bool(rectify_mode and str(rectify_mode).startswith("强制")),
        "single_ood": bool(single_ood),
        "save_annotated": bool(save_annotated),
    }
    if out is not None:
        args["out"] = str(out)

    bad = set(args) - _RUN_ONE_KEYS
    if bad:
        raise ValueError(
            "deploy_args 输出了 run_one 不认识的键 %s —— 契约校验失败，"
            "说明映射写错了键名（这正是要消灭的那类 bug）。" % sorted(bad))
    return args


def _pipeline_src():
    """定位 pipeline.py（兼容源布局 `02_code/` 与交付包布局 `code/`）。"""
    root = Path(__file__).resolve().parent.parent
    for cand in (root / "02_code" / "pipeline.py", root / "code" / "pipeline.py"):
        if cand.exists():
            return cand.read_bytes().decode("utf-8-sig")
    raise FileNotFoundError("找不到 pipeline.py（期望位于 02_code/ 或 code/）")


def contract_selfcheck():
    """对照 pipeline.py 源码，断言 `_RUN_ONE_KEYS` 与 run_one 实际 args.get 键集一致。

    返回 (一致?, 源码有而表里没有, 表里有而源码没有)。
    任何一边不等 ⇒ 契约过期，必须同步。
    """
    actual = set(re.findall(r'args\.get\("([a-z_0-9]+)"', _pipeline_src()))
    return (actual == _RUN_ONE_KEYS,
            sorted(actual - _RUN_ONE_KEYS),
            sorted(_RUN_ONE_KEYS - actual))


if __name__ == "__main__":
    ok, missing_in_table, extra_in_table = contract_selfcheck()
    print("run_one 契约自检：", "✅ 一致" if ok else "❌ 不一致")
    if not ok:
        print("  源码有而 _RUN_ONE_KEYS 缺：", missing_in_table)
        print("  _RUN_ONE_KEYS 有而源码缺：", extra_in_table)
    raise SystemExit(0 if ok else 1)
