# -*- coding: utf-8 -*-
"""
step0_download_roboflow.py —— 多源数据集下载（Roboflow SDK 通道）

为什么走 SDK 而不是网页：
  实测 `universe.roboflow.com` 对脚本请求返回 403（Cloudflare 拦截），
  但 `api.roboflow.com` 返回 200 —— 所以程序化下载必须走 SDK。

前置：
  1. pip install roboflow
  2. 注册 https://app.roboflow.com → Settings → API Key 拿到私钥
  3. 把 key 写入 环境变量 ROBOFLOW_API_KEY，或本目录下 .roboflow_key 文件（勿提交）

⚠️ 数据集 workspace/project/version 三元组会变，下面的配置是「候选」，
   必须在浏览器里打开数据集页确认后再跑。跑之前脚本会打印将要下载的地址。

用法：
  set ROBOFLOW_API_KEY=xxxx     （PowerShell: $env:ROBOFLOW_API_KEY="xxxx"）
  python step0_download_roboflow.py
  python step0_download_roboflow.py --list      # 只打印配置，不下载
  python step0_download_roboflow.py --only=jq3_building_defect

输出：
  01_数据/00_原始下载/<别名>/**
  04_结果/评估/download_log.json
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from common import RAW_DIR, EVAL_DIR, argv_flag, dump_json, ensure_dirs, log

# --------------------------------------------------------------------------
# 候选数据源（下载前请逐个在浏览器核实）
#   alias          : 本地目录名，后续 step3 映射表用这个名字
#   workspace      : Roboflow workspace slug
#   project        : project slug
#   version        : 版本号
#   note           : 备注（类别数、来源）
# --------------------------------------------------------------------------
SOURCES: list[dict] = [
    {
        "alias": "jq3_building_defect",
        "workspace": "jq3",
        "project": "building_defect",
        "version": 1,
        "note": "约 5760 张；crack/corrosion/mold/peeling/spalling；建筑缺陷主力源",
    },
    {
        "alias": "surface-defects-in-heritage",
        "workspace": "defects-dataset",
        "project": "surface-defects-in-heritage",
        "version": 1,
        "note": "约 3.3k；Crack/Corrosion/Vegetation/Blackening/Brick Spalling",
    },
    {
        "alias": "paint-defects",
        "workspace": "main-zxmvk",
        "project": "paint-defects-dfbjj",
        "version": 1,
        "note": "743 张；含 delamination/efflorescence/exposed rebar 等罕见类，补长尾",
    },
]


def get_api_key() -> str | None:
    """优先环境变量，其次本地文件。"""
    k = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if k:
        return k
    f = Path(__file__).resolve().parent / ".roboflow_key"
    if f.exists():
        k = f.read_text(encoding="utf-8").strip()
        if k:
            return k
    return None


def main() -> None:
    only = argv_flag("only")
    list_only = argv_flag("list") is not None

    targets = [s for s in SOURCES if (only is None or s["alias"] == only)]
    if not targets:
        log(f"未找到别名 {only}")
        return

    log("=" * 70)
    log("将要下载的数据集（请先在浏览器核实 workspace/project/version 是否存在）：")
    for s in targets:
        url = f"https://universe.roboflow.com/{s['workspace']}/{s['project']}/dataset/{s['version']}"
        log(f"  [{s['alias']}]")
        log(f"    URL : {url}")
        log(f"    备注: {s['note']}")
    log("=" * 70)

    if list_only:
        log("--list 模式，仅打印不下载。")
        return

    key = get_api_key()
    if not key:
        log("!! 未找到 Roboflow API Key")
        log("   PowerShell 里执行：$env:ROBOFLOW_API_KEY=\"你的key\"")
        log("   或写入 02_代码/.roboflow_key 文件（一行，无引号）")
        log("   获取地址：https://app.roboflow.com/settings/api")
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        log("!! 未安装 roboflow 包，请先执行：")
        log("   D:\\下载\\python.exe -m pip install roboflow")
        sys.exit(1)

    ensure_dirs(RAW_DIR, EVAL_DIR)
    rf = Roboflow(api_key=key)

    records = []
    for s in targets:
        dest = RAW_DIR / s["alias"]
        log(f"下载 {s['alias']} → {dest}")
        rec = {"alias": s["alias"], "dest": str(dest), "status": "pending"}
        try:
            project = rf.workspace(s["workspace"]).project(s["project"])
            version = project.version(s["version"])
            ds = version.download(
                "yolov8",
                location=str(dest),
                overwrite=False,
            )
            rec["status"] = "ok"
            rec["location"] = str(getattr(ds, "location", dest))
            log(f"  ✓ 完成: {rec['location']}")
        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = repr(e)
            log(f"  ✗ 失败: {e}")
            log("    常见原因：workspace/project/version 不匹配、key 无权限、网络不通")
            log("    对策：浏览器打开数据集页 → Dataset → Download Dataset → 选 yolov8 → 下载 ZIP，")
            log(f"         解压到 {dest}")
        records.append(rec)

    dump_json(
        {"sources": SOURCES, "results": records, "api_key_present": True},
        EVAL_DIR / "download_log.json",
    )
    ok = sum(1 for r in records if r["status"] == "ok")
    log(f"下载结束：成功 {ok}/{len(records)}")
    log("下一步：python step1_scan_images.py")


if __name__ == "__main__":
    main()
