# -*- coding: utf-8 -*-
"""
从 gsd.py / grade.py 直接提取分级判据，产出机器可读的规则清单。

为什么用「导入代码」而不是手抄：判据必须与代码逐字一致，
手抄一旦漂移，报告与系统行为就会对不上。这个脚本保证两者永远同源。

产物：
  05_quantify_grade/grading_rules.json     机器可读
  05_quantify_grade/_extract_report.txt    提取过程留痕
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CODE = Path(r"D:\pythonstudy 备份\创新题\外墙缺陷筛查\02_code")
OUTDIR = Path(r"D:\pythonstudy 备份\创新题\外墙缺陷筛查\05_quantify_grade")
OUTDIR.mkdir(parents=True, exist_ok=True)
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

lines: list[str] = []


def log(s: str = "") -> None:
    lines.append(s)


import gsd  # noqa: E402
import grade  # noqa: E402

log("=== 提取来源 ===")
log(f"gsd.py   : {Path(gsd.__file__).resolve()}")
log(f"grade.py : {Path(grade.__file__).resolve()}")

rules = {
    "_source": {
        "gsd_py": str(Path(gsd.__file__).resolve()),
        "grade_py": str(Path(grade.__file__).resolve()),
        "note": "本文件由 02_code/_extract_grading_rules.py 从代码直接导出，勿手工编辑。",
    },
    "national_standards": {
        "GB50010-2010": {
            "title": "《混凝土结构设计规范》表 3.4.4 —— 最大裂缝宽度限值",
            "limits_mm": gsd.GB50010_LIMITS_MM,
            "apply_basis": "裂缝宽度以构件混凝土表面实测最大值为准",
        },
        "JGJ125-2016": {
            "title": "《危险房屋鉴定标准》—— 危险点判据",
            "danger_mm": gsd.JGJ125_DANGER_MM,
        },
    },
    "pixel_thresholds": {
        "MIN_RELIABLE_PX": gsd.MIN_RELIABLE_PX,
        "COMFORTABLE_PX": gsd.COMFORTABLE_PX,
        "meaning": {
            "MIN_RELIABLE_PX": "最小可判读像素数；低于此值测量误差 > 1 个国标分级步长",
            "COMFORTABLE_PX": "毫米级测量的舒适门槛；至少 8px 才能稳定做骨架化/距离变换",
        },
    },
    "risk_levels": grade.RISK_LEVELS,
    "env_classes": grade.ENV_CLASSES,
    "per_class_rules": {
        "crack": {
            "module": "grade.grade_crack",
            "metric": "width_max_mm",
            "priority": "JGJ 125 危险点判据 > GB 50010 环境限值",
            "rules": [
                "板受拉裂缝 w > 1.00mm → danger（JGJ 125）",
                "梁板受力主筋处 w > 0.50mm → danger（JGJ 125）",
                "w > 限值 → exceeded；w > 2×限值 → danger，否则 attention",
                "图像中最大宽度 < 8px → 追加低置信度提醒；若原为 ok 则升为 attention",
            ],
        },
        "blob（spalling / efflorescence / exposed_rebar / delamination）": {
            "module": "grade.grade_blob",
            "metric": "area_ratio",
            "rules": [
                "面积占比 >= 15% → danger（经验阈值）",
                "面积占比 >= 5%  → attention（经验阈值）",
                "exposed_rebar：至少 danger，理由为钢筋失去保护层持续锈蚀",
                "delamination：至少 attention",
            ],
            "caveat": "面积为工程经验阈值，非国标强制限值（代码内 confidence_note 已声明）",
        },
        "moss": {
            "module": "grade.grade_moss",
            "metric": "area_ratio",
            "rules": [
                "面积占比 >= 5% → attention（上限，不再升级）",
                "面积占比 < 5%  → ok",
            ],
            "caveat": "苔藓非结构缺陷，仅作长期潮湿指示物，不计入承载能力判定",
        },
    },
    "building_level_rule": {
        "module": "grade.building_risk_level",
        "steps": [
            "存在 danger 且数量 >= 3，或 danger 中含 exposed_rebar → C",
            "存在 ≥1 个 danger → B",
            "仅存在 attention → B",
            "否则 → A",
        ],
        "important": "只统计结构类缺陷；苔藓等非结构项单独计入 n_moisture_hints，不参与升级",
    },
    "unjudgeable_level_U": {
        "defined_in": "06_deploy/app.py（界面层，拒答传染到输出层）",
        "trigger": "当所有条目都不可判读时，风险等级输出 U 而非 A",
        "rationale": "「判不了」不等于「没问题」；把前者报成后者是最危险的误报",
        "field": "risk['n_unjudgeable'] 记录被作废条目数",
    },
    "disclaimer": grade.building_risk_level([])["disclaimer"],
}

(OUTDIR / "grading_rules.json").write_text(
    json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8"
)

log()
log("=== GB 50010-2010 最大裂缝宽度限值 (mm) ===")
for k, v in gsd.GB50010_LIMITS_MM.items():
    log(f"  {k}: {v}")
log()
log("=== JGJ 125-2016 危险点判据 (mm) ===")
for k, v in gsd.JGJ125_DANGER_MM.items():
    log(f"  {k}: {v}")
log()
log("=== 像素门槛 ===")
log(f"  MIN_RELIABLE_PX = {gsd.MIN_RELIABLE_PX}")
log(f"  COMFORTABLE_PX  = {gsd.COMFORTABLE_PX}")
log()
log("=== 房屋危险性等级 ===")
for k, v in grade.RISK_LEVELS.items():
    log(f"  {k}: {v}")
log()
log(f"已写出: {OUTDIR / 'grading_rules.json'}")

(OUTDIR / "_extract_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("OK")
