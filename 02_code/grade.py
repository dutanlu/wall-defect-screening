# -*- coding: utf-8 -*-
"""
grade.py —— 把「毫米」翻译成「规范等级」（第三段：规则代码负责国标分级）

这是 detection -> assessment 的跃迁，也是本项目第二个核心创新点：
输出不是框，而是**可直接对接房屋体检「一房一档」台账的结论**。

依据：
  GB 50010-2010《混凝土结构设计规范》表 3.4.4 —— 最大裂缝宽度限值
      一类环境（室内干燥）  0.30 mm
      二类环境（露天/潮湿）  0.20 mm
      三类环境（干湿交替/海风）0.20 mm
      规范明确：「裂缝宽度以构件混凝土表面实测最大值」为准

  JGJ 125-2016《危险房屋鉴定标准》—— 危险点判据
      梁板受力主筋处横向/斜裂缝 > 0.50 mm  → 危险
      板受拉裂缝 > 1.00 mm                  → 危险
      房屋危险性分级：A(无危险点) / B(有个别危险点) / C(局部危险) / D(整体危险)

  重庆《房屋结构施工质量评定与安全性鉴定标准》（地方标准，作参考）
      弯曲裂缝：室内正常环境主要构件 > 0.30 → c_u 级，> 0.50 → d_u 级

重要边界（答辩必须主动说明）：
  本系统输出的是**筛查层面的风险提示**，不是法定鉴定结论。
  达到 C/D 级或出现危险点判据时，应触发「请专业机构进场鉴定」的流程，
  而不是由本系统直接下结论。这个「知道自己边界」的设计正是我们的亮点。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from gsd import COMFORTABLE_PX, GB50010_LIMITS_MM, JGJ125_DANGER_MM
from measure import Measurement

# 环境类别（决定裂缝宽度限值用哪个）
ENV_CLASSES = list(GB50010_LIMITS_MM.keys())

# 房屋危险性等级（JGJ 125-2016）
RISK_LEVELS = {
    "A": "结构承载力能满足正常使用要求，未发现危险点，房屋结构安全。",
    "B": "结构承载力基本能满足正常使用要求，个别构件存在危险点，但不影响主体结构安全。",
    "C": "部分承重结构承载力不能满足正常使用要求，局部出现险情，构成局部危房。",
    "D": "承重结构承载力已不能满足正常使用要求，房屋整体出现险情，构成整幢危房。",
}


@dataclass
class DefectGrade:
    """单个缺陷的分级结论。"""
    cls_name: str
    bbox_xyxy: tuple
    metric_name: str          # 分级所依据的量，如 "width_max_mm"
    metric_value: float
    limit_mm: float | None    # 适用限值；None 表示该类无宽度限值
    exceeded: bool            # 是否超限
    severity: str             # ok / attention / danger
    reason: str               # 判定理由（可直接进报告）
    confidence_note: str = "" # 口径提醒（如测量置信度）
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox_xyxy"] = list(self.bbox_xyxy)
        return d


def grade_crack(meas: Measurement, env_class: str = ENV_CLASSES[1],
                is_main_rebar_zone: bool = False,
                is_slab_tension: bool = False) -> DefectGrade:
    """
    裂缝分级。判据优先级：JGJ 125 危险点 > GB 50010 环境限值。

    env_class            : 环境类别，决定 GB 50010 限值
    is_main_rebar_zone   : 是否位于梁板受力主筋处（触发 0.5mm 危险判据）
    is_slab_tension      : 是否板受拉裂缝（触发 1.0mm 危险判据）
    """
    w = meas.width_max_mm
    reasons = []
    severity = "ok"
    limit = GB50010_LIMITS_MM.get(env_class, 0.30)
    exceeded = False

    # 1) JGJ 125 危险点判据优先
    if is_slab_tension and w > JGJ125_DANGER_MM["板受拉裂缝"]:
        severity = "danger"
        exceeded = True
        reasons.append(f"板受拉裂缝宽 {w:.2f}mm > {JGJ125_DANGER_MM['板受拉裂缝']:.2f}mm"
                       f"（JGJ 125-2016 危险点判据）")
        limit = JGJ125_DANGER_MM["板受拉裂缝"]
    elif is_main_rebar_zone and w > JGJ125_DANGER_MM["梁板受力主筋处横向/斜裂缝"]:
        severity = "danger"
        exceeded = True
        reasons.append(f"梁板受力主筋处裂缝宽 {w:.2f}mm > "
                       f"{JGJ125_DANGER_MM['梁板受力主筋处横向/斜裂缝']:.2f}mm"
                       f"（JGJ 125-2016 危险点判据）")
        limit = JGJ125_DANGER_MM["梁板受力主筋处横向/斜裂缝"]

    # 2) GB 50010 环境限值
    if severity != "danger":
        if w > limit:
            exceeded = True
            # 超过限值 2 倍以上，提示更严重
            severity = "danger" if w > 2 * limit else "attention"
            reasons.append(f"裂缝宽 {w:.2f}mm > 限值 {limit:.2f}mm"
                           f"（GB 50010-2010，{env_class}）")
        else:
            reasons.append(f"裂缝宽 {w:.2f}mm 未超限值 {limit:.2f}mm"
                           f"（GB 50010-2010，{env_class}）")

    conf = ""
    # 自检 P2-1：此处原为魔数 8，与 gsd.COMFORTABLE_PX 是同一个物理门槛
    # （「至少 8px 才能稳定做骨架化/距离变换」，见 _extract_grading_rules.py），
    # 两处各写一份会在调参时静默漂移，故统一引用常量。
    if meas.width_max_px < COMFORTABLE_PX:
        conf = (f"注意：该裂缝在图像中最大仅 {meas.width_max_px:.1f}px，"
                f"测量置信度偏低，结论仅供筛查参考，建议靠近复拍。")
        if severity == "ok":
            severity = "attention"

    return DefectGrade(
        cls_name=meas.cls_name, bbox_xyxy=meas.bbox_xyxy,
        metric_name="width_max_mm", metric_value=round(w, 3),
        limit_mm=limit, exceeded=exceeded, severity=severity,
        reason="；".join(reasons), confidence_note=conf,
        extra={"length_mm": round(meas.length_mm, 1),
               "elongation": round(meas.elongation, 2),
               "env_class": env_class,
               "px_width_max": round(meas.width_max_px, 2)},
    )


def grade_blob(meas: Measurement, elem_type: str = "wall") -> DefectGrade:
    """
    块状缺陷（剥落/空鼓/泛碱/露筋）分级。

    目前国内对单个剥落/空鼓没有像裂缝那样统一的毫米级限值，
    实际房屋体检更看「面积占比 + 是否露筋 + 是否影响承重」。
    因此这里采用**面积占比**为主判据，并给出明确的口径说明 ——
    诚实标注「本判据为工程经验值，非国标强制限值」，这比编造一个阈值更严谨。

    elem_type: wall(墙体) / column(柱) / beam(梁) / slab(板)，影响宽容度
    """
    ratio = meas.area_ratio
    reasons = []
    severity = "ok"

    # 面积占比经验判据（工程经验值，非国标）
    if ratio >= 0.15:
        severity = "danger"
        reasons.append(f"缺陷面积占局部区域 {ratio:.1%} >= 15%（经验阈值），"
                       f"存在局部承载力削弱风险")
    elif ratio >= 0.05:
        severity = "attention"
        reasons.append(f"缺陷面积占局部区域 {ratio:.1%} >= 5%（经验阈值），建议记录观察")
    else:
        reasons.append(f"缺陷面积占局部区域 {ratio:.1%}，占比小")

    # 露筋是最敏感的类别：钢筋失去保护层即开始锈蚀，长期削弱截面
    if meas.cls_name == "exposed_rebar":
        severity = "danger" if severity == "ok" else severity
        reasons.append("存在露筋：钢筋失去混凝土保护层后将持续锈蚀，"
                       "削弱有效截面并可能胀裂混凝土，建议尽快处理")
    elif meas.cls_name == "delamination":
        if severity == "ok":
            severity = "attention"
        reasons.append("空鼓/分层：存在后续剥落风险，应纳入观察或处置清单")

    return DefectGrade(
        cls_name=meas.cls_name, bbox_xyxy=meas.bbox_xyxy,
        metric_name="area_ratio", metric_value=round(ratio, 4),
        limit_mm=None, exceeded=(severity != "ok"),
        severity=severity, reason="；".join(reasons),
        confidence_note="本项判据为工程经验阈值，非国标强制限值，仅供筛查排序使用。",
        extra={"area_m2": round(meas.area_m2, 5),
               "equiv_diameter_px": round(meas.equiv_diameter_px, 1),
               "elem_type": elem_type},
    )


def grade_moss(meas: Measurement, elem_type: str = "wall") -> DefectGrade:
    """
    苔藓/生物附着 —— **不作为结构缺陷定级，最高只到 attention**。

    为什么单独处理：苔藓本身不削弱承载力，但它是「长期潮湿」的可靠指示物。
    长期潮湿会推动泛碱、冻融剥落、钢筋锈蚀。
    若按通用块状规则走，一块 15% 面积的苔藓会被判成 danger，
    进而把整栋楼推到 C 级 —— 这是典型的**误报**，会让筛查结果失去可信度。
    因此这里刻意把上限压在 attention，并在 reason 里说明真实含义（渗漏线索）。
    """
    ratio = meas.area_ratio
    if ratio >= 0.05:
        severity = "attention"
        reason = (f"苔藓/生物附着面积占局部 {ratio:.1%}：本身不削弱结构，"
                  f"但指示该部位长期潮湿，存在渗漏或排水不畅，"
                  f"是泛碱/冻融剥落/钢筋锈蚀的前兆，建议查明水源")
    else:
        severity = "ok"
        reason = f"零星苔藓附着（占局部 {ratio:.1%}），暂作清洁维护事项"

    return DefectGrade(
        cls_name=meas.cls_name, bbox_xyxy=meas.bbox_xyxy,
        metric_name="area_ratio", metric_value=round(ratio, 4),
        limit_mm=None, exceeded=(severity != "ok"),
        severity=severity, reason=reason,
        confidence_note="苔藓为潮湿指示物，非结构缺陷；本项不计入承载能力判定。",
        extra={"area_m2": round(meas.area_m2, 5),
               "equiv_diameter_px": round(meas.equiv_diameter_px, 1),
               "elem_type": elem_type, "is_structural": False},
    )


def grade_all(measurements: list[Measurement], env_class: str = ENV_CLASSES[1],
              elem_type: str = "wall",
              is_main_rebar_zone: bool = False,
              is_slab_tension: bool = False) -> list[DefectGrade]:
    """
    批量分级，自动按类别选择判据。

    is_main_rebar_zone / is_slab_tension —— JGJ 125-2016 两级危险点判据的开关。

    接线履历（自检报告 P1-6）：这两个开关在 grade_crack() 里早已实现，但
    **从未被端到端链路传入过** —— 于是 0.50mm（主筋处）/ 1.00mm（板受拉）
    两级危险判据在 Web 演示中永远不触发，只有 GB 50010 的 0.30/0.20mm 生效。
    现在由 app.py 的「构件部位」勾选组与 pipeline.py 的 `--jgj125` 显式传入。

    默认 False，因此不勾选时输出与接线前**逐字节一致**。
    """
    out = []
    for m in measurements:
        if m.cls_name == "crack":
            out.append(grade_crack(m, env_class=env_class,
                                   is_main_rebar_zone=is_main_rebar_zone,
                                   is_slab_tension=is_slab_tension))
        elif m.cls_name == "moss":
            # 苔藓走专用规则：只指示潮湿，不作为结构缺陷升级
            out.append(grade_moss(m, elem_type=elem_type))
        else:
            out.append(grade_blob(m, elem_type=elem_type))
    return out


def building_risk_level(grades: list[DefectGrade]) -> dict:
    """
    由缺陷集合推出房屋危险性等级（对齐 JGJ 125-2016 的 A/B/C/D 思路）。

    这是一个**筛查层面的粗判**，用于把巡检结果排序、决定谁先请专业机构进场，
    不能替代法定鉴定。实现上采用保守规则：
      - 有任何 danger 且数量 >= 3，或 danger 中存在露筋 → C
      - 有 danger → B
      - 仅有 attention → B
      - 全 ok（或无结构类缺陷）→ A

    ⚠️ **只统计结构类缺陷**。苔藓这类非结构指示物（见 grade_moss）即使
    数量很多、severity=attention，也不参与 C 级升级 —— 否则一次满墙苔藓
    会把建筑判成 C 级，属典型误报。非结构项另行汇总为「潮湿线索」返回。
    """
    # 非结构类（苔藓）：只指示潮湿，不参与承载能力分级
    structural = [g for g in grades
                  if (g.extra or {}).get("is_structural", True)]

    n_danger = sum(1 for g in structural if g.severity == "danger")
    n_attention = sum(1 for g in structural if g.severity == "attention")
    has_rebar = any(g.cls_name == "exposed_rebar" and g.severity == "danger"
                    for g in structural)
    n_moisture = sum(1 for g in grades if not (g.extra or {}).get("is_structural", True)
                     and g.severity != "ok")

    if n_danger >= 3 or has_rebar:
        level = "C"
    elif n_danger >= 1:
        level = "B"
    elif n_attention >= 1:
        level = "B"
    else:
        level = "A"

    return {
        "level": level,
        "level_desc": RISK_LEVELS[level],
        "n_defects": len(grades),
        "n_danger": n_danger,
        "n_attention": n_attention,
        "n_moisture_hints": n_moisture,
        "structural_defects": len(structural),
        "need_professional_inspection": level in ("B", "C", "D"),
        "disclaimer": ("本等级为图像筛查层面的风险排序结果，"
                       "依据 GB 50010-2010 与 JGJ 125-2016 的判据做粗略对齐，"
                       "不构成法定鉴定结论。达到 B/C 级应转由具备资质的"
                       "专业机构进场鉴定。"),
    }
