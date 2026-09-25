# -*- coding: utf-8 -*-
"""
advice.py —— C 路：采集闭环的「反解物理约束」模块

===== 为什么需要（本模块要解决的问题）=====

项目已有 gsd.calibrate_by_camera()：**正向**（给定距离 → 算 GSD → 判能不能测）。
缺的是**反向**：用户拿手机拍了一张照片、系统说「不可判读」——
用户接下来该怎么做？「靠近点」是没用的废话，必须给出**具体到米**的动作。

本模块把「拒答」从**终点**变成**起点**：
    要判 0.3mm 裂缝  ⇒  需要 GSD ≤ 0.1 mm/px  ⇒  24mm 主摄须靠近到 ≤ 0.27 m
                                               ⇒  （若已在 1.2 m）请再靠近 0.93 m

===== 数学（与 calibrate_by_camera 严格互逆，可往返验证）=====

    fov        = 2·atan(sensor_w / (2·f35))          # 该 35mm 等效焦距的视场角
    cover(d)   = 2·d·tan(fov/2)                      # 距离 d 处的覆盖宽度(mm)
    gsd(d)     = cover(d) / image_width_px           # 每像素代表多少 mm
             = d · 1000 · (2·tan(fov/2)) / W         # 线性于 d ★
    ⇒ 反解： gsd_target 对应  d_max = gsd_target · W / (1000 · 2·tan(fov/2))

★ 关键性质：gsd 与 d **成正比** ⇒ 反解是闭式的、无歧义的。
（这也是能被评委用计算器复算的原因 —— 与项目一贯风格一致。）

===== 本模块不做的事（诚实边界）=====
· 不假设用户知道自己的焦距：提供常见档位（超广角 13mm / 主摄 24mm / 长焦 48mm）。
· 不解决「靠太近拍不全」：会同时给出「该距离下的覆盖宽度」，
  让用户看到取舍（近了更准但画面更小）——这正是赛道二「硬约束取舍」的体现。
· 距离是**沿光轴到墙面**的直线距离，不是斜距；斜拍请先正射校正（rectify.py）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

# 常见手机镜头（35mm 等效焦距）
LENS_PRESETS = {
    "超广角 13mm": 13.0,
    "主摄 24mm": 24.0,
    "长焦 48mm": 48.0,
}
DEFAULT_SENSOR_W_MM = 36.0


@dataclass
class DistanceAdvice:
    """「要判 X mm 目标，需在多远处拍」的完整回答。"""
    target_mm: float
    required_gsd: float          # mm/px
    lens_label: str
    focal_35mm: float
    image_width_px: int
    current_distance_m: float | None
    max_distance_m: float        # 满足要求的最远距离
    cover_mm_at_max: float       # 在该距离下的覆盖宽度（画面实际能拍多宽）
    action: str                  # 给用户的一句话动作（CLI 用）
    feasible: bool               # 是否物理可行（有些档位连贴脸都不够）

    # ⚠️ 关于 `action` 字段（重要，避免误读）：
    #   `action` 是给 **CLI / 命令行** 读的一段完整句子。
    #   Web 界面（`06_deploy/app.py`）**不使用** `action`，而是用
    #   `required_gsd / max_distance_m / cover_mm_at_max` **自己拼**文案
    #   （因为它要嵌进 Markdown、还要带 `【补拍建议】` 前缀）。
    #   ⇒ 两者的**数字必然一致**（同源），但**措辞不同**。
    #   若你要断言界面文案，请断言 app.py 的措辞；要断言数值，用这三个字段。
    def to_dict(self) -> dict:
        return asdict(self)


def _fov_rad(focal_35mm: float, sensor_w_mm: float = DEFAULT_SENSOR_W_MM) -> float:
    return 2.0 * math.atan(sensor_w_mm / (2.0 * focal_35mm))


def gsd_at_distance(distance_m: float, image_width_px: int,
                    focal_35mm: float = 24.0,
                    sensor_w_mm: float = DEFAULT_SENSOR_W_MM) -> float:
    """正向：距离 → GSD（mm/px）。与 gsd.calibrate_by_camera 同式。"""
    fov = _fov_rad(focal_35mm, sensor_w_mm)
    cover_mm = 2.0 * (distance_m * 1000.0) * math.tan(fov / 2.0)
    return cover_mm / image_width_px


def distance_for_gsd(gsd_target: float, image_width_px: int,
                     focal_35mm: float = 24.0,
                     sensor_w_mm: float = DEFAULT_SENSOR_W_MM) -> float:
    """反向：目标 GSD → 最远拍摄距离（m）。"""
    fov = _fov_rad(focal_35mm, sensor_w_mm)
    return gsd_target * image_width_px / (1000.0 * 2.0 * math.tan(fov / 2.0))


def advise_distance(target_mm: float,
                    image_width_px: int = 4000,
                    lens_label: str = "主摄 24mm",
                    min_reliable_px: float = 3.0,
                    current_distance_m: float | None = None,
                    sensor_w_mm: float = DEFAULT_SENSOR_W_MM) -> DistanceAdvice:
    """
    核心入口：给目标尺寸，回答「该怎么拍」。

    参数
    ----
    target_mm : 要判读的目标宽度（如裂缝 0.3mm、剥落 10mm）
    image_width_px : 图像宽度像素（决定像素预算）
    lens_label : LENS_PRESETS 的键
    min_reliable_px : 可靠判读的最少像素数（项目现行 MIN_RELIABLE_PX = 3）
    current_distance_m : 若已知当前拍摄距离，则给出「还需靠近多少」
    """
    focal = LENS_PRESETS.get(lens_label, 24.0)
    # 可靠判读 ⇒ 目标至少占 min_reliable_px 像素 ⇒ gsd ≤ target / min_px
    required_gsd = target_mm / min_reliable_px
    d_max = distance_for_gsd(required_gsd, image_width_px, focal, sensor_w_mm)
    cover_mm = 2.0 * (d_max * 1000.0) * math.tan(_fov_rad(focal, sensor_w_mm) / 2.0)

    # 物理可行性：最远距离须 ≥ 最近对焦（取 0.10 m 为手机典型最近工作距离）
    NEAR_LIMIT_M = 0.10
    feasible = d_max >= NEAR_LIMIT_M

    if feasible:
        if current_distance_m is not None and current_distance_m > d_max:
            delta = current_distance_m - d_max
            action = (f"当前约 {current_distance_m:.2f} m，需**再靠近 {delta:.2f} m**"
                      f"（至 ≤ {d_max:.2f} m）")
        elif current_distance_m is not None:
            action = f"当前约 {current_distance_m:.2f} m，**已满足**（上限 {d_max:.2f} m）"
        else:
            action = f"请靠近至 **≤ {d_max:.2f} m** 拍摄"
        action += f"；该距离下画面覆盖约 {cover_mm/1000:.2f} m 宽"
    else:
        # ⚠️ 措辞要点：`d_max` 是「满足 GSD 要求的**最远**距离」。
        #   它 < 最近工作距离 ⇒ 意味着「你得站到比最近对焦还近」⇒ 物理不可行。
        #   若把它写成"无论多近都不够"，与「最远距离」的语义是**反的**（越近越准），
        #   容易被误读 ⇒ 这里明确说清「即便贴到最近工作距离也不够」。
        action = (f"用「{lens_label}」即使贴到最近工作距离（{NEAR_LIMIT_M*100:.0f} cm）"
                  f"也不够（该组合的理论最大距离仅 {d_max*100:.1f} cm）"
                  f"⇒ 请改用更长焦段或增大图像宽度")
        # 与 Web 界面（app.py）的措辞保持**语义一致**：
        #   app.py 写「即使贴到 {d_max*100:.0f}cm 也不够」
        #   此处写「即使贴到最近工作距离(10cm)也不够（理论最大 X cm）」
        #   两者数字同源（均来自 d_max），措辞更明确，不再有"无论多近"的反语义。

    return DistanceAdvice(
        target_mm=target_mm,
        required_gsd=round(required_gsd, 4),
        lens_label=lens_label,
        focal_35mm=focal,
        image_width_px=image_width_px,
        current_distance_m=current_distance_m,
        max_distance_m=round(d_max, 3),
        cover_mm_at_max=round(cover_mm, 1),
        action=action,
        feasible=feasible,
    )


def advising_table(image_width_px: int = 4000,
                   current_distance_m: float | None = None,
                   lens_label: str = "主摄 24mm") -> list[dict]:
    """C 路答辩用的一张表：各典型目标尺寸 → 该怎么拍。"""
    rows = []
    for name, mm in [("裂缝定级 0.3mm", 0.3), ("裂缝定级 0.5mm", 0.5),
                     ("明显裂缝 1.0mm", 1.0), ("露筋 ~12mm", 12.0),
                     ("小剥落 10mm", 10.0), ("中等剥落 50mm", 50.0)]:
        a = advise_distance(mm, image_width_px, lens_label,
                            current_distance_m=current_distance_m)
        rows.append(dict(item=name, **a.to_dict()))
    return rows


# --------------------------------------------------------------------------
# 与 gsd.calibrate_by_camera 的互逆性自检（往返验证，可作为单测）
# --------------------------------------------------------------------------
def _selftest_roundtrip() -> None:
    for d in (0.27, 0.5, 1.0, 2.5, 5.0):
        for f in (13.0, 24.0, 48.0):
            g = gsd_at_distance(d, 4000, f)
            d2 = distance_for_gsd(g, 4000, f)
            assert abs(d2 - d) < 1e-9, f"往返不一致 d={d} f={f}: {d2}"
    print("往返自检通过：distance_for_gsd ∘ gsd_at_distance = identity")


if __name__ == "__main__":
    _selftest_roundtrip()
    print()
    print("=" * 78)
    print("C 路 · 采集建议表（图像宽 4000px · 主摄 24mm 等效）")
    print("=" * 78)
    for r in advising_table(4000):
        print(f"{r['item']:<18} 需 GSD≤{r['required_gsd']:.4f} mm/px"
              f"  ⇒ 距离≤{r['max_distance_m']:.2f} m"
              f"  覆盖{r['cover_mm_at_max']/1000:.2f} m")
    print()
    print("示例：当前距离 1.2 m，要判 0.3mm 裂缝")
    a = advise_distance(0.3, 4000, "主摄 24mm", current_distance_m=1.2)
    print("  →", a.action)
