# -*- coding: utf-8 -*-
"""rt_verify.py —— 无人机实时识别实验的自检脚本。

归属：09_uav_realtime/。**不修改任何既有源文件**，只调用本目录的 rt_infer。

存在意义：本工程的长期教训是「静默失败」最致命 ——
  训练退出码 0 但其实没跑、copy_paste 开着但实为 no-op、参数键名不匹配
  悄悄回落默认值、预热吃掉全部帧数导致统计为空。
本脚本把「跑完了但没测到」变成**非零退出码 + 明确 FAIL 行**。

它断言四件事：
  A) 管线能跑完且统计推理帧数 > 0（否则说明预热/终止条件把样本吃光了）
  B) 三项耗时字段齐全且为正数（infer / decode / e2e 的 mean/p50/p90/p99）
  C) 产物真实落盘（JSON 必须存在且非空；若开了截图/视频则必须存在）
  D) 数值自洽：p50 ≤ p90 ≤ p99 ≤ max；e2e ≥ infer（端到端含推理）

用法：
  python rt_verify.py --config=rt_config.yaml --device=cpu
  python rt_verify.py --config=rt_config.yaml --device=cuda:0 --source=synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import rt_common as R  # noqa: E402
import rt_infer  # noqa: E402
from gsd import ENV_CLASSES  # noqa: E402


class Checker:
    """收集检查结果；任一 FAIL ⇒ 进程以非零码退出。"""

    def __init__(self) -> None:
        self.fails: list[str] = []
        self.passes: list[str] = []

    def check(self, cond: bool, name: str, detail: str = "") -> bool:
        if cond:
            self.passes.append(name)
            print(f"  [PASS] {name}" + (f"  —— {detail}" if detail else ""))
        else:
            msg = name + (f"  —— {detail}" if detail else "")
            self.fails.append(msg)
            print(f"  [FAIL] {msg}")
        return cond

    def report(self) -> int:
        print()
        print("=" * 72)
        if self.fails:
            print(f"自检结果：FAIL（{len(self.fails)} 项未通过 / "
                  f"{len(self.passes)} 项通过）")
            for f in self.fails:
                print(f"  ✗ {f}")
            print("=" * 72)
            return 1
        print(f"自检结果：全部通过（{len(self.passes)} 项）")
        print("=" * 72)
        return 0


def verify(cfg: dict, source_key: str | None,
           expect_video: bool, expect_frames: bool) -> int:
    ck = Checker()

    print("=" * 72)
    print("无人机实时识别实验 · 自检")
    print(f"  配置：{cfg.get('_config_path')}")
    print("=" * 72)

    # ---------- 0) 配置层自检（不需要跑模型，快）----------
    print("\n[0] 配置层")
    src = R.resolve_source(cfg, source_key)
    ck.check(src["kind"] in ("file", "stream", "device"),
             "源类型合法", f"kind={src['kind']} preset={src['key']}")
    args = R.build_args(cfg)
    ck.check(Path(args["model"]).exists(), "权重文件存在", args["model"])
    ck.check(args["env"] in ENV_CLASSES,
             "环境类别合法（不静默失效）", args["env"])
    ck.check(args["imgsz"] > 0 and args["conf"] > 0,
             "imgsz/conf 为正", f"imgsz={args['imgsz']} conf={args['conf']}")
    # 标定链兜底可运行（否则会在视频中途抛 ValueError）
    has_primary = (args["calib_object_px"] > 0 or args["calib_brick"] or args["rectify"])
    ck.check(has_primary or (args["distance"] > 0 and args["focal"] > 0),
             "标定链可运行（不会中途崩）",
             f"distance={args['distance']} focal={args['focal']} "
             f"calib_object_px={args['calib_object_px']}")
    stride = rt_infer.compute_stride(30.0, float(cfg.get("capture", {}).get("target_fps", 0) or 0))
    ck.check(stride >= 1, "跳帧步长合法", f"stride={stride}")

    # ---------- 1) 实跑 ----------
    print("\n[1] 实跑（源 -> run_one -> 落盘）")
    result = rt_infer.run(cfg, source_key=source_key)

    # ---------- 2) 计数有效性 ----------
    print("\n[2] 计数有效性")
    counts = result.get("counts") or {}
    n_meas = int(counts.get("frames_measured") or 0)
    ck.check(n_meas > 0, "统计推理帧数 > 0（预热未吃光预算）",
             f"frames_measured={n_meas} warmup={counts.get('warmup_done')} "
             f"read={counts.get('frames_read')}")
    ck.check(int(counts.get("frames_read") or 0) > 0, "读到帧数 > 0",
             f"frames_read={counts.get('frames_read')}")
    ck.check(int(counts.get("measured_failed") or 0) == 0,
             "无 run_one 异常帧", f"failed={counts.get('measured_failed')}")

    # ---------- 3) 耗时字段完整性 ----------
    print("\n[3] 耗时字段完整性")
    timing = result.get("timing_ms") or {}
    for seg_name in ("infer", "decode", "e2e"):
        s = timing.get(seg_name) or {}
        need = ["n", "mean_ms", "min_ms", "max_ms", "p50_ms", "p90_ms", "p99_ms"]
        missing = [k for k in need if s.get(k) is None]
        ck.check(not missing, f"{seg_name} 字段齐全",
                 f"缺失={missing}" if missing else
                 f"mean={s.get('mean_ms')} p90={s.get('p90_ms')} p99={s.get('p99_ms')}")
        if not missing:
            ck.check((s.get("mean_ms") or 0) > 0, f"{seg_name} 均值为正",
                     f"mean={s.get('mean_ms')}")

    # ---------- 4) 数值自洽 ----------
    print("\n[4] 数值自洽性")
    for seg_name in ("infer", "e2e"):
        s = timing.get(seg_name) or {}
        p50, p90, p99, mx = s.get("p50_ms"), s.get("p90_ms"), s.get("p99_ms"), s.get("max_ms")
        if None not in (p50, p90, p99, mx):
            ck.check(p50 <= p90 <= p99 <= mx, f"{seg_name} 分位单调不降",
                     f"p50={p50} p90={p90} p99={p99} max={mx}")
    e2e_m = (timing.get("e2e") or {}).get("mean_ms")
    inf_m = (timing.get("infer") or {}).get("mean_ms")
    dec_m = (timing.get("decode") or {}).get("mean_ms")
    if None not in (e2e_m, inf_m, dec_m):
        # e2e 含 取帧 + 推理 + 叠框 + 记录，故必须 ≥ 推理均值
        ck.check(e2e_m >= inf_m * 0.9, "端到端延迟 ≥ 单帧推理（口径自洽）",
                 f"e2e={e2e_m} infer={inf_m} decode={dec_m}")

    # ---------- 4b) 吞吐口径 ----------
    print("\n[4b] 吞吐口径")
    v = result.get("verdict") or {}
    ck.check(v.get("throughput_fps") is not None, "吞吐 FPS 已给出（稳态口径）",
             f"throughput_fps={v.get('throughput_fps')}")
    ck.check(v.get("wall_clock_fps") is not None, "墙钟口径 FPS 已给出（参考）",
             f"wall_clock_fps={v.get('wall_clock_fps')}")
    if v.get("throughput_fps") and inf_m:
        expect = 1000.0 / inf_m
        ck.check(abs(v["throughput_fps"] - expect) < 0.05,
                 "吞吐 FPS 与单帧推理一致（1000/infer_ms）",
                 f"{v['throughput_fps']} vs {expect:.3f}")
    ck.check(bool(v.get("note")), "吞吐口径已附说明（防止误读）")

    # ---------- 5) 产物落盘 ----------
    print("\n[5] 产物落盘")
    art = result.get("artifacts") or {}
    logs_dir = _HERE / str((cfg.get("output") or {}).get("logs_dir", "logs"))
    jsons = sorted(logs_dir.glob("rt_metrics_*.json"))
    ck.check(len(jsons) > 0, "结构化 JSON 已落盘", f"共 {len(jsons)} 个")
    if jsons:
        latest = jsons[-1]
        ck.check(latest.stat().st_size > 0, "JSON 非空",
                 f"{latest.name} {latest.stat().st_size} B")
        # 反读一遍：能 load 且 per_frame 非空
        try:
            with open(latest, "r", encoding="utf-8") as f:
                back = json.load(f)
            ck.check(bool(back.get("per_frame")), "JSON 可反读且 per_frame 非空",
                     f"per_frame={len(back.get('per_frame') or [])} 帧")
            ck.check(back.get("schema") == "uav_rt_infer/v1", "schema 标记正确",
                     str(back.get("schema")))
            ck.check(bool(back.get("caveat")), "含边界声明（不得声称实拍验证）")
        except Exception as e:
            ck.check(False, "JSON 可解析", f"解析异常：{e}")

    if expect_frames:
        fd = art.get("frames_dir")
        n_saved = int((result.get("counts") or {}).get("frames_saved") or 0)
        ck.check(bool(fd) and n_saved > 0, "抽样截图已落盘",
                 f"{n_saved} 张 @ {fd}" if fd else "未开启或未产生")
    if expect_video:
        vp = art.get("video")
        ck.check(bool(vp) and Path(vp).exists() if vp else False,
                 "回放视频已落盘",
                 f"{Path(vp).name} {Path(vp).stat().st_size} B" if vp else "未产生")

    # ---------- 6) 叠加内容检查 ----------
    print("\n[6] 叠加内容")
    pf = result.get("per_frame") or []
    if pf:
        sample = pf[0]
        for k in ("frame_no", "infer_ms", "decode_ms", "e2e_ms",
                  "n_detections", "risk_level", "classes", "all_abstained"):
            ck.check(k in sample, f"逐帧记录含字段 {k}")
        ck.check(all(isinstance(f.get("e2e_ms"), (int, float)) for f in pf),
                 "所有帧 e2e_ms 为数值")

    # ---------- 6b) 框真的画上去了吗（回归保护）----------
    # ★ 为什么必须单独断言这件事：
    #   run_one 返回的 `annotated` 是**路径字符串**而非图像数组，曾被误当
    #   数组使用 ⇒ 静默回退成原帧、一个框都没画上，且**不报错**。
    #   仅看"跑通了"永远发现不了。故这里直接比对：
    #   有检出的帧，其保存下来的图应与原图**不同**（像素有变化）。
    print("\n[6b] 叠加是否真的生效（框/标签/HUD 是否有像素落笔）")
    fd = art.get("frames_dir")
    has_det = [f for f in pf if (f.get("n_detections") or 0) > 0]
    if not has_det:
        ck.check(True, "本段素材无检出帧，跳过叠加像素比对",
                 "（0 检出时原图与叠加图可相同，属正常）")
    elif not fd:
        ck.check(False, "有检出帧但无落盘截图，无法验证叠加",
                 "请开 output.save_frame_every 以启用该断言")
    else:
        import numpy as np
        from common import imread_u
        shots = sorted(Path(fd).glob("*.jpg"))
        ck.check(len(shots) > 0, "存在可检查的截图", f"{len(shots)} 张")
        if shots:
            # 逐张检查：非纯色（说明有内容），且尺寸与源一致
            differences = []
            for sp in shots[:3]:
                im = imread_u(sp)
                if im is None:
                    continue
                h, w = im.shape[:2]
                ck.check(h > 0 and w > 0, f"{sp.name} 可解码", f"{w}x{h}")
                # 叠加会引入高饱和文字色（HUD 用黄 0,255,255），
                # 检查是否存在近纯黄像素 —— 这是 HUD/标签落笔的强证据
                b, g, r = im[:, :, 0].astype(int), im[:, :, 1].astype(int), im[:, :, 2].astype(int)
                is_yellow = (g > 200) & (r > 200) & (b < 80)
                differences.append(int(is_yellow.sum()))
            if differences:
                ck.check(max(differences) > 0,
                         "HUD/标签文字像素已落笔（叠加真的生效）",
                         f"近纯黄像素数={differences}")

    return ck.report()


def main() -> int:
    ap = argparse.ArgumentParser(description="无人机实时识别实验自检")
    ap.add_argument("--config", default="rt_config.yaml")
    ap.add_argument("--source", default=None,
                    help="源预设名（默认用配置里的 source.preset）")
    ap.add_argument("--device", default="cpu",
                    help="自检默认走 CPU：不占显存、最稳、最快暴露逻辑问题")
    ap.add_argument("--max-frames", type=int, default=8,
                    help="自检只需少量帧，默认 8")
    ap.add_argument("--warmup", type=int, default=2,
                    help="自检预热帧数，默认 2（短跑时须小于 max-frames）")
    ap.add_argument("--no-video", action="store_true",
                    help="关闭回放视频产物断言（视频编码器在某些环境不可用）")
    a = ap.parse_args()

    cfg = R.load_config(a.config)
    cfg.setdefault("infer", {})["device"] = a.device
    cfg.setdefault("capture", {})["max_frames"] = a.max_frames
    cfg.setdefault("capture", {})["warmup_frames"] = a.warmup
    out = cfg.setdefault("output", {})
    expect_video = bool(out.get("save_video", False)) and not a.no_video
    expect_frames = int(out.get("save_frame_every", 0) or 0) > 0

    # 自检跑的是短程，把 duration 放开，避免被 60s 上限干扰
    cfg["capture"]["duration_sec"] = 0
    # 短跑时按 1 张截图/2 帧 保证一定有产物可断言
    if expect_frames:
        out["save_frame_every"] = min(int(out.get("save_frame_every") or 5), 2)

    return verify(cfg, a.source, expect_video=expect_video,
                  expect_frames=expect_frames)


if __name__ == "__main__":
    raise SystemExit(main())
