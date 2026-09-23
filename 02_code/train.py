# -*- coding: utf-8 -*-
"""
train.py —— 外墙缺陷检测训练脚本（本项目主训练入口）

设计要点：
1. **多组对照一次跑完**，直接产出答辩必答题第 3 问「为何选该 YOLO 版本」的实测数据：
     yolov8s@640   —— 用户选定的主力配置
     yolo11s@640   —— 版本对比
     yolov8n@640   —— 规格对比（赛道二 体积<10MB 需要）
     yolov8s@1024  —— 小目标实验（裂缝/露筋是细长小目标）
   可用 --runs=v8s640,v11s640 只跑子集。

2. **workers=0**：Windows + 中文路径下多进程 dataloader 易卡死/报错，属该项目已知环境坑。
3. **中文路径安全**：Ultralytics 内部用 cv2 读图。若出现大量「corrupt image」警告，
   先确认数据集路径不含中文，或改用 run_in_ascii_path() 把数据镜像到纯英文路径再训。

用法：
  python train.py                          # 跑全部 4 组
  python train.py --runs=v8s640            # 只跑主力配置
  python train.py --epochs=5 --runs=v8s640 # 快速冒烟
  python train.py --data=D:/xxx/wall_defects.yaml

输出：
  04_结果/训练/<run_name>/**（Ultralytics 标准输出：weights/best.pt、results.csv 等）
  04_结果/评估/train_summary.json
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from common import (
    CLASSES,
    DATASET_DIR,
    DATASET_YAML_NAME,
    EVAL_DIR,
    TRAIN_DIR,
    WEIGHTS_DIR,
    argv_flag,
    dump_json,
    ensure_dirs,
    log,
    resolve_weight,
)

# --------------------------------------------------------------------------
# 训练配方（keys 用于 --runs 选择）
#   weight : 预训练权重（首次会自动下载到当前目录）
#   imgsz  : 输入尺寸
#   batch  : 批大小（12GB 显存下 1024 需降到 16 左右）
#   note   : 该组实验的目的（写进报告）
# --------------------------------------------------------------------------
RECIPES: dict[str, dict] = {
    "v8s640": {
        "weight": "yolov8s.pt", "imgsz": 640, "batch": 24,
        "note": "主力配置：精度/速度平衡，兼顾 <100ms 推理",
    },
    "v11s640": {
        "weight": "yolo11s.pt", "imgsz": 640, "batch": 24,
        "note": "版本对照：v11 vs v8（答辩「为何选该版本」依据）",
    },
    "v8n640": {
        "weight": "yolov8n.pt", "imgsz": 640, "batch": 24,
        "note": "规格对照：极致轻量，服务赛道二 (<10MB)",
    },
    "v8s1024": {
        # [2026-09-21 提速改动·用户指示] batch 12→16：12.2 GB 显存实测余量充足（640 档仅用 37%），
        # 且本文件头注释「12GB 显存下 1024 需降到 16 左右」本就按 16 设计。
        # ⚠️ 对 2026-09-21 正在跑的 V3 队列【不生效】（该队列内存中仍是 batch=12）。
        "weight": "yolov8s.pt", "imgsz": 1024, "batch": 16,
        "note": "小目标实验：裂缝/露筋为细长小目标，提高分辨率验证增益",
    },
    # ----------------------------------------------------------------------
    # [2026-09-23 新增] 数据增强实验（对照 v11s640，其余全同）
    # 动机来自本项目**实测的两条事实**，不是随便开的开关：
    #   ① 真实部署中 GSD 随拍摄距离大幅变化（约 5m / 20m / 50m 三个量级）
    #      ⇒ scale 0.5→0.9：把尺度抖动区间对齐这个物理先验（尺度鲁棒性）
    #   ② 极端长尾：rust 13685 框 vs spalling 313 框 = **43:1**
    #      ⇒ mixup 0.15：把两张图混合，让稀有类在更多上下文里出现
    # ⚠️ 原计划的 copy_paste **已实测不可用**（只支持分割任务，见 COMMON_ARGS 注释），
    #    因此本实验只用对纯检测确实生效的两项。
    # 用 `aug` 字段按实验覆盖 COMMON_ARGS 的对应项；未写 aug 的实验行为完全不变。
    # ----------------------------------------------------------------------
    # [2026-09-23 新增] 类平衡损失实验（对照 v11s640，其余全同）
    #
    # 动机：本项目 7 类实例数极度长尾 —— rust 9433 / spalling 207 = **45.6 : 1**，
    #   且 rust 一类独占训练集 74.7% 的实例。而 ultralytics 的 cls 分支是
    #   逐 (anchor, class) 的 sigmoid + BCE，**默认各类权重完全相同**，
    #   等价于「默认假设七类同等重要」—— 与巡检业务事实严重不符：
    #   裂缝/露筋的漏检代价远高于锈迹。至今本项目未对长尾做任何重采样或损失加权。
    #
    # 手段：ultralytics 内置的 `cls_pw`（utils/loss.py:438-441 + detect/train.py:157-186）
    #   权重 w_c = (1/n_c)^pw 后归一化到均值 1，逐类乘到 BCE 上。
    #   实测 pw=0.5 的权重表（logs/_CLSBAL_FULLCHECK.txt，与独立复算逐类一致）：
    #       crack 1.349 / spalling 1.622 / efflorescence 0.804 / exposed_rebar 0.748
    #       rust 0.240 / delamination 1.058 / moss 1.178
    #   把「原始实例占比 45.6:1」压缩为「有效损失占比 6.75:1」。
    #
    # ★ 开训前的有效性闸门（必做，勿省）：
    #   本项目在 copy_paste 上踩过「参数设了、不报错、零效果」的坑，
    #   故对 cls_pw 做了 L1 参数层 / L2 训练器层 / L3 损失层三重验证：
    #     L1 OK — 参数被配置系统接受，越界会被拒绝；
    #     L2 OK — 全量训练下 trainer 实测权重与公式复算逐类一致；
    #     L3 OK — 受控构造下 rust 的损失占比由 10.9% 降到 2.5%。
    #   证据：logs/_CLSBAL_VERIFY.txt、_CLSBAL_FULLCHECK.txt
    #   ⚠️ 期间发现：用 fraction<1 探测会得到「多个类权重并列」的假象
    #      （抽样把类采成 0 实例 → 公式置 1.0），**探测必须用全量**。
    #
    # 为什么选 pw=0.5 而不是 1.0：
    #   pw=1.0 是完全反频率，会把极稀有类的权重抬到 2.24、rust 压到 0.049
    #   （有效占比 1:1 拉平），对检测任务过于激进、易引入大量误检；
    #   pw=0.5 是常用折中（等价于按 1/√n 加权），保留长尾修正而不失衡。
    #
    # ★ 预期管理（诚实前置）：单种子下组间方差可能与 0.03 同量级，
    #   本实验**很可能又测不出精度差异**。它要拿的是「做了损失函数优化」的
    #   技术深度分，**不是**「提升了 mAP」的分。判读严格按 CI 半宽纪律执行。
    #
    # ★★ 2026-09-23 17:32 事故处置：workers 4 → 0（本配方单独覆盖）
    #   第 9 轮时 DataLoader worker 进程 2 抛
    #   `cv2.error: ... Failed to allocate 1228800 bytes in function 'cv::OutOfMemoryError'`
    #   ⇒ 训练直接失败（18 分钟，非 28 分钟线）。
    #   本机总内存仅 15.72 GB、训练时常剩 ~4 GB；4 个 worker 各缓存解码数据
    #   是主要内存压力源。
    #   ★ 改回 workers=0 有**双重收益**：
    #     ① 不派生子进程 ⇒ 内存最省，规避 OOM；
    #     ② 基线的 args.yaml 正是 workers=0（它 09-21 14:11 开跑、早于提速改动）
    #        ⇒ 此举**同时消除了 A/B 的变量差异**，本实验回归「只差 cls_pw」。
    #   代价：dataloader 单进程，训练变慢（基线当时约 95–125 s/epoch）。
    #   ⚠️ 这是**中途改配置后 --resume**：ultralytics 在 resume 时会读
    #      last.pt 里的 train_args，故须确认 workers 真的被改成 0（查新 args.yaml）。
    "v11s640_clsbal": {
        "weight": "yolo11s.pt", "imgsz": 640, "batch": 24,
        "note": "类平衡损失：cls_pw=0.5 按逆频率加权（对照 v11s640）",
        "aug": {"cls_pw": 0.5, "workers": 0},
    },
    "v11s640_aug": {
        "weight": "yolo11s.pt", "imgsz": 640, "batch": 24,
        "note": "数据增强：GSD 尺度扩展 + mixup（对照 v11s640）",
        "aug": {"scale": 0.9, "mixup": 0.15},
    },
}

COMMON_ARGS = dict(
    epochs=100,
    patience=25,          # 早停
    optimizer="auto",
    lr0=0.01,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=3.0,
    # 数据增强：外墙缺陷以「位置/尺度/光照」变化为主，
    # 因此 mosaic 适当下调（mosaic 会把目标缩小，对细长裂缝不利）
    mosaic=0.5,
    close_mosaic=10,
    fliplr=0.5,
    flipud=0.0,           # 垂直翻转不符合物理（墙面有重力方向）
    degrees=5.0,          # 轻微旋转，模拟拍摄角度
    scale=0.5,
    translate=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    # [2026-09-23 显式化] copy_paste 明写为 0.0（= ultralytics 默认值，语义不变）。
    # ★★ 为什么不把它当增强手段用（实测结论，勿再尝试）：
    #    ultralytics 的 `CopyPaste.__call__`（data/augment.py:1921-1924）开头是
    #        if len(labels["instances"].segments) == 0 or self.p == 0:
    #            return labels          # 原样返回
    #    即 **只对分割任务生效**。本项目是**纯检测数据集（YOLO txt 只有 bbox，
    #    没有 segments）** ⇒ CopyPaste 是个**静默 no-op**：
    #    设 copy_paste=0.4 与 0.0 训练结果完全相同，且不会有任何警告。
    #    （测过的对照：Mosaic / MixUp 都不检查 segments，对纯检测有效。）
    copy_paste=0.0,
    # 训练控制
    # [2026-09-21 提速改动·用户指示] workers 0→4：GPU 实测 14–96% 波动，瓶颈在 dataloader。
    # ⚠️ 对 2026-09-21 正在跑的 V3 队列【不生效】（配置已读入内存，该队列仍是 workers=0），
    #    仅对之后的训练生效；本机未验证 workers>0，先用 1-epoch 冒烟确认不卡死。
    workers=4,
    # [未启用] cache='ram' 本机不可开：总内存 15.7 GB、训练时常剩 ~2.5 GB；
    # 640 档约需 +2~2.5 GB、1024 档约需 +5~6 GB，有 OOM/换页风险。
    # 将来在 ≥32 GB 内存的机器上可打开下面这行：
    # cache="ram",
    val=True,
    plots=True,
    save=True,
    exist_ok=True,
    verbose=True,
    seed=42,
    deterministic=True,
)


def preflight_leak_check(data_yaml: Path) -> bool:
    """
    训练前再查一次 train/val 的文件名交集。

    为什么在训练入口再查一遍：step4 已经查过 9 项交集，但数据集是
    「可被手工修改」的目录。训练要跑几个小时，如果此时划分已被污染，
    跑出来的 mAP 全是虚高，而报告里又看不出来。宁可开跑前多花 2 秒。
    """
    try:
        from ultralytics.data.utils import check_det_dataset
        d = check_det_dataset(str(data_yaml))
    except Exception:
        return True          # 校验失败交给调用方处理
    root = Path(d.get("path", data_yaml.parent))
    names: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        rel = d.get(split)
        if not rel:
            continue
        p = Path(rel)
        p = p if p.is_absolute() else root / p
        if not p.exists():
            continue
        names[split] = {f.stem for f in p.glob("*")
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
    ok = True
    splits = sorted(names)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            inter = names[a] & names[b]
            flag = "OK" if not inter else "!! 泄漏"
            log(f"  {flag} {a} ∩ {b} = {len(inter)}")
            if inter:
                ok = False
                log(f"     示例: {sorted(inter)[:5]}")
    return ok


def ensure_arial_font() -> None:
    """
    把系统 Arial 字体预置到 ultralytics 的用户配置目录（断掉一个致命网络依赖）。

    为什么必须做：ultralytics 的 `check_det_dataset()` 结尾会无条件调用
    `check_font("Arial.ttf")`（见 ultralytics/data/utils.py:664），而
    `check_font()`（ultralytics/utils/checks.py:447）的查找顺序是：
        ① USER_CONFIG_DIR/Arial.ttf  → 有就直接返回
        ② matplotlib 系统字体里找名字含 "Arial.ttf" 的 → **大小写敏感子串匹配**
        ③ 联网从 ultralytics.com 下载
    本机在这三步上连踩两个坑：
      - 第 ② 步匹配不上：Windows 系统字体实际文件名是全小写 `arial.ttf`，
        而匹配串是 `"Arial.ttf"`，大小写不符 → 永远匹配不到；
      - 第 ③ 步必失败：本机走代理，github/ultralytics 隧道 502。
    于是每次都落到联网，而联网是否抛异常又取决于 `downloads.is_url()` 的
    探测结果 —— 2026-09-20 16:03 前后实测到两种截然不同的结局：
      - v8s1024：探测通了 → 真去下载 → 3 次重试全失败 → 抛异常 → 训练中止；
      - v11s640：探测没通 → 直接静默返回 None → 反而跑起来了。
    也就是说，**同一台机器同一个脚本，能不能开训取决于网络抖动**，
    训练入口因此不可复现。这里直接从 C:\\Windows\\Fonts 拷一份到
    USER_CONFIG_DIR，让第 ① 步就命中，彻底不碰网络。

    注意：本函数只预置 ASCII 的 `Arial.ttf`。若将来把类别名改成非 ASCII
    （中文），ultralytics 会转而请求 `Arial.Unicode.ttf`，那一步仍会联网。
    本项目类别名全为英文（见 common.CLASSES），因此不受影响。
    """
    try:
        from ultralytics.utils import USER_CONFIG_DIR
    except Exception as e:  # ultralytics 不可用时不必让训练入口崩在字体上
        log(f"!! 跳过字体预置（无法导入 ultralytics.utils）: {e}")
        return

    dst = Path(USER_CONFIG_DIR) / "Arial.ttf"
    if dst.exists():
        log(f"字体已就位: {dst}")
        return

    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\Arial.ttf"),
    ]
    for src in candidates:
        if not src.exists():
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            log(f"已预置 Arial 字体: {src} -> {dst}")
        except Exception as e:
            log(f"!! 预置 Arial 字体失败: {e}")
        return

    log("!! 未找到系统 arial.ttf，check_det_dataset 可能因联网下载失败而中止训练")
    log("   -> 可手动把任意 .ttf 复制为 " + str(dst))


def main() -> None:
    runs_sel = argv_flag("runs")
    epochs = argv_flag("epochs")
    data = argv_flag("data", str(DATASET_DIR / DATASET_YAML_NAME))
    device = argv_flag("device", "0")
    resume = argv_flag("resume")

    keys = list(RECIPES) if not runs_sel else [k.strip() for k in runs_sel.split(",")]
    bad = [k for k in keys if k not in RECIPES]
    if bad:
        log(f"!! 未知实验名 {bad}，可选：{list(RECIPES)}")
        return

    ensure_dirs(TRAIN_DIR, EVAL_DIR, WEIGHTS_DIR)

    # 先校验数据集可加载（fail fast，别等加载完模型才发现路径错）
    if not Path(data).exists():
        log(f"!! 数据集配置不存在: {data}")
        log("   请先跑 step4_split_dataset.py 生成")
        return
    log(f"数据集配置: {data}")

    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    # 必须在 check_det_dataset 之前：后者末尾会调 check_font()，字体缺失时
    # 会联网下载，本机代理 502 会让它抛异常，整个训练直接开不起来。
    ensure_arial_font()

    try:
        d = check_det_dataset(str(data))
        log(f"数据集校验通过：nc={d.get('nc')} names={d.get('names')}")
        log(f"  train={d.get('train')}")
        log(f"  val  ={d.get('val')}")
    except Exception as e:
        log(f"!! 数据集校验失败: {e}")
        return

    log("训练前泄漏复检:")
    if not preflight_leak_check(Path(data)):
        log("!! train/val 存在同名交集，数据划分已污染。")
        log("   继续训练会得到虚高指标，报告不可用。已中止。")
        log("   请重跑 step4_split_dataset.py 后重试。")
        return
    log("泄漏复检通过，开始训练。")

    records = []
    for key in keys:
        cfg = RECIPES[key]
        # 解析成本地路径，避免联网下载失败（本机 github 隧道 502）
        resolved = resolve_weight(cfg["weight"])
        if resolved == cfg["weight"]:
            log(f"!! 未在 {WEIGHTS_DIR} 找到 {cfg['weight']}，将尝试联网下载")
            log(f"   -> 若失败，请手动下载后放入 {WEIGHTS_DIR}")

        # ---- 断点续训（--resume）----
        # 修复：此前 `resume = argv_flag("resume")` 只读不用，是个死变量 ——
        # 传了 --resume 也依然从头开一轮新训练，且脚本不会报任何错，
        # 属于「静默失效」。这里补上真正的续训逻辑。
        # 注意：ultralytics 在 resume=True 时会拿 checkpoint 里的 train_args
        # 覆盖本次传入的 project / name / epochs，所以只在该 run 的 last.pt
        # 确实存在时才置位；否则明确提示「将从零训练」，不假装续上了。
        resuming = False
        if resume:
            last_pt = TRAIN_DIR / key / "weights" / "last.pt"
            if last_pt.exists():
                resolved = str(last_pt)
                resuming = True
            else:
                log(f"!! {key} 未找到 {last_pt}，--resume 不生效，将从零训练")

        args = dict(COMMON_ARGS)
        # [2026-09-23 新增] 按实验覆盖参数。
        # 字段名叫 `aug` 是历史遗留（最早只用于增强实验），现在它是
        # 「该实验相对 COMMON_ARGS 的差异项」的通用载体 —— 例如
        # v11s640_clsbal 用它覆盖 loss 参数 cls_pw。
        # 未写该字段的实验 args 与改动前**逐键相同**，行为不变。
        if cfg.get("aug"):
            args.update(cfg["aug"])
        args.update(
            data=str(data),
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            device=device,
            project=str(TRAIN_DIR),
            name=key,
        )
        if resuming:
            args["resume"] = True
        elif epochs:
            args["epochs"] = int(epochs)

        log("=" * 70)
        log(f"[{key}] {cfg['note']}")
        log(f"  weight={resolved} imgsz={cfg['imgsz']} batch={cfg['batch']} "
            f"epochs={args['epochs']} device={device} resume={resuming}")
        log("=" * 70)

        rec = {"run": key, "weight": resolved, "imgsz": cfg["imgsz"],
               "batch": cfg["batch"], "epochs": args["epochs"], "note": cfg["note"],
               "status": "pending"}
        t0 = time.time()
        try:
            model = YOLO(resolved)
            results = model.train(**args)
            rec["status"] = "ok"
            rec["seconds"] = round(time.time() - t0, 1)

            # 收集关键指标
            rd = getattr(results, "results_dict", None) or {}
            rec["metrics"] = {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                              for k, v in rd.items()}
            save_dir = str(getattr(results, "save_dir", TRAIN_DIR / key))
            rec["save_dir"] = save_dir

            best = Path(save_dir) / "weights" / "best.pt"
            if best.exists():
                mb = best.stat().st_size / 1024 / 1024
                rec["best_pt"] = str(best)
                rec["best_pt_mb"] = round(mb, 2)
                log(f"  最佳权重: {best} ({mb:.2f} MB)")

                # 把 best.pt 也归到 03_权重/ 便于统一取用
                import shutil
                dst = WEIGHTS_DIR / f"{key}_best.pt"
                shutil.copy2(best, dst)
                log(f"  已归档: {dst}")

            log(f"  指标: {rec.get('metrics')}")
        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = repr(e)
            rec["seconds"] = round(time.time() - t0, 1)
            log(f"  !! 训练失败: {e}")
        records.append(rec)

    dump_json(
        {
            "data_yaml": str(data),
            "recipes": RECIPES,
            "results": records,
            "classes": CLASSES,
        },
        EVAL_DIR / "train_summary.json",
    )

    log("=" * 70)
    for r in records:
        m = r.get("metrics") or {}
        log(f"{r['run']:10s} {r['status']:8s} "
            f"mAP50={m.get('metrics/mAP50(B)')} mAP50-95={m.get('metrics/mAP50-95(B)')} "
            f"size={r.get('best_pt_mb')}MB")
    log("=" * 70)
    log("下一步：python evaluate.py  （在独立测试集上评估并产出每类 AP）")


if __name__ == "__main__":
    main()
