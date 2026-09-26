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
  python train.py --runs=v11s896 --imgsz=896 --batch=16   # 显式钉住 imgsz/batch
  python train.py --runs=v11s640 --tag=_b16 --imgsz=640 --batch=16 --workers=0

★ 支持的参数：--runs= --tag= --epochs= --data= --seed= --resume --device=
  --imgsz= --batch= --workers=  （**没有 --help**；加未知参数会被静默忽略并直接开跑）
⚠️ 本机系统内存仅 ~16.9 GB：workers=4 可能在训练中途 MemoryError（且退出码仍 0）。
   长训练建议显式 --workers=0。

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
    # ------------------------------------------------------------------
    # [2026-09-25 新增] 针对「露筋」短板的两个候选（见 logs/_WEAK_CLASS_DIAG.txt）
    #
    # 依据：露筋 AP50=0.4568（七类唯一低于 0.70），召回仅 0.3946。
    #   错误形态归因：完全漏检 41.5% + 低置信度被切 13.6%，「框对但类错」为 0
    #   ⇒ 既不是类间混淆，也不是阈值问题，而是纯粹的召回不足。
    #   几何证据：露筋长宽比 P90=10.3（七类最高），平均归一化宽仅 0.1535
    #   ⇒ 640 输入下宽约 98px 的细长目标，IoU 对端点偏移极敏感。
    #
    # ⚠️ 已证伪的路径（勿重复）：**推理侧**单纯提高 imgsz 反而全线下降
    #   （896: mAP50 −0.0185 / 1024: −0.0492，七类无一例外），
    #   因为权重是在 640 学的，属训练-推理尺度失配。
    #   故必须**在训练侧**提分辨率。见 logs/_EXP_IMGSZ_INFER.md。
    #
    # 对照关系：两个配方只差 mosaic 一项，各自与 v11s640 比。
    # ------------------------------------------------------------------
    "v11s896": {
        "weight": "yolo11s.pt", "imgsz": 896, "batch": 16,
        "note": "高分辨率训练：imgsz 640→896（露筋召回专项，对照 v11s640）",
        # batch 从 24 降到 16：896² 的激活占用约为 640² 的 2 倍，
        # 12.2 GB 显存下 24 会 OOM（估）。16 是保守起点。
        # ★ workers=0 必须显式设：896 档 4 个 worker 会各自缓存解码后的
        #   896×896 图，本机可用内存仅 6 GB ⇒ 实测 20 步后
        #   `DataLoader worker exited unexpectedly`（2026-09-25 冒烟复现）。
        #   v11s640_clsbal 配方已因同样原因改用 workers=0。
        "aug": {"workers": 0},
    },
    "v11s896_nomosaic": {
        "weight": "yolo11s.pt", "imgsz": 896, "batch": 16,
        "note": "高分辨率 + 关闭 mosaic（mosaic 进一步压扁细长目标）",
        "aug": {"mosaic": 0.0, "close_mosaic": 0, "workers": 0},
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
    # 兼容工程包被拷贝到别的机器：若 yaml 里的 path 仍指向训练机旧路径，就地自愈
    from common import ensure_dataset_yaml, DATASET_YAML_NAME
    # ----------------------------------------------------------------------
    # [2026-09-25 新增] --data= 支持
    #
    # 动机：露筋标注碎片合并实验（logs/_watch_merge_ab.py）需要在**另一个数据集**
    #   （dataset_merged/）上训练，唯一变量是标签。但此前本函数把 data 写死为
    #   DATASET_DIR/wall_defects.yaml —— 文件头 docstring 第 21 行承诺支持
    #   `--data=`，**而实现里从未读取**（一次「只在注释里承诺」的坑，
    #   与纪律「docstring 点名的产物必须真存在」同类）。
    #
    # 设计：
    #   --data=<path>  显式指定数据集 yaml；不传时行为与改动前**逐键相同**。
    #   ★ 显式指定时**不调用 ensure_dataset_yaml**（就地自愈只适用于默认数据集；
    #     对实验用 yaml 若强行改写其 path，会破坏实验配置的可追溯性）。
    # ----------------------------------------------------------------------
    data_arg = argv_flag("data")
    if data_arg:
        data = str(Path(data_arg))
        if not Path(data).exists():
            log(f"!! --data 指定的数据集配置不存在: {data}")
            raise SystemExit(2)
        log(f"[data] 使用显式指定的数据集配置（跳过自愈）: {data}")
    else:
        data = str(ensure_dataset_yaml(str(DATASET_DIR / DATASET_YAML_NAME)))
    device = argv_flag("device", "0")
    resume = argv_flag("resume")
    # ----------------------------------------------------------------------
    # [2026-09-25 新增] --imgsz= / --batch= 覆盖
    #
    # 动机：imgsz 对照实验（logs/_watch_imgsz_ab.py）存在**混杂因子** ——
    #   配方表里 v11s640 是 batch=24、v11s896 是 batch=16，两者同时变，
    #   所以严格说只能表述为「imgsz640+batch24」vs「imgsz896+batch16」，
    #   不能归因到 imgsz。要补做 **batch 单变量**对照（640@16 vs 896@16），
    #   就必须能把 batch 显式钉住 —— 而在此之前 imgsz/batch 只能从 RECIPES 取，
    #   无法覆盖（又一次「docstring 承诺了却做不到」的同款问题）。
    #
    # 设计：
    #   不传时行为与改动前**逐键相同**（仍用 cfg["imgsz"] / cfg["batch"]）。
    # ----------------------------------------------------------------------
    imgsz_override = argv_flag("imgsz")
    batch_override = argv_flag("batch")
    # [2026-09-25 新增] --workers= 覆盖
    #   动机：COMMON_ARGS 里 workers=4（2026-09-21 提速改动）。但本机系统内存仅 ~16.9 GB，
    #   4 个 dataloader worker 各自缓存解码后的图会把内存吃穿 ——
    #   实测：v11s640_b16 跑到第 7 轮时
    #     `numpy._ArrayMemoryError: Unable to allocate 4.69 MiB ...`
    #     `Caught MemoryError in DataLoader worker process 1`
    #   而退出码仍是 0（静默失败）。v11s640_clsbal / v11s896 配方已各自改用 workers=0，
    #   但**默认配方 v11s640 没有** ⇒ 需要一个能显式钉住 workers 的开关。
    #   不传时行为与改动前**逐键相同**。
    workers_override = argv_flag("workers")
    # ----------------------------------------------------------------------
    # [2026-09-24 新增] 多随机种子支持（--seed / --tag）
    #
    # 动机：目前所有实验都是 seed=42 单种子。但本项目的核心结论之一
    #   「类平衡损失未带来可测量的精度提升」依赖的是**组间差值 + CI 半宽**
    #   的判据；若只跑一个种子，就无法回答「这个差值会不会只是种子抖动？」
    #   ⇒ 需要多种子复现，把「种子导致的方差」也量化出来。
    #
    # 设计：
    #   --seed=<int>  覆盖 COMMON_ARGS 的 seed（并保持 deterministic=True）
    #   --tag=<str>   给 run 名加后缀，避免多个种子写同一个输出目录
    #                 （ultralytics exist_ok=True 会**静默覆盖**，必须分开）
    #   例：python train.py --runs=v11s640_clsbal --seed=123 --tag=_s123
    #       ⇒ 输出到 04_results/train/v11s640_clsbal_s123/
    #
    # ⚠️ 不传 --tag 时行为与改动前**逐键相同**（run 名不加后缀）。
    # ----------------------------------------------------------------------
    seed_override = argv_flag("seed")
    tag = argv_flag("tag", "") or ""
    if seed_override is not None:
        try:
            seed_val = int(seed_override)
        except ValueError:
            log(f"!! --seed 必须是整数，收到 {seed_override!r}")
            return
        log(f"多随机种子模式：seed={seed_val}，run 名后缀={tag!r}")

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
        # [2026-09-24] 实际 run 名 = key + tag。tag 为空时与原行为完全一致。
        run_name = f"{key}{tag}"
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
            last_pt = TRAIN_DIR / run_name / "weights" / "last.pt"
            if last_pt.exists():
                resolved = str(last_pt)
                resuming = True
            else:
                log(f"!! {run_name} 未找到 {last_pt}，--resume 不生效，将从零训练")

        args = dict(COMMON_ARGS)
        # [2026-09-23 新增] 按实验覆盖参数。
        # 字段名叫 `aug` 是历史遗留（最早只用于增强实验），现在它是
        # 「该实验相对 COMMON_ARGS 的差异项」的通用载体 —— 例如
        # v11s640_clsbal 用它覆盖 loss 参数 cls_pw。
        # 未写该字段的实验 args 与改动前**逐键相同**，行为不变。
        if cfg.get("aug"):
            args.update(cfg["aug"])
        # [2026-09-24] 多种子覆盖。放在 aug 之后，保证「种子」是最后的显式决定。
        if seed_override is not None:
            args["seed"] = seed_val
        # 有效 imgsz / batch：命令行覆盖优先（不传时 == cfg 值，行为不变）
        eff_imgsz = int(imgsz_override) if imgsz_override else cfg["imgsz"]
        eff_batch = int(batch_override) if batch_override else cfg["batch"]
        args.update(
            data=str(data),
            imgsz=eff_imgsz,
            batch=eff_batch,
            device=device,
            project=str(TRAIN_DIR),
            name=run_name,
        )
        if resuming:
            args["resume"] = True
        elif epochs:
            args["epochs"] = int(epochs)
        if workers_override is not None:
            args["workers"] = int(workers_override)

        log("=" * 70)
        log(f"[{run_name}] {cfg['note']}")
        log(f"  weight={resolved} imgsz={eff_imgsz} batch={eff_batch} "
            f"workers={args.get('workers')} "
            f"epochs={args['epochs']} device={device} resume={resuming} "
            f"seed={args.get('seed')} cls_pw={args.get('cls_pw')}")
        log("=" * 70)

        rec = {"run": run_name, "weight": resolved, "imgsz": eff_imgsz,
               "batch": eff_batch, "workers": args.get("workers"),
               "epochs": args["epochs"], "note": cfg["note"],
               "seed": args.get("seed"), "status": "pending"}
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
            save_dir = str(getattr(results, "save_dir", TRAIN_DIR / run_name))
            rec["save_dir"] = save_dir

            best = Path(save_dir) / "weights" / "best.pt"
            if best.exists():
                mb = best.stat().st_size / 1024 / 1024
                rec["best_pt"] = str(best)
                rec["best_pt_mb"] = round(mb, 2)
                log(f"  最佳权重: {best} ({mb:.2f} MB)")

                # 把 best.pt 也归到 03_权重/ 便于统一取用
                # [2026-09-24] 用 run_name 而非 key：多种子时若用 key，
                # 三个种子会反复覆盖同一个 v11s640_clsbal_best.pt。
                import shutil
                dst = WEIGHTS_DIR / f"{run_name}_best.pt"
                shutil.copy2(best, dst)
                log(f"  已归档: {dst}")

            log(f"  指标: {rec.get('metrics')}")
        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = repr(e)
            rec["seconds"] = round(time.time() - t0, 1)
            log(f"  !! 训练失败: {e}")
        records.append(rec)

    # ----------------------------------------------------------------------
    # [2026-09-24 修复] train_summary.json 必须**按 run 名合并**，不能整体重写。
    #
    # 踩过的坑：此前这里直接 dump 本次的 records，导致**任何一次 train.py 调用
    #   都会把整个 summary 覆盖掉**。后果实例：2026-09-24 为验证多种子入口跑的
    #   1 轮 smoke（seed=999）把 100 轮的正式记录冲成了 mAP50=0.0001，
    #   而该文件是报告数字的来源之一 ⇒ **证据被静默污染**。
    #
    # 现在的语义：
    #   · 读回既有 results，按 run 名做「upsert」——同名覆盖、异名追加；
    #   · 顶层的 data_yaml / recipes / classes 一并刷新（它们本就是全局定义）；
    #   · 读回失败（文件损坏等）时**不静默丢弃**，改为保留旧内容并另存一份
    #     `.corrupt_<时间戳>`，再写新文件。
    # ----------------------------------------------------------------------
    summary_path = EVAL_DIR / "train_summary.json"
    merged, kept = [], 0
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            merged = list(old.get("results") or [])
            kept = len(merged)
        except Exception as e:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            broken = summary_path.with_name(f"train_summary.json.corrupt_{stamp}")
            try:
                shutil.copy2(summary_path, broken)
                log(f"!! 既有 train_summary.json 解析失败（{e!r}），已另存 {broken.name}")
            except Exception as e2:
                log(f"!! 既有 train_summary.json 解析失败（{e!r}），另存也失败：{e2!r}")
            merged, kept = [], 0

    new_names = {r["run"] for r in records}
    replaced = sum(1 for r in merged if r.get("run") in new_names)
    merged = [r for r in merged if r.get("run") not in new_names] + records

    dump_json(
        {
            "data_yaml": str(data),
            "recipes": RECIPES,
            "results": merged,
            "classes": CLASSES,
        },
        summary_path,
    )
    log(f"train_summary：新增/覆盖 {len(records)} 条"
        f"（覆盖 {replaced} 条，保留其它 {kept - replaced} 条），现共 {len(merged)} 条")

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
