# -*- coding: utf-8 -*-
"""
多种子训练接力器 —— 依次跑完 [seed1, seed2, seed3] 的 v11s640_clsbal。

为什么需要「接力器」而不是一次 `train.py --runs=...`：
  本环境后台任务**约 28 分钟被外部终止**（exit -1、无 traceback）。
  100 轮 × 110 s/轮 ≈ 3.06 h/种子，必然跨多个 28 分钟窗口。
  ⇒ 唯一可行办法：跑一块 → 被终止 → 用 `--resume` 从 last.pt 续 → …

设计（承接 _clsbal_train_loop.py 的 v3 经验）：
  1. 判据用「进程是否还在」（不要用 last.pt 是否变化 —— 单轮内它会静止 100+ 秒）。
  2. 每个种子的训练输出必须显式重定向到日志文件，否则续训日志丢失、崩了无从查因。
  3. `--resume` 后 ultralytics 会从 last.pt 读回 train_args，
     **因此 seed 必须由 last.pt 保证**（首次启动时写入），接力时不必重复传。
  4. 每块结束检查 results.csv 轮数是否推进；未推进且日志含 OOM 特征 ⇒ 停止并报错。

用法（后台）：
  D:/下载/python.exe logs/_multiseed_loop.py
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r"D:\下载\python.exe"
TRAIN = ROOT / "02_code" / "train.py"
TRAIN_DIR = ROOT / "04_results" / "train"

BASE_RUN = "v11s640_clsbal"
SEEDS = [42, 123, 2024]        # 42 已存在（复用），123/2024 为新增
# ★ 种子 42 的正史产物目录就是 `v11s640_clsbal`（不带后缀）——
#   它在 2026-09-23 已跑满 100 轮，是报告里「类平衡损失单种子结论」的来源。
#   **不要**把它复制成 `_s42` 副本：那样会产生两份 19 MB 同名权重，
#   且报告到底引用哪一份会变得含糊。⇒ 这里做「别名」而不是「复制」。
EXISTING_RUNS = {42: "v11s640_clsbal"}   # seed -> 既有 run 目录名（缺省则用 _s<seed>）
TARGET = 100
MAX_CHUNKS_PER_SEED = 14       # 每种子最多接力块数（14×~15轮 > 100 轮，留足余量）
POLL = 30

LOG = ROOT / "logs" / "_multiseed_loop.log"


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_name(seed):
    """该种子对应的**实际输出目录名**（可能复用既有 run）。"""
    return EXISTING_RUNS.get(seed, "%s_s%d" % (BASE_RUN, seed))


def results_csv(seed):
    return TRAIN_DIR / run_name(seed) / "results.csv"


def train_log(seed):
    return ROOT / "logs" / ("_train_%s.log" % run_name(seed))


def done_epochs(seed):
    rc = results_csv(seed)
    if not rc.exists():
        return 0
    try:
        with open(rc, "r", encoding="utf-8", newline="") as f:
            return max(0, sum(1 for _ in csv.reader(f)) - 1)
    except Exception as e:
        log("  !! 读 results.csv 失败：%r" % e)
        return 0


def detect_oom(seed, tail_bytes=200000):
    p = train_log(seed)
    if not p.exists():
        return False
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            t = f.read().decode("utf-8", "replace")
        return ("OutOfMemoryError" in t) or ("Insufficient memory" in t) or ("bad allocation" in t)
    except Exception:
        return False


def is_training_alive(run):
    """当前是否有指定 run 的训练主进程在跑（排除 dataloader 子进程）。"""
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*train.py*' "
        "-and $_.CommandLine -like '*%s*' "
        "-and $_.CommandLine -notlike '*multiprocessing*' } | "
        "Measure-Object | Select-Object -ExpandProperty Count" % run
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=60,
        )
        val = (out.stdout or "").strip()
        return int(val) > 0 if val.isdigit() else False
    except Exception as e:
        log("  !! 进程探测失败：%r —— 保守返回 True（不启动新训练）" % e)
        return True


def free_gb():
    import ctypes

    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    m = M()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1024 ** 3


def train_one_seed(seed):
    """把一个种子跑到 TARGET 轮（跨多个 28 分钟窗口接力）。"""
    run = run_name(seed)
    log("=" * 72)
    log("开始种子 %d （run=%s，目标 %d 轮）" % (seed, run, TARGET))
    log("=" * 72)

    for chunk in range(1, MAX_CHUNKS_PER_SEED + 1):
        n = done_epochs(seed)
        if n >= TARGET:
            log("  种子 %d 已完成 %d/%d 轮，收工。" % (seed, n, TARGET))
            return 0

        # 等已有训练（同一个 run）退出
        waited = 0
        while is_training_alive(run):
            if waited == 0:
                log("  检测到 %s 的训练进程在跑（已完成 %d 轮），等待…" % (run, n))
            time.sleep(POLL)
            waited += POLL

        n2 = done_epochs(seed)
        if n2 >= TARGET:
            log("  等待期间已跑满 %d/%d 轮，收工。" % (n2, TARGET))
            return 0

        avail = free_gb()
        log("  内存可用 %.2f GB" % avail)
        if avail < 2.0:
            log("  !! 可用内存仅 %.2f GB，训练很可能 OOM。等待 120s 后重试…" % avail)
            time.sleep(120)
            continue

        cmd = [PY, str(TRAIN), "--runs=" + BASE_RUN]
        if n2 == 0:
            # 首次启动：显式传 seed 与 tag
            cmd += ["--seed=%d" % seed, "--tag=_s%d" % seed]
        else:
            cmd.append("--resume")

        log("  --- 第 %d 块：已完成 %d/%d 轮，启动" % (chunk, n2, TARGET))
        with open(train_log(seed), "a", encoding="utf-8", errors="replace") as lf:
            lf.write("\n\n" + "=" * 72 + "\n")
            lf.write("[接力器] 种子 %d 第 %d 块启动于 %s\n"
                     % (seed, chunk, time.strftime("%Y-%m-%d %H:%M:%S")))
            lf.write("命令: %s\n" % " ".join(cmd))
            lf.write("=" * 72 + "\n")
            lf.flush()
            rc = subprocess.call(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)

        after = done_epochs(seed)
        log("  --- 第 %d 块结束 rc=%s，轮数 %d -> %d" % (chunk, rc, n2, after))

        if after <= n2 and not is_training_alive(run):
            if detect_oom(seed):
                log("  !! 种子 %d 因 OOM 失败。停止整个接力器。" % seed)
                return 4
            log("  !! 种子 %d 本块未推进且进程已退出（rc=%s）。"
                "查 %s 末段。停止。" % (seed, rc, train_log(seed).name))
            return 2

    log("  !! 种子 %d 达最大块数仍未跑满（当前 %d）。" % (seed, done_epochs(seed)))
    return 3


def main():
    log("#" * 72)
    log("多种子接力器启动：seeds=%s，每种子目标 %d 轮" % (SEEDS, TARGET))
    log("#" * 72)

    rc_all = 0
    for seed in SEEDS:
        rc = train_one_seed(seed)
        log("种子 %d 返回 rc=%s" % (seed, rc))
        if rc != 0:
            rc_all = rc
            log("!! 种子 %d 未正常完成，停止后续种子。" % seed)
            return rc_all

    log("#" * 72)
    log("全部种子完成：")
    for seed in SEEDS:
        log("  种子 %-5d -> %d 轮" % (seed, done_epochs(seed)))
    log("#" * 72)

    # ---- 训练收尾不在这里 hand-off，直接在同一进程内串起评估 + 方差统计 ----
    # 原因：本环境里「后台作业在工具调用返回时被回收」，单独挂一个收尾进程
    # 会在下一次工具调用时消失。既然接力器本身就跑在同一条命里，收尾也走这里最稳。
    post = ROOT / "logs" / "_post_multiseed.py"
    if post.is_file():
        log("调用收尾流水线（逐种子评估 + 方差统计）")
        r = subprocess.run([PY, str(post)], capture_output=True)
        for ln in r.stdout.decode("utf-8", "replace").splitlines()[-30:]:
            log("  | " + ln)
        log("收尾流水线 rc=%s" % r.returncode)
    else:
        log("（未找到 logs/_post_multiseed.py，跳过收尾）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
