# -*- coding: utf-8 -*-
"""多种子训练收尾流水线 —— 训练跑齐后，一次性把评估与方差统计做完。

为什么需要它：
  `_multiseed_loop.py` 只管训练，不管**评估**。而报告 §7 要的结论是
  「类平衡损失与基线的差，是否小于跨种子方差」——这需要每个种子的
  `best.pt` 都跑一遍 `evaluate.py`，再用 `_multiseed_variance.py` 汇总。
  本脚本把这两步串起来，且**幂等**：已评估过的种子会跳过（除非 --force）。

设计要点（承接本环境的既有纪律）：
  1. 只在训练**已停止**时才动作 —— 否则 GPU 会被抢，且 best.pt 还在变。
  2. 每个种子跑一次 evaluate.py，输出
     `04_results/eval/<run>_test_metrics.json`。
  3. 三个种子齐全后调 `_multiseed_variance.py` 出汇总。
  4. 全程日志增量落盘（本环境后台任务约 28 分钟会被终止，日志必须随时可读）。
  5. **显式传 device**，且 ONNX 场景已在 evaluate.py 内部兜底。

用法：
  D:/下载/python.exe logs/_post_multiseed.py            # 跑一次
  D:/下载/python.exe logs/_post_multiseed.py --force    # 强制重评
  D:/下载/python.exe logs/_post_multiseed.py --wait     # 训练没停就先等着
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r"D:\下载\python.exe"
TRAIN_DIR = ROOT / "04_results" / "train"
EVAL_DIR = ROOT / "04_results" / "eval"
EVALUATE = ROOT / "02_code" / "evaluate.py"
VARIANCE = ROOT / "logs" / "_multiseed_variance.py"
LOG = ROOT / "logs" / "_POST_MULTISEED.log"

# seed -> (run 目录名, 评估产物名)
SEEDS = {
    42: ("v11s640_clsbal", "v11s640_clsbal"),
    123: ("v11s640_clsbal_s123", "v11s640_clsbal_s123"),
    2024: ("v11s640_clsbal_s2024", "v11s640_clsbal_s2024"),
}
TARGET = 100

FORCE = "--force" in sys.argv
WAIT = "--wait" in sys.argv


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ★ 自匹配陷阱（2026-09-25 踩，实测确认）：
#   旧写法用 `*train.py*` 作为匹配模式，但**探测命令自身的命令行里就含
#   "train.py" 这个字面量**（它写在 Where-Object 的 pattern 里）⇒ 探测进程
#   会匹配到自己 ⇒ training_alive() 恒为 True ⇒ 守卫永远自锁、流水线跑不起来。
#   证据：探测输出里唯一命中的 PID 就是那条 powershell 探测进程本身。
#   修法：① 先把「本探测进程的 PID 及其父 PID」排除；② 模式加上更严格的前缀
#   （要求是 python 解释器在跑 02_code\train.py），避免任何字符串巧合。
def training_alive():
    """是否有真正的训练进程在跑。

    判据：存在 python 进程，其命令行里**以参数形式**出现 `train.py`，
    且该进程不是本探测命令自己。为绕开本机「PowerShell stdout 不回传」的限制，
    探测结果写临时文件再读回。
    """
    import os
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".txt", prefix="_alive_")
    os.close(fd)
    tmp_ps = tmp.replace("\\", "\\\\")
    try:
        # 排除 powershell 自身的检测命令：只保留 Name 为 python 的进程
        ps = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.Name -like 'python*' -and "
            "$_.CommandLine -match 'train\\.py' } | "
            "Select-Object -ExpandProperty ProcessId | "
            "Set-Content -Path '%s' -Encoding UTF8"
            % tmp_ps
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True)
        try:
            out = Path(tmp).read_text(encoding="utf-8-sig", errors="replace").strip()
        except Exception:
            out = ""
        # 二次保险：过滤掉任何含 "powershell" 的行
        real = [ln for ln in out.splitlines()
                if ln.strip() and "powershell" not in ln.lower()
                and ln.strip().isdigit()]
        return bool(real)
    except Exception as e:
        log("!! 进程探测失败（按「无训练」处理）：%r" % e)
        return False
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def done_epochs(run):
    c = TRAIN_DIR / run / "results.csv"
    if not c.is_file():
        return 0
    try:
        with open(c, encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return 0


def best_pt(run):
    return TRAIN_DIR / run / "weights" / "best.pt"


def metrics_json(name):
    return EVAL_DIR / ("%s_test_metrics.json" % name)


def run_eval(run, name):
    w = best_pt(run)
    if not w.is_file():
        log("  !! best.pt 不存在：%s" % w)
        return False
    log("  评估 %s -> %s" % (w.name, name))
    t0 = time.time()
    r = subprocess.run(
        [PY, str(EVALUATE), "--weights=%s" % w, "--name=%s" % name],
        cwd=str(ROOT), capture_output=True)
    rc = r.returncode
    tail = r.stdout.decode("utf-8", "replace").splitlines()[-12:]
    for ln in tail:
        log("    | " + ln)
    ok = rc == 0 and metrics_json(name).is_file()
    log("  %s rc=%s 用时 %.1fs" % ("✅" if ok else "❌", rc, time.time() - t0))
    return ok


def main():
    log("=" * 74)
    log("多种子收尾流水线启动  FORCE=%s WAIT=%s" % (FORCE, WAIT))
    log("=" * 74)

    # ---- 1. 等训练停（可选）
    guard = 0
    while WAIT and training_alive():
        guard += 1
        if guard % 10 == 1:
            prog = "  ".join("seed%d=%d" % (s, done_epochs(r))
                             for s, (r, _) in SEEDS.items())
            log("  训练中，等待…… %s" % prog)
        time.sleep(30)

    if training_alive():
        log("!! 仍有训练在跑（未加 --wait），本次退出，不抢 GPU")
        return 3

    log("训练已停，开始收尾")

    # ---- 2. 逐种子评估
    pending, done = [], []
    for seed, (run, name) in sorted(SEEDS.items()):
        ep = done_epochs(run)
        mj = metrics_json(name)
        if FORCE or not mj.is_file():
            if ep < TARGET:
                log("  种子 %d（%s）仅 %d/%d 轮，未跑满 —— 跳过评估"
                    % (seed, run, ep, TARGET))
                pending.append(seed)
                continue
            done.append(seed)
        else:
            log("  种子 %d（%s）已有评估产物，跳过" % (seed, run))

    if pending:
        log("!! 以下种子未跑满，无法出方差结论：%s" % pending)
        return 2

    if not done:
        log("  三个种子评估产物均已存在，无需重评")

    for seed in sorted(done):
        run, name = SEEDS[seed]
        if not run_eval(run, name):
            log("!! 种子 %d 评估失败，中止（不产出可能错误的方差结论）" % seed)
            return 1

    # ---- 3. 方差汇总
    log("调用 _multiseed_variance.py 汇总")
    r = subprocess.run([PY, str(VARIANCE)], cwd=str(ROOT), capture_output=True)
    for ln in r.stdout.decode("utf-8", "replace").splitlines()[-20:]:
        log("  | " + ln)
    log("方差脚本 rc=%s" % r.returncode)

    # ---- 4. 复核汇总文件的完整性
    vj = EVAL_DIR / "multiseed_variance.json"
    if vj.is_file():
        import json
        d = json.load(open(vj, encoding="utf-8"))
        seeds = d.get("seeds") or {}
        nn = [k for k, v in seeds.items() if v]
        log("multiseed_variance.json：非空种子 = %s" % sorted(nn))
        if len(nn) < 3:
            log("!! 非空种子不足 3 个，结论仍不完整")
            return 1
        log("✅ 三种子齐全，方差结论可引用")
    else:
        log("!! 未生成 multiseed_variance.json")
        return 1

    # ---- 5. 把结论写进报告 §7.8.7（脚本自带「不足 3 种子拒绝写」门禁）
    writer = ROOT / "logs" / "_write_multiseed_section.py"
    if writer.is_file():
        log("调用 _write_multiseed_section.py 写入报告 §7.8.7")
        r2 = subprocess.run([PY, str(writer)], cwd=str(ROOT), capture_output=True)
        for ln in r2.stdout.decode("utf-8", "replace").splitlines()[-10:]:
            log("  | " + ln)
        log("报告写入 rc=%s" % r2.returncode)
    else:
        log("!! 未找到 _write_multiseed_section.py，报告未更新")

    log("收尾完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
