# -*- coding: utf-8 -*-
"""
step0b_download_modelscope.py —— 从 ModelScope 国内镜像下载数据集

为什么有这条通道：
  Roboflow 登录接口在本机网络下不可用（app.roboflow.com 的 API 域请求失败），
  经实测 modelscope.cn / hf-mirror.com / gitee.com 均可直连（HTTP 200），
  且 ModelScope 的数据集**无需注册、无需 API key**，直接给 HTTP 下载端点。

本脚本当前下载：古建筑青砖表面损伤数据集（YOLO 格式）
  - 数据集 ID : destinylhj/greybrick-yolo
  - 许可      : Apache License 2.0
  - 体积      : 259.30 MB（4 个 zip）
  - 学术出处  : Gray Brick Wall Surface Damage Detection of Traditional
                Chinese Buildings in Macau（澳门岭南青砖建筑，375 张标注图 / 8 类）
  - 类别      : CRACK / W_E / ALKALI / MISS / MOSS（5 类，共 800 张图）

⚠️⚠️ 两条必须知道的坑（都实际踩过）⚠️⚠️

  1) 本机的 git（GitHubDesktop 自带那个）是精简版，**缺 remote-https helper**，
     跑 `git clone https://...` 会直接失败：
         git: 'remote-https' is not a git command.
         fatal: remote helper 'https' aborted session
     → 所以本脚本走 HTTP 端点，不用 git。

  2) **绝对不要用 PowerShell 的 Expand-Archive 解压本项目的 zip**。
     实测在「中文路径 + 大量文件 + 深层嵌套」下它**静默失败**：
     建好了目录骨架，但文件没写出来，而且**一个错误都不报**。
     当时 images.zip 的 800 张只出来 677 张，label/train/valid 更是全空。
     → 本脚本一律用 Python 标准库 zipfile，且**逐条校验文件大小**。

用法：
  python step0b_download_modelscope.py                # 下载 + 解压 + 校验
  python step0b_download_modelscope.py --list         # 只列文件清单，不下载
  python step0b_download_modelscope.py --skip-extract # 只下载

输出：
  01_data/raw/greybrick/
      src/{images,label,train,valid}/   ← 解压后的可用数据
      data.yaml  README.md              ← 元数据
      _zips/                            ← 原始压缩包（保留以备复核）
"""

from __future__ import annotations

import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from common import RAW_DIR, argv_flag, dump_json, ensure_dirs, log

# --------------------------------------------------------------------------
# 数据集配置
# --------------------------------------------------------------------------
DS_NAMESPACE = "destinylhj"
DS_NAME = "greybrick-yolo"
DS_REVISION = "master"
DS_ALIAS = "greybrick"

API_BASE = f"https://www.modelscope.cn/api/v1/datasets/{DS_NAMESPACE}/{DS_NAME}"
TREE_URL = f"{API_BASE}/repo/tree?Revision={DS_REVISION}&Recursive=true"
FILE_URL = f"{API_BASE}/repo?Revision={DS_REVISION}&FilePath={{fname}}"

# 期望的文件大小（bytes）—— 用于下载后校验，来自官方 tree 接口
EXPECTED = {
    "data.yaml": 106,
    "README.md": 326,
    "images.zip": 86_301_043,
    "label.zip": 86_496_172,
    "train.zip": 69_616_221,
    "valid.zip": 16_880_183,
}

# 期望的解压后文件数 —— 用于解压后校验
EXPECTED_EXTRACT = {
    "images": {"jpg": 800, "txt": 0},
    "label": {"jpg": 800, "txt": 801},
    "train": {"jpg": 640, "txt": 640},
    "valid": {"jpg": 160, "txt": 160},
}


def http_get_json(url: str) -> dict:
    import json
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def download(url: str, dst: Path, expect_size: int | None = None) -> bool:
    """下载单文件，带体积校验。返回是否成功。"""
    if dst.exists() and expect_size and dst.stat().st_size == expect_size:
        log(f"  已存在且体积正确，跳过: {dst.name}")
        return True
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        t0 = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=1800) as r, open(tmp, "wb") as f:
            total = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        dt = time.time() - t0
        if expect_size and total != expect_size:
            log(f"  !! 体积不符 {dst.name}: 实际 {total:,} 期望 {expect_size:,}")
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dst)
        log(f"  OK {dst.name}  {total:,} B  ({dt:.0f}s)")
        return True
    except Exception as e:
        log(f"  !! 下载失败 {dst.name}: {e}")
        tmp.unlink(missing_ok=True)
        return False


def unzip_verified(zp: Path, dest: Path) -> bool:
    """
    用 Python zipfile 解压，并逐条校验解压后文件大小。
    绝不使用 PowerShell Expand-Archive（本项目实测会静默失败）。
    """
    try:
        with zipfile.ZipFile(zp) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
            n_fail = 0
            for i in infos:
                z.extract(i, dest)
                out = dest / i.filename
                if not out.exists() or out.stat().st_size != i.file_size:
                    n_fail += 1
                    log(f"    [校验失败] {i.filename}")
            log(f"  {zp.name}: 条目 {len(infos)} → 失败 {n_fail}")
            return n_fail == 0
    except zipfile.BadZipFile as e:
        log(f"  !! {zp.name} 不是有效 zip: {e}")
        return False


def verify_extract(src: Path) -> bool:
    """核对解压结果的 jpg/txt 数量是否符合预期。"""
    all_ok = True
    for sub, exp in EXPECTED_EXTRACT.items():
        d = src / sub
        if not d.exists():
            log(f"  !! 缺少目录 {sub}/")
            all_ok = False
            continue
        n_jpg = len(list(d.rglob("*.jpg")))
        n_txt = len(list(d.rglob("*.txt")))
        ok = (n_jpg == exp["jpg"] and n_txt == exp["txt"])
        flag = "OK" if ok else "!!"
        log(f"  [{flag}] {sub:8s} jpg={n_jpg}/{exp['jpg']}  txt={n_txt}/{exp['txt']}")
        if not ok:
            all_ok = False
    return all_ok


def main() -> None:
    root = Path(argv_flag("root", str(RAW_DIR))) / DS_ALIAS
    zips_dir = root / "_zips"
    src_dir = root / "src"
    list_only = "--list" in sys.argv
    skip_extract = "--skip-extract" in sys.argv

    log(f"数据集: {DS_NAMESPACE}/{DS_NAME} @ {DS_REVISION}")
    log(f"目标目录: {root}")

    # ---------- 列清单 ----------
    log("拉取文件清单 ...")
    try:
        tree = http_get_json(TREE_URL)
    except Exception as e:
        log(f"!! 无法获取清单: {e}")
        return
    files = tree.get("Data", {}).get("Files", [])
    if not files:
        log("!! 清单为空，数据集可能已变更或需要授权")
        return
    log(f"清单 {len(files)} 项:")
    for f in files:
        log(f"    {f['Type']:5s} {f['Size']:>12,}  {f['Path']}  (LFS={f['IsLFS']})")
    if list_only:
        return

    ensure_dirs(root, zips_dir, src_dir)

    # ---------- 下载 ----------
    log("=" * 60)
    log("下载")
    log("=" * 60)
    results = {}
    for f in files:
        name = f["Path"]
        if "/" in name:      # 跳过嵌套路径，本数据集顶层平铺
            continue
        url = FILE_URL.format(fname=name)
        dst = (zips_dir / name) if name.endswith(".zip") else (root / name)
        ok = download(url, dst, EXPECTED.get(name))
        results[name] = ok

    n_ok = sum(1 for v in results.values() if v)
    log(f"下载完成: {n_ok}/{len(results)} 成功")

    # ---------- 解压 ----------
    if not skip_extract:
        log("=" * 60)
        log("解压（Python zipfile + 逐条校验）")
        log("=" * 60)
        for zp in sorted(zips_dir.glob("*.zip")):
            unzip_verified(zp, src_dir)

        log("=" * 60)
        log("校验解压结果")
        log("=" * 60)
        all_ok = verify_extract(src_dir)

    dump_json(root / "download_report.json", {
        "dataset": f"{DS_NAMESPACE}/{DS_NAME}",
        "revision": DS_REVISION,
        "root": str(root),
        "files": {k: ("ok" if v else "fail") for k, v in results.items()},
        "expected": EXPECTED,
        "expected_extract": EXPECTED_EXTRACT,
    })
    log(f"完成。数据目录: {src_dir}")
    log("提示: train/valid 已配对好，可直接用；images/ 和 label/ 是全集，"
        "若走 step2 去重会把三份冗余全部识别出来。")


if __name__ == "__main__":
    main()
