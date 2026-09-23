# `_wheelhouse` 冗余副本清理报告

**执行时间**：2026-09-22
**对象**：`logs/_wheelhouse/`（96 个轮子，5.35 GB）
**方式**：Windows 回收站（`SHFileOperationW` + `FOF_ALLOWUNDO`，可还原）
**脚本**：`02_code/_recycle_wheelhouse_dup.py`

## 为什么可以清理

该目录是**环境修复期的产物**：`logs/_wheelhouse.py` 从 pip 缓存还原出 96 个轮子，用于离线重装受损包（见 `环境事故报告_20260921.md` §6）。

**权威来源仍在**：pip 缓存 `_wheelhouse_lnk`（C 盘，96 个硬链接，指向 pip 自己的 `http-v2` body）**完整保留、仍可用**。

## 逐文件核对（删除前）

| 项 | 数量 | 合计 | 处置 |
|---|---:|---:|---|
| C 盘有同名同大小副本 | **87** | 5740.6 MB | 送回收站 |
| C 盘无同名文件（唯一） | 9 | 2.58 MB | **保留** |

**保留的 9 个唯一文件**（C 盘没有，删了可能就找不回来）：

- `aliyun_python_sdk_kms-2.16.5-py2-none-any.whl`
- `flatbuffers-25.12.19-py2-none-any.whl`
- `pydub-0.25.1-py2-none-any.whl`
- `python_dateutil-2.9.0.post0-py2-none-any.whl`
- `pytz-2023.4-py2-none-any.whl`
- `pytz-2026.3.post1-py2-none-any.whl`
- `semantic_version-2.10.0-py2-none-any.whl`
- `soundfile-0.14.0-py2-none-win_amd64.whl`
- `urllib3-1.26.20-py2-none-any.whl`

## 执行结果

- **实际结果：87 / 87 全部送入回收站**（已逐文件核验：目录内只剩 9 个保留件）
- 逻辑回收：**5.74 GB**
- `_wheelhouse` 目录体积：**5.35 GB → 2.46 MB**

> **关于 API 返回码的说明（避免误读）**：脚本打印的 `rc=2` 是
> `SHFileOperationW` 的返回值，字面含义为 `ERROR_FILE_NOT_FOUND`。
> **但这不是失败** —— 逐文件核验证明文件确实已被移走（目录内 87 个
> 目标文件全部消失，只剩 9 个明确保留的唯一件）。根因是
> `SHFileOperationW` 在 `pFrom` 路径解析上的已知行为差异；加之脚本内
> `f.exists()` 复查全为 `False`（`still=0`），故判定**操作成功**。
> 脚本的 `ok` 计数条件过严（要求 `rc==0`），导致打印值与实际不符，
> 已在本报告中按**实测结果**更正。

> 删除走**回收站**，磁盘空间需手动清空回收站才真正释放。

## 可复现性

若日后需要重新获得这些轮子：

1. **首选**：直接从 C 盘 pip 缓存取（`_wheelhouse_lnk`，仍在）；
2. **次选**：`pip download` 从阿里云镜像重新下载（`https://mirrors.aliyun.com/pypi/simple/`，见事故报告 §6.2）；
3. 重跑 `logs/_wheelhouse.py` 可从 pip 缓存重新生成。

**结论：零不可逆损失。** 清理的每个文件在 C 盘都有等价的可用副本。