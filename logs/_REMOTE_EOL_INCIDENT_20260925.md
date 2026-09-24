# 远端行尾污染事件与修复（2026-09-25）

## 事件

推送提交 `ce5b27e`（P0 交付包修复）时，`gh_push.py` 把 **4 个文件的远端行尾
从 LF 变成了 CRLF**。

| 文件 | 修复前(远端) | 污染后(远端) | 修复后(远端) |
|---|---|---|---|
| `02_code/common.py` | CR=0 LF=405 | **CR=499** | CR=0 LF=499 |
| `02_code/evaluate.py` | CR=0 LF=334 | **CR=336** | CR=0 LF=336 |
| `02_code/train.py` | CR=0 LF=519 | CR=0 LF=521 | CR=0 LF=521 |
| `07_report/README.md` | CR=0 LF=306 | **CR=311** | CR=0 LF=311 |
| `审查报告_赛题符合性复核_20260924.md` | CR=0 LF=566 | CR=0 LF=653 | CR=0 LF=653 |

> train.py 与 审查报告 工作区本就是纯 LF ⇒ 未受影响。

## 成因

`gh_push.py` 用 `open(path,'rb').read()` 读**工作区**文件。本机
`core.autocrlf=true`（无 `.gitattributes`）⇒ 工作区是 CRLF ⇒ 原样建成 blob 上传。

**git 对象库自身仍是纯 LF**（`git cat-file blob HEAD:<p>` 实测 CR=0），
所以本地查不出问题——只有查 GitHub 的 `/git/blobs/<sha>` 才暴露。

## 修复

脚本 `logs/_fix_remote_eol_push.py`：从 **git 对象库**（`cat-file blob HEAD:<path>`）
取纯 LF 字节 → 重建 blobs → 带 `base_tree` 建 tree → 建 commit（parent=被污染的 HEAD）
→ `PATCH refs/heads/main`。

- 被污染的 commit: `8c8cbd0e`
- 修复 commit      : `8c233fa1`
- 证明：逐文件「**归一为 LF 后 sha256**」与本地对象库一致 ⇒ **只改了行尾，内容未变**。

## 验证（唯一权威 = api.github.com）

`logs/_remote_eol_cmp.py` 取远程新旧两版 blob 的字节统计：
修复后 5 个文件全部 **CR=0**。

## 两条防坑纪律

1. **推送后必须抽查「本次推送过的那批文件」的远端 blob 行尾**。
   ⚠️ 别顺序扫全库 blob（几百请求会 SIGTERM）——只查本次推送的那批。
2. **别用 Git Bash 管道统计行尾**：`git show`/`cat-file` 经 Git Bash 会被
   smudge 回 CRLF，`grep -c $'\r'` 会**假报 CR>0**（本次就因此先误判「仓库被污染」）。
   必须用 Python `subprocess` 取 stdout **bytes** 统计，或以 `/git/blobs` API 为准。

## 技能已同步修正

`~/.workbuddy/skills/github-push-via-rest-api/`：
- `scripts/gh_push.py`：上传前改为 `git cat-file blob HEAD:<path>` 取字节（不再读工作区）。
- `SKILL.md`：新增「第五个坑：行尾会被工作区污染」，含成因、修法、验证纪律。
