# 收尾操作卡 —— Tag 推送 + 仓库改名

> 生成时间：2026-09-24
> 前置状态：**代码已全部推送成功**（远程 `main` = `debe49a`，217 个文件，2.55 MB，零泄漏）
> 剩余两件事都必须在 **GitHub Desktop** 里做，原因见文末「为什么 CLI 做不了」。

---

## 一、当前状态（已核实）

| 项目 | 状态 |
|---|---|
| 远程仓库 | `https://github.com/dutanlu/-` ✅ 可访问 |
| 远程 `main` | `debe49a01ef634b2d4b78791be4445f8b47486c7` ✅ 与本地一致 |
| 文件数 / 体积 | 217 个 / 2.55 MB ✅ |
| 开源证据文档 | `07_report/开源发布记录.md` ✅ 在线 |
| 本地 tag | `v1.0-submission` → `debe49a`（**已重建为附注标签，指向最终快照**）|
| **远程 tag** | ❌ **不存在**（待推送）|
| 仓库名 | `-`（一个横杠，不规范，待改名）|

---

## 二、任务 A：推送 tag `v1.0-submission`

**位置**：GitHub Desktop 窗口顶部菜单栏 → **Repository** → **Push**（如果灰着，先点 **Fetch origin**）

> 注意：GitHub Desktop 默认会把**当前分支上的标签**随普通 push 一起推上去。
> 因为本地 `main` 已经和远程一致，单纯点 Push 可能是灰的。
> 此时用下面任一条路：

**路径 1（推荐，最稳）**：菜单栏 **Repository** → **Push**（若可用直接点）
**路径 2**：菜单栏 **Branch** → 找 **Push tag** 相关项；或在 **History** 列表里找到提交 `debe49a`（提交信息「报告与答辩材料补开源章节（含可核验证据）」）→ 右键 → **Create Tag**，名字填 `v1.0-submission`，然后 Push。

**验证方法**（推送后，在浏览器直接打开这个链接）：
```
https://github.com/dutanlu/-/tags
```
看到 `v1.0-submission` 出现即成功。

或者打开：
```
https://api.github.com/repos/dutanlu/-/tags
```
返回内容里出现 `"name": "v1.0-submission"` 即成功（**这是最权威的验证方式**）。

---

## 三、任务 B：仓库改名 `-` → `wall-defect-screening`

**位置**：GitHub Desktop 菜单栏 → **Repository** → **Repository settings...**（快捷键 `Ctrl+Shift+,`）

1. 在弹出的设置窗口点 **Rename this repository**（或直接在 **Name** 输入框改）
2. 把名字从 `-` 改成 **`wall-defect-screening`**
3. 点 **Rename repository**

**为什么建议改**：
- `-`（单个横杠）在拷贝粘贴时极易被当成排版符号丢掉或看漏，别人拿到链接会打不开
- GitHub 会**自动保留旧地址的重定向**，改了名旧链接依然能打开，不存在「改了就失效」的风险
- 一个名字规范的开源仓库，在评审眼里是「这个人认真在做开源」的直接信号

**如果改名了，需要我做的事**：告诉我一声，我会把项目内所有引用仓库 URL 的地方（README、技术报告 §10.5、答辩材料、演示视频脚本文案）批量更新到新地址，并重新推送。**这一步请务必回来找我，否则文档里的链接和实际仓库会不一致。**

⚠️ **顺序建议**：**先做任务 B（改名），再做任务 A（推 tag）**。
原因是改名会自动更新本地 remote 地址；如果先推 tag 再改名，虽然也没问题，
但先改名可以少一次同步操作。

---

## 四、为什么 CLI 做不了（已彻底定位，非猜测）

我做了完整的连通性实测：

| 目标 | 结果 |
|---|---|
| `https://api.github.com/` | ✅ **200**，耗时 0.48s |
| `https://github.com/`（直连） | ❌ 000，超时 20s |
| `https://github.com/`（经代理 `127.0.0.1:57656`） | ❌ 000，超时 10s |
| `https://github.com/dutanlu/-.git/info/refs`（git 端点） | ❌ 000，超时 10s |

**结论**：本机当前的网络代理**只放行 `api.github.com`，不放行 `github.com` 主站**。
`git push` 必须访问 `github.com` 的 git 端点，因此**在当前代理策略下结构性不可能成功** ——
与 tag、与凭据、与操作方式都无关（凭据也查过了：本机凭据管理器里没有 GitHub 条目，
GitHub Desktop 的 OAuth 令牌不暴露给命令行）。

之前那些 `git push` 的 502 / `CONNECT tunnel failed` / `could not read Username`，
都是同一个网络限制的不同表现，不是配置错误。

> 唯一的绕法是用 `gh` CLI 的 API 通道，但本机未安装 `gh`，且安装后仍需单独登录授权，
> 不比你点两下 Desktop 更省事。所以走 Desktop。

---

## 五、做完之后

两件事都完成后，告诉我，我来：
1. 用 API 复核远程 tag 确实存在
2. 如果已改名，批量更新全部文档里的仓库 URL 并重新推送
3. 把最终状态记入项目备忘

至此，**代码推送 + tag 标记 + 仓库规范化**这条线就彻底闭环了。
