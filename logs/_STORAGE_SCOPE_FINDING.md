# 照片站存储可见性：核查结论（2026-09-20 18:38）

## 一、我此前给用户的两个选项，本身是错的

我原先说「把 shared 改成 private 桶」，**这个说法不成立**。核查官方 SDK 文档后：

| 我原先的说法 | 实际情况 |
|---|---|
| 「改成私有桶」 | **SDK 根本没有「切换桶可见性」这个能力** |
| 暗示 `user_private` 可以让「只有你（管理员）能看」 | `users/<uid>/...` 是**上传者本人** + 管理员可读写；它解决的是**网友之间互不可见**，不是「你看不到」 |

`references/storage/code-generation.md:197-202`「Unsupported in Generated Applications」明确列出**不支持**：
- Creating, updating, emptying, deleting or **switching logical Buckets**
- **Public Buckets / public URLs**

且 `code-generation.md:11`：SDK 只暴露**一个** logical Bucket = `runtime`。
⇒ **「换私有桶」在技术上做不到**，我必须撤回这个建议。

## 二、真正的三种 scope（文档 `management.md:44-48`）

| Scope | 物理前缀 | 谁能读 | 谁能改/删 |
|---|---|---|---|
| `user_private` | `users/<uid>/...` | 所有者 + 管理员 | 所有者 + 管理员 |
| `app_shared`（网友所有） | `shared/<uid>/...` | **所有已登录用户** + 管理员 | 所有者 + 管理员 |
| `app_shared`（应用所有） | `shared/_application/...` | **所有已登录用户** + 管理员 | 仅管理员 |

**关键点：`user_private` 与 `app_shared` 的唯一区别是「其他登录用户能否读」。
两种 scope 下「管理员都能读」——所以「不让任何人看到」这个目标不在选项里。**

## 三、本项目的实际情况（亲核）

站点**只有 1 处 storage 调用**（`app.js:240`），即**上传**：

```js
// cloud.js:72-77
/* ---- 存储：上传到 shared/<uid>/，所有登录用户可读，仅本人可改可删 ---- */
const path = cloud.storage.sharedPath(session.user.id, `facade/${uuid}.${ext}`);
```

- **没有任何地方读照片**：`app.js` 里没有 `signedUrl()` / `download()` 调用；
- 「大家的贡献」区展示的字段是 `cloud.js:56` 的
  `select('id, contributor_name, building_note, shot_distance_m, created_at')`
  —— **只有文字元数据，不含照片路径**；
- 也就是说：**理论上登录用户能读照片（若知道路径），但站点从不展示路径**。

⇒ 现状是「**照片对登录用户可读，但页面不暴露路径**」。
风险等级：**中低**（需刻意猜测 UUID 路径），但**不是「私有」**。

## 四、页面文案与实际是否一致（亲核）

`index.html:681`：
```
<li>照片只会用于算法训练与研究，不会公开你的邮箱。</li>
```

**这句是成立的**（只承诺了邮箱不公开）。但缺一句关于照片本身的说明。

## 五、★ 一个被文档明确否定的技术事实：UUID 不能当安全屏障

`cloud-code-generation.md:163` / `management.md:163`：
> Permission is checked when the URL is issued.
> URLs are short-lived secrets.

即**签名 URL 是短时效密钥**；而 `shared/<uid>/` 是**所有登录用户可读**的设计。
把「别人猜不到 UUID」当作隐私保护，属于 **security by obscurity**，不能在答辩材料里声称。

## 六、因此正确的做法是两条（改代码 + 改文案），而不是「换桶」

### 方案 A（推荐）：改 `user_private` + 文案说清楚

1. **代码**：`cloud.js` 的 `uploadPhoto()` 由 `sharedPath` 改 `userPath`
   ⇒ 照片变成**只有上传者本人 + 你能读**，其他登录网友**读不到**；
2. **文案**：页面改成明确承诺 ——
   「照片仅用于本项研究，**不会对其他网友公开**；只有研究者本人可访问。」
   （这句在 `user_private` 下是**真的**；在 `shared` 下是**假的**。）
3. **你仍然能拿到全部照片**（管理员权限），不影响你收集素材。

### 方案 B：保持 `shared`，只把文案改准确

保留现状，但文案必须**不承诺照片不公开**，改成
「照片用于算法训练与研究（其他登录用户可见文件存储路径，但页面不展示）」——
这句对普通网友不友好，**不建议**。

## 七、⚠️ 需要用户确认的两点

1. **改 `userPath` 后，已经用 `sharedPath` 上传的旧照片不会自动搬过来**
   （旧路径仍是 shared，仍对登录用户可读）。可选：
   - 用 `cloud.storage.move()` 批量搬到 `users/<uid>/`（需逐个网友身份，**实际做不到**，
     因为 move 需要源所有者权限，而你没有他们的会话）；
   - 或者**接受旧照片留在 shared**，只保证**今后上传的**是 private。
   ⇒ **实际可执行的只有「今后上传走 private」**。这点我必须如实告知，不能假装能全覆盖。

2. **当前是否已有网友投稿？** 若一张都没有，那「旧照片问题」自动不存在，
   可以干净地改成 private。**这需要用户在管理后台看一眼。**

## 八、结论（给用户的一句话）

> 「换私有桶」这个说法**是我给错了** —— SDK 不允许切换桶可见性，也不支持公共桶。
> 真正能做、且应该做的是：**把上传路径从 `shared/<uid>/` 改成 `users/<uid>/`**
> （效果：其他登录网友读不到，你和上传者本人能读），
> 同时把页面文案从「不公开邮箱」加强为「不对其他网友公开」。
> 旧照片若已存在，只能保证「今后上传的」生效。
