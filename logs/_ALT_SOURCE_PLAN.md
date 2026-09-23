# 补类数据源改道方案（Kaggle 不可用后的替代路线）

> 记录时间：2026-09-20
> 触发原因：用户 Kaggle 登录不通，需改用匿名可访问的数据源。

---

## 一、本机网络实测连通性（2026-09-20 21:16）

| 站点 | 结果 | 耗时 | 结论 |
|---|---|---|---|
| **Mendeley Data** | ✅ 通 | 2.9 s | **可用**（BFDD 正常下载中） |
| **Roboflow Universe** | ✅ 通 | 1.8 s | **可用** |
| **GitHub API** | ✅ 通 | 1.4 s | 可用 |
| **Kaggle** | ⚠️ 首页通 | 2.7 s | 但**登录态拿不到**，API 走不通 |
| **Zenodo** | ❌ 403 | 0.9 s | 被拦截：「unusual traffic from your network」 |
| **HuggingFace** | ❌ 连接失败 | exit=7 | 不可达 |
| 百度 | ✅ 通 | 0.3 s | 基线正常 |

---

## 二、评估过的候选源及其排除原因

| 源 | 结论 | 排除原因 |
|---|---|---|
| **Kaggle Urban Infrastructure Anomalies** | ❌ 不可用 | 需登录认证，用户登不上 |
| **GitHub `lonlonago/Rust-and-Corrosion-...`** | ❌ 排除 | 实为**付费引流页**（要 $89 / Stripe）；且内容是**螺栓螺母锈蚀**，非建筑外墙；303 图 / 6 类全是 `rust_bolt`/`rust_nut` 等紧固件 |
| **Zenodo** | ❌ 不可用 | 403 反爬拦截 |
| **HuggingFace** | ❌ 不可用 | 网络不可达 |
| **Roboflow Universe** | ⚠️ 部分可用 | rust 检索结果**绝大多数是 Rust 游戏玩家检测**或**钢结构腐蚀**，与建筑外墙语义不符；需逐个甄别 |

---

## 三、仍然有效的可用数据源

### ✅ BFDD（正在下载）
- 来源：Mendeley（网通）
- 文件：`BFDD Dataset_1x_20260408.tar.gz`，553,316,751 B（527.7 MB）
- 状态：**下载中**（截至 21:18 已 16.18 MB / 3.07%）
- 用途：**仅作学术论据**（用户已拍板），证明「空鼓需红外手段」；含 Hollow Areas 标注，788 对 RGB-IR 严格对齐
- 许可：CC BY 4.0

### ⬜ 尚未验证但理论可用的 Mendeley 数据集
- **RC 1841**（1,841 图，含 major spalling / delamination、rebar corrosion，143 MB）
- **MDMCS**（1,200 图，含 exposed rebar，195 MB）
- 两者均为 Mendeley 直链，**理论上可匿名下载**（同 BFDD 路径），需逐个实测 `HTTP 206` 验证

---

## 四、待用户决策的路线

| 选项 | 内容 | 能补的类 |
|---|---|---|
| **A** | 验证并下载 Mendeley 上的 RC 1841 / MDMCS | delamination、exposed_rebar |
| **B** | 用户在浏览器手动下载 Kaggle zip（不登录 API，只网页点击） | rust、delamination |
| **C** | 换用 Roboflow 上甄别出的建筑类数据集 | 视具体数据集 |
| **D** | 接受现状，rust 保持零数据并在报告中说明 | — |

---

## 五、方法论教训（可复用）

1. **判断「能否下载」必须用 GET + Range，HEAD 会被反爬层拦截并给出误导性状态码**（本项目已验证：HEAD 全 404，GET 才见真相）。
2. **Kaggle 数据集页面「看起来能匿名访问」是假象**——页面 GET 返回 200 + HTML 登录页，实际文件下载必须认证。
3. **GitHub 上标题带完整类别名 + 画质预览图的仓库，多半是数据集售卖引流页**，不是真开源数据。特征是：README 极长、列出精确框数、末尾带 Stripe/pay 链接、仓库里只有几张预览图 + README。
4. **检索关键词要带领域限定**：单搜 `rust` 会被 Rust 游戏和钢结构淹没，需加 `facade` / `concrete` / `building` 限定。
5. **各站点可达性差异极大**，动手前先做连通性探测，避免在不可达的源上浪费时间。
