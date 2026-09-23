# Kaggle 匿名直下通道突破报告（2026-09-20 22:53 — 23:05）

## 一、要解决的问题

本项目 7 类外墙缺陷里，`rust`（锈蚀）与 `delamination`（空鼓/脱层）**零数据**，
`exposed_rebar`（露筋）仅 9 框/2 图 —— 答辩必被追问。

此前已穷尽三条路：
- **hf-mirror**：找到 `famerL` / `ROI-1555` 两个「露筋」候选，**抽样看图后全部否决**（场景/语义不匹配）。
  用 7 组关键词检索确认 **hf-mirror 上不存在能补这三类的来源**。
- **Mendeley**：RC2119 / RC1841 / HRCDS 三个数据集类别对口，但**只有 15 KB/s**，
  下载 0.95 GB 约需 18 小时，仍在后台自愈。
- **Kaggle**：此前判断「登不上 ⇒ 整体不可用」，**已放弃**。

**本次突破点就是推翻了第三条判断。**

---

## 二、★ 核心发现：Kaggle 公开 API 可匿名直下

### 2.1 此前误判的根因

因为「没有 Kaggle 账号 / API token / 网页登不上」，就把 Kaggle 整条路划掉。
**这是「平台级判死」的典型错误** —— 实际情况是：

**Kaggle 有一套公开 REST API，一部分数据集不需要任何认证就能查看元信息并下载。**

```bash
# 元信息（无需登录）—— 返回真实 JSON
curl -sS 'https://www.kaggle.com/api/v1/datasets/view/<owner>/<slug>'

# 下载（无需登录）
curl -sS -L -A 'Mozilla/5.0' -o out.zip \
     'https://www.kaggle.com/api/v1/datasets/download/<owner>/<slug>'
```

**`view` 返回的关键字段**：
`title` / `subtitle` / `totalBytes` / `licenseName` / `lastUpdated` / `downloadCount` /
`usabilityRating` / `description` / `files` / `versions`

### 2.2 ★★ 但必须逐个探测 —— **不存在「Kaggle 能用/不能用」的整体结论**

同一条通道上两类数据集并存。实测 6 个：

| 数据集 | view API | download API | 结论 |
|---|---|---|---|
| `sanjuchinni/urban-infrastructure-anomalies` | ✅ 200 | ✅ **206 + `50 4B 03 04`** | **可匿名下** |
| `mennamahmoudd/mbdd2025-building-defects` | ✅ 200 | ✅ **206 + 真 ZIP** | **可匿名下** |
| `saisirishan/construction-defects` | ✅ 200（`totalBytes=0`） | ❌ **403** | 需登录 |
| `andy8744/building-defect-detection` | ✅ 200（`totalBytes=0`） | ❌ 403 | 需登录 |
| `aioctopus/construction-defects` | ✅ 200（`totalBytes=0`） | ❌ 403 | 需登录 |
| `jeremieguinard/concrete-cracks` | ✅ 200（`totalBytes=0`） | ❌ 403 | 需登录 |

**⇒ 三条判据**：
1. **`view` API 的 200 不能作为可用性判据** —— 它对**所有**数据集都返回 200。
2. **`totalBytes=0` 是「需登录」的强信号。**
3. **只有 `download` + Range 取 1 KB 看文件头**，拿到 `50 4B 03 04`（ZIP）才能定论。

### 2.3 落地节点与速度

**302 重定向 → `storage.googleapis.com`（Google Cloud Storage）**，带签名 URL：
```
Location: https://storage.googleapis.com:443/kaggle-data-sets/<id>/<vid>/bundle/archive.zip
          ?X-Goog-Algorithm=GOOG4-RSA-SHA256
          &X-Goog-Expires=259200          ← 3 天有效
          &X-Goog-Signature=...
```

**单连接速度实测**：

| 参数组合 | 实测速度 |
|---|---|
| 默认（http2/3） | 75 KB/s |
| **`--http1.1`** | **116 KB/s**（+55%） |
| `--no-keepalive` | 95 KB/s |
| 带 `-C -` 断点续传 | ⚠️ 反而降到 24 KB/s（疑似并发抢占） |

**★★ 关键加速手段：GCS 支持 Range，且各段速度互相独立。**
实测同一文件 4 个分段各测 12 秒：

```
seg0 [0-293679015]            75,645  B/s
seg1 [293679016-587358031]   113,154  B/s
seg2 [587358032-881037047]   129,058  B/s
seg3 [881037048-1174716063]  155,346  B/s
```

⇒ **4 段合计约 470 KB/s = 单连接的 6 倍。**
⇒ **这就是本次采用的方案**：8 段并行 + `--http1.1` + 每段独立断点续传。

---

## 三、★★ 由此拿到的两个 P0 数据集

### 3.1 Urban Infrastructure Anomalies（**最关键**）

| 项 | 值 |
|---|---|
| 标识 | Kaggle `sanjuchinni/urban-infrastructure-anomalies` |
| 体积 | **1,174,716,064 B = 1,120.3 MB** |
| 许可 | **Apache 2.0** |
| 规模 | **~16,946 图**，已划好 Train 12.9k / Val 2,772 / Test 1,274 |
| 格式 | **YOLOv8 `.txt`** ⇒ **与我们管线零转换成本** |
| 官方基线 | YOLOv8s：mAP@50 **0.8031** / mAP@50-95 0.5970 / P 0.7367 / R 0.7192 |
| 来源 | DST-SGP 资助的 Urban Vision 项目，含无人机与监控影像 |

**13 类**：
`corrosion`、**`delamination`**、`cover_detachment`、`efflorescence`、`crack`、
**`rust`**、`scaling`、`spalling`、`void`、`tile_crack`、`tile_delamination`、`tile_loss`、`peeling`

**⇒ 这是唯一能同时补齐 `rust` + `delamination` 两个零数据类的来源。**
其中 `efflorescence`、`crack`、`spalling` 与我们**逐字同名**（但仍需核定义）。

⚠️ **必须注意的差异**：
- `tile_*` 三类是**瓷砖墙**，与我们混凝土/砖墙**异源**；
- 官方自述 `void` / `tile_loss` **样本偏少**；
- `corrosion`（锈蚀）与 `rust`（锈迹）在该集里是**两个不同类**，需看定义区分。

### 3.2 MBDD2025

| 项 | 值 |
|---|---|
| 标识 | Kaggle `mennamahmoudd/mbdd2025-building-defects` |
| 体积 | 2,561,278,781 B = **2,442.6 MB** |
| 许可 | **MIT** |
| 规模 | 14,471 图，UAV 航拍，覆盖 6 种结构（钢/木/砌体/砖木/砖混/钢筋混凝土） |
| 格式 | **PASCAL VOC（XML）** ⇒ 需转 YOLO |
| 类别 | `crack` / `leakage` / `abscission` / `corrosion` / **`bulge`** |

⚠️ **`bulge` = 外观可见的外凸鼓包 ≠ 我们定义的「饰面层与基层脱开的内部空鼓」**
⇒ **不可直接改名充数**，须在报告注明语义差异。价值主要是 `corrosion` 与结构多样性。

---

## 四、⚠️ 新增待核查项（**明天第一件事**）

**Urban 数据集虽类别对口，但还没做过「抽样看图」。**

**为什么必须做**：今晚 famerL 与 ROI-1555 **两次都是「类名对口、场景不对口」**：
- `rebar-seg` 看着是露筋，实为**施工钢筋绑扎网格**（且每根一框，平均 21.4 框/图）；
- ROI-1555 看着是露筋，实为**实验室摆拍试件**（环氧地坪 + 砖块垫高 + 夹具）。

**⇒ 纪律：下完必须先渲染带框图亲眼确认场景。** 判据沿用：
1. **看背景** —— 是否外墙现场 / 是否可见光实景（而非实验室、棚拍、室内）；
2. **看平均框数** —— 异常高（如 >10 框/图）通常是「部件实例」而非「缺陷实例」；
3. **看框的粒度** —— 一处缺陷一个框，还是每个构件一个框；
4. **看类别定义** —— 尤其 `rust` vs `corrosion`、`delamination` vs `void` vs `cover_detachment`。

**三者（场景 + 粒度 + 语义）全部通过，才可并入。**

---

## 五、当前下载状态（23:05）

| 任务 | 内容 | 方式 | 状态 |
|---|---|---|---|
| `oxsqop` | Urban Infrastructure Anomalies 1.12 GB | **8 段并行** | 🔄 运行中（~330 KB/s） |
| `irengs` | MBDD2025 2.44 GB | **8 段并行** | 🔄 已启动 |
| `ceg5gs` | Mendeley 7 文件（含 RC2119/RC1841/HRCDS） | 单流自愈 | 🔄 运行中（15 KB/s） |
| `phohyd` | famerL 890 MB | — | ✅ 已完成（SHA256 校验通过） |
| `13e1r4` | ROI-1555 | — | ⛔ 已停止（场景不匹配） |

**下载器**：`logs/_kaggle_seg_dl.py`（Python，因本机 `Start-Job` 与 `.ps1` 均被禁用）
- 解析 302 拿 GCS 签名 URL；
- 8 段并行 `curl --http1.1 -r a-b`，每段独立断点续传（最多 200 次重试）；
- 全部段完整后二进制拼接，校验总字节数；
- 日志：`logs/_seg_dl.log`；每段日志：`<zip>.partN.log`。

---

## 六、可复用方法论（已写入技能库）

1. **平台级判死必须「逐资源探测」** —— 不能因一次入口失败否掉整个平台。
   本项目**两次**栽在这上面：HF（官方域 502 ⇒ 差点放弃，实则 hf-mirror 可用）、
   Kaggle（登不上 ⇒ 放弃，实则公开 API 匿名可用）。
2. **换入口 → 换资源 → 看真实字节**，三步缺一不可：
   - 换入口：官方域不通试镜像；网页不通试 REST API；认证不通试匿名；
   - 换资源：同平台内逐数据集试（Kaggle 实测两类并存）；
   - 看字节：只有文件头对（`50 4B 03 04`）才算可下载。
3. **大文件下载先测「能否分段并行」** —— GCS/S3 通常支持 Range，
   分段速度独立 ⇒ 可线性提速（本次 6 倍）。**先花 1 分钟测 4 个 Range 的各自速率**，
   再决定是否值得写并行下载器。
4. **`--http1.1` 在国内连 GCS 时比默认协议快 55%** —— 值得作为默认参数。
5. **数据集「类别对口」不等于「可用」** —— 必须抽样看图，三者（场景/粒度/语义）全过才并。

---

## 七、留痕文件

| 文件 | 内容 |
|---|---|
| `logs/_KAGGLE_BREAKTHROUGH.md` | **本文件** |
| `logs/_kaggle_api.txt` | 元信息 + Range 探测（urban 1,120.3 MB / Apache 2.0） |
| `logs/_kaggle_multi.txt` | 6 个数据集的 view API 批量核查 |
| `logs/_kag_verify.txt` | 6 个数据集的 download API 逐个探测（206 vs 403） |
| `logs/_kag_speed.txt` | 302 响应头（GCS 签名 URL）与速度对比 |
| `logs/_kag_fast.txt` | 三种 curl 参数组合的速度对比 |
| `logs/_gcs_parallel.txt` | **4 段 Range 并行速度实测（关键证据）** |
| `logs/_kaggle_seg_dl.py` | **分段并行下载器**（可复用） |
| `logs/_seg_dl.log` | 分段下载实时日志 |
