# OpenDataLab 访问方式破解记录

> 时间：2026-09-20 21:45
> 目的：绕过 Zenodo 403，从 OpenDataLab 获取 CODEBRIM（含 `exposed bars` + `corrosion` 类）

---

## 一、结论速览

| 项目 | 结果 |
|---|---|
| OpenDataLab 是否可达 | ✅ 可达（国内平台，网络无阻） |
| CODEBRIM 是否在库 | ✅ 在库，`id = 1660`，`name = OpenDataLab/CODEBRIM` |
| 平台显示体积 | `fileSize: 2203` MB（约 2.2 GB），`fileNum: 3` |
| 下载次数 | 411（说明确实可下） |
| 许可 | **Academic use only / 禁止商业用途**，原始出处 Zenodo `10.5281/zenodo.2620293` |
| 前端渲染方式 | 纯 SPA（React + Remix Router），所有内容走 XHR |
| **官方下载方式** | **`odl get <dataset_name>`**（`odl` CLI） |

---

## 二、★ 已破解的 API 端点（实测有效）

前端站点：`https://openxlab.org.cn/datasets/...`
JS bundle：`https://webpub.shlab.tech/dps/opendatalab-web/xlab_v5.2103/assets/index-cccb28a0.js`

### 2.1 搜索建议（可用）
```
GET /datasets/api/v3/search/autoSuggest?keywords=<kw>
→ 200 {"code":0,"msg":"ok","data":["CODEBRIM (COncrete DEfect BRidge IMage Dataset)"]}
```

### 2.2 ★ 数据集列表 / 搜索（最有用）
```
POST /datasets/api/v3/datasets/list
Body: {"keywords":"concrete","pageNo":1,"pageSize":20}

→ 200 {"code":0,"data":{"list":[
    {"id":1660,"name":"OpenDataLab/CODEBRIM",
     "displayName":"CODEBRIM (...)",
     "fileSize":2203,"fileNum":3,"downloadCount":411,
     "attrs":{"fileBytes":2353,"fileCount":1,...}}
  ],"total":1}}
```
**这是拿数据集 id / name 的主入口。**

### 2.3 ★ 数据集详情（用 id）
```
GET /datasets/api/v3/datasets/<id>          → 例 /datasets/api/v3/datasets/1660
GET /datasets/api/v2/datasets/<id>          → 同样有效
```
返回字段包含：`attrs.citation`、`attrs.license`、`attrs.publisher`、`attrs.paperUrl`、
`publicCommit`（git 提交哈希）、`fileSize`、`downloadCount` 等。

### 2.4 其他已验证路径
| 端点 | 结果 |
|---|---|
| `GET /datasets/api/v3/datasets/<name>` | ❌ 404（**必须用 id，不能用 name**） |
| `GET /datasets/api/v2/datasets/detail?id=<id>` | ❌ 404 |
| `GET /datasets/api/v3/datasets/<name>/r/main` | ❌ 404 |
| `GET /datasets/api/v1/downloadCheck/...` | ❌ 返回 HTML |

### 2.5 从前端 JS 提取到的其他模板（未全部验证）
```js
// datasetsV3-b4eca022.js
o.post("api/v3/datasets/list", {...})                       // 列表 ✅ 已验证
o.get (`/datasets/api/v3/datasets/${a}`)                    // 详情（a 需为 id）✅
o.post("/datasets/api/v3/datasets", a)
o.get (`api/v3/datasets/${ds}/r/${ref}`, {params})          // 仓库内容
o.get (`${C}/downloadCheck/${a}/${b}`)                      // 下载检查

// datasets.detail-68410920.js
e.post(`${o}/datasets/${ds}/r/${ref}/_agg`, body)           // 聚合
e.post(`${o}/datasets/${ds}/r/${ref}/_search`, body)        // 搜索（405 说明路径在，需认证头）
```

---

## 三、★ 官方下载通道：`odl` CLI

来自官方文档 `opendatalab.github.io/dsdl-docs/tutorials/dataset_download/`：

```bash
# 安装（pip 包名待确认，疑似 openxlab 或 dsdl）
pip install openxlab

# 下载
odl get CODEBRIM
```

**下载后目录结构**：
```
<root>/
├── compressed/        # 原始数据集压缩包
└── dsdl/              # 标注定义
    ├── defs/
    ├── set-train/  set-val/  set-test/
    └── tools/prepare.py
```

**注意**：文档未说明是否需要登录 / API token。**实际能否匿名下载需实测。**

---

## 四、待验证事项

1. `odl get` 是否需要登录（可能要 `openxlab login` 填 AK/SK）
2. CODEBRIM 在平台上 `fileNum: 3` / `fileSize: 2203 MB`，但 `attrs.fileCount: 1` —— 
   **需确认实际托管的文件与 Zenodo 原版（4 个子集）是否一致**
3. 该平台是「转发+镜像」，部分数据集（情形 1.2/1.4）**只提供 DSDL 标注，不含原始媒体文件**

---

## 五、方法论价值

- **SPA 站点的内容必须从 JS bundle 反查**：页面 HTML 只有 2.6 KB 的框架壳，
  真实接口全在 `index-*.js` 与 `datasetsV3-*.js` 等分包里。
- **`405 Method Not Allowed` 是强信号**：说明路径存在、只是方法或鉴权不对。
- **先用「列表/搜索」接口拿 id，再用 id 查详情** —— 直接猜 name 拼路径会 404。
