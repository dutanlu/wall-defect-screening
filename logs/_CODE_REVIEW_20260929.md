# 综合代码审查报告（2026-09-29）

> **审查对象**：`D:\pythonstudy 备份\创新题\外墙缺陷筛查\`（YOLO+OpenCV+规则 的外墙缺陷筛查系统）
> **审查方式**：两个探查代理分域探查 + 我本人对全部高危项的逐条人工实证
> **报告性质**：纯只读分析，未改动任何工程代码。所有高危项均经二次实证，中低危项来自探查代理
>   的实证扫描（其自身已用 python/git 命令实证关键条目）。
>
> ⚠️ **已修复项不计入新发现**——`window_ok` 默认值、裸 flag 收口、环境类别收口、交付包同步脚本
> 盲区、`single_oom`/`change_detect`/`stitch_facade` 接线，均已在本轮审计-修复中解决，见**附录 A**。

---

## 0. 摘要

**一句话定性：研究纪律一流、工程收口不足。** 核心算法链路（检测→标定 GSD→可判读性自检→
OpenCV 量化→GB50010/JGJ125 分级→弃权）设计严谨，且有罕见的自我验证文化；但**部署层的参数契约
与入口实现缺乏单一真源**，导致多处**静默失效**。

| 严重程度 | 条数 | 主题 |
|---|---|---|
| **高** | 8 | 数据正确性 3 · 参数契约 3 · 安全 1 · 兜底死代码 1 |
| **中** | 14 | 死配置/死代码/死键 · 无鉴权暴露面 · 声明与行为不符 · 测量口径 |
| **低** | 10 | 命名口径漂移 · 字面量 · 调色板多份 · 兜底不一致 |

**最该担心的 2 类问题**：
1. **静默错误结果**——`cls` 恒为 None、伪中位数、安全闸输入失真。这类问题比崩溃更危险：
   285 张图跑完一切"正常"，结论却是错的。
2. **入口分叉导致的口径漂移**——单图/视频/批量/实时流四条路，任何口径改进只改一处，
   就会制造隐性不一致（视频 JGJ125 已是既成事实）。

**优先处理前三条**：① compare_pair `cls` 键名（数据正确性，一行可修）；② `_cert` 私钥出 git
（安全卫生，成本极低）；③ 视频 JGJ125 漏接（功能性失信，一行热修）。

---

## 1. 审查范围与方法

| 范围 | 说明 |
|---|---|
| `02_code/` 核心库与主链路 | `common` `gsd` `measure` `grade` `rectify` `advice` `pipeline` `batch_screen` `video_screen` `evaluate` `train` `change_detect` `single_oom` `stitch_facade` `bench_pipeline` + step0~step6 流水线 |
| `06_deploy/` 部署层 | `app.py`（2182 行）`live_stream.py`（648 行）`phone_live.py`（1122 行） |
| `09_uav_realtime/` 无人机 | `rt_common` `rt_infer` `rt_verify` `uav_probe` `uav_receiver` `uav_loopback_test` + `rt_config.yaml` |
| 方法 | 两探查代理分域探查；我本人对全部高危项做了逐条人工实证（读源码 + 对照读取端/写入端） |

---

## 2. 高危问题

### 2.A 数据正确性（静默产生错误结果，最危险）

---

**H1 ｜ 变化检测输出类名恒为 None（compare_pair 键名错位）**

| 项 | 内容 |
|---|---|
| 位置 | `02_code/pipeline.py:421`（compare_pair 适配器 `_dets`） vs `02_code/change_detect.py:294,323,325` |
| 表现 | 适配器组装配对结果时写 `{"bbox_xyxy": ..., "cls": m.get("cls_name")}`，但 `change_detect.compare_defects` 用 `a.get("cls_name")` 读取。键名不匹配 ⇒ **配对成功时 `cls_t1`/`cls_t2` 及新增/消失缺陷的 `cls` 全部取到 `None`** |
| 影响 | 变化检测输出的类名**全部丢失**，且**静默**（不抛异常、字段还在、只是值为 None）。下游所有按类别统计/展示变化量的结论全部错误。此前我用两张配对失败的测试图验证过"弃权路径"，恰好没触发这条；**配对成功的路径下此 bug 必然发生** |
| 建议 | 改写入端 `cls` → `cls_name`（读取端 `cls_name` 用法多于写入端，改写入端动静最小）。**反向验证**：补丁前构造配对成功用例断言 `cls is None`（固化 bug 存在），补丁后断言 `cls == 期望类名`。**重跑历史变化检测批次**刷新已被污染的结果文件 |
| 严重度 | **高（高危第一条）**——静默产生错误结果。⚠️ 这是我自己写的适配器引入的真 bug |

---

**H2 ｜ 视频聚合的「中位数」实为均值（med() 伪中位数）**

| 项 | 内容 |
|---|---|
| 位置 | `02_code/video_screen.py:116-118` |
| 表现 | 跨帧聚合函数名为「中位数」，实现却是 `statistics.fmean`（算术平均），结果字段名写作 `*_median` |
| 影响 | 字段名 `*_median` 进结果 JSON，下游若有人按"中位数"语义做鲁棒统计（抗离群帧），拿到的是均值——**统计语义说谎**。视频识别卖点就是"时间冗余取中位数"，此处实现与承诺不符 |
| 建议 | 二选一：`fmean`→`median`（改实现，会改变视频汇总数值，需核对演示截图里的数字）；或字段改名 `*_mean`（更稳，只改 JSON 键）。**先 grep 下游是否有人真的依赖 median 语义**再决定 |

---

**H3 ｜ medianBlur 核截断后 `m.ks` 记录的是未截断值（安全闸输入失真）**

| 项 | 内容 |
|---|---|
| 位置 | `02_code/measure.py:116-119` |
| 表现 | 块状分支取核 `ks = max(9, (min(shape)//8) \| 1)` 后 `cv2.medianBlur(enh, min(ks, 31) if ks <= 31 else 31)`——核被静默截断到 31，但随后 `m.ks = int(_KS_RE.search(m.method).group(1))` 记录的是**未截断的 ks** |
| 影响 | `window_ok` 的上界用 `ks−1` 判定"测量窗口是否在可靠区间"。核被截断但记录值未截断 ⇒ **window_ok 的输入失真**，安全闸可能在该放行时拦、该拦时放。而 `window_ok` 是已实证的关键安全闸（285 图零改数证明），其输入失真直接威胁已建立的信任 |
| 建议 | medianBlur 截断后立即把 `m.ks` 回写为**实际使用的截断值**（`min(ks,31)`），使记录与实现一致。**重跑 285 图回归**确认弃权集合不变（或变更可解释）。存疑点：实际触发条件是 ROI 短边 > 248px（ks>31 才截断），需抽样确认覆盖到的图占比 |

### 2.B 参数契约无单一真源（M1 主题簇，本项目病根）

---

**H4 ｜ `analyze()` 内联重写 `run_one` 约 250 行（双轨制）**

| 项 | 内容 |
|---|---|
| 位置 | `06_deploy/app.py:249-549` vs `02_code/pipeline.py:158-392` |
| 表现 | 单图入口 `analyze()` 把检测→标定→可判读性→量化→分级→弃权→渲染整条链路**内联重写**约 250 行（近乎逐行复制 run_one）；而视频/批量入口却调用正主 `run_one`。**单图走副本、视频/批量走正主** |
| 影响 | 代码注释自述弃权逻辑曾因双份漂移导致界面与 CLI 结论不同。现存差异：副本**没有 single_ood 钩子**、没有 `can_report_mm` 字段、标定优先级顺序不同。任何算法改进改 `run_one` 后，单图入口**不会自动得到** |
| 建议 | 见优先级 #4：`analyze()` 改为 `run_one` 薄适配器（组 args → run_one → 只保留渲染）。与新建 `deploy_args.py` 统一参数映射同批做。**注意：抽走渲染链路大概率触发「改可见界面必须重录视频」的连锁成本，排在界面冻结后做** |

---

**H5 ｜ 视频入口 JGJ125 勾选 100% 静默失效**

| 项 | 内容 |
|---|---|
| 位置 | `06_deploy/app.py:636-637` vs `02_code/pipeline.py:229-233` |
| 表现 | `analyze_video` 把 UI 中文串 `"梁板受力主筋处（0.50mm 危险点）"` 直接 `",".join(jgj125_parts)` 写进 `args["jgj125"]`；而 `run_one` 解析是集合精确匹配 `{...} & {"main-rebar","rebar","主筋"}`。整串中文≠任何规范键 ⇒ `is_main_rebar_zone`/`is_slab_tension` **恒为 False** |
| 影响 | 自检 P1-6 专门修复过的危险点判据，**单图入口接了**（`app.py:402-404` 用中文串 `in _parts` 正确解析）、**视频入口漏了**。用户在视频页勾选后，结论仍只走 GB 50010，且**无任何提示**——本项目最忌讳的「参数静默失效」在安全判据上复发 |
| 建议 | 把单图入口 `app.py:402-404` 的中文→英文标识映射抽成小函数 `_jgj125_to_ids(parts)`，两处共用。**真调一次**：取含主筋外露样本，勾选/不勾选两组对照，断言 JGJ125 分级字段出现差异（先固化「补丁前 100% 失效」的实证，补丁后翻绿）。**报告中注明：此前该开关下产出的视频结果无效** |

---

**H6 ｜ 实时流喂给 `run_one` 的 args 字典 11 键错 7 个**

| 项 | 内容 |
|---|---|
| 位置 | `06_deploy/live_stream.py:297-317`（`_load_default_args`）、`06_deploy/phone_live.py:1065-1077` |
| 表现 | 两处用 `analyze()` 的**形参名**组 args：`calib_mode/distance_m/focal_mm/env_class/conf_thr/rectify_mode/jgj125_parts`。但 `run_one` 实际读取的键是 `distance/focal/env/conf/rectify/jgj125/calib_object_px/calib_object_mm/brick_pitch_mm`。**7 个键对不上** |
| 影响 | 之所以"没炸"纯属巧合：错键对应的 run_one 默认值恰好都等于期望值（distance/focal/env/conf）。但 `jgj125` 键**缺失** ⇒ **实时流恒不走 JGJ125**（且与 `rt_common.py:253` 默认 `jgj125=True` 口径相反）。任何人改 `_load_default_args` 任意值都会被**静默忽略**——这是埋着的雷 |
| 建议 | 抽共享「UI/部署参数 → run_one args」映射函数（照 `rt_common.build_args` 的显式映射手法），app/live_stream/phone_live 三处统一调用。根治版 = `deploy_args.py` 用 run_one 实际 `args.get` 键集做白名单校验（多余键报警、缺失键记录）。**验证 = 参数契约测试**：deploy_args 输出键集 ⊆ run_one 读取键集 |

### 2.C 安全

---

**H7 ｜ HTTPS 私钥已 git 追踪入库**

| 项 | 内容 |
|---|---|
| 位置 | `06_deploy/_cert/localhost-key.pem`（另有 `02_code/localhost-key.pem` 拷贝） |
| 表现 | HTTPS 私钥**已被 git 追踪**（`git ls-files` 实证），`.gitignore` 无 `*.pem`/`_cert/` 排除；证书 SAN 仅含 `{localhost, 127.0.0.1, 192.168.3.215}`（硬编码单值） |
| 影响 | 私钥随开源仓库公开（任何人 clone 即得）；机器换 IP 后 SAN 不匹配，手机端告警升级；养成「证书入库」的坏先例。**旧私钥已在 git 历史 = 视为泄露** |
| 建议 | `git rm --cached` + `.gitignore` 加 `*.pem`/`_cert/` + 重签（新私钥永不入库、签发脚本入库、SAN 按需生成而非硬编码）。localhost 自签开发证书，重签即等价吊销。报告注明：不重写 git 历史（成本 > 收益，私钥仅 localhost 开发用） |
| 严重度 | **流程高危 / 技术中危**（攻击价值低，但开源供应链卫生 + 信号意义大）。修复成本极低，放优先清单第一梯队 |

### 2.D 单点

---

**H8 ｜ 默认权重兜底 glob 深度错误（死兜底）**

| 项 | 内容 |
|---|---|
| 位置 | `02_code/common.py:486`（`default_weight()`） |
| 表现 | 兜底 glob 模式 `*/weights/best.pt` 深度错误，源工程目录结构下**永不命中**（权重实际在 `03_weights/`） |
| 影响 | 默认权重解析的兜底分支是**死代码**——主权重缺失时不会兜底而是直接报错；若依赖该兜底（如交付包/新环境），会因找不到权重静默降级或报错 |
| 建议 | 修 glob 深度（`03_weights/*best*.pt` 或候选列表逐个探测）。**反向验证**：临时改名主权重，断言兜底能命中次级权重 |

---

## 3. 中危问题

### 3.A 死配置 / 死代码 / 死键

| # | 位置 | 表现 | 影响 | 建议 |
|---|---|---|---|---|
| M1 | `02_code/batch_screen.py:351-352` | 同时塞 `"env"` 与 `"env_class"` 两个键（后者为死键） | 键名口径混乱，读错键则静默落空 | 统一为 run_one 实际读取的单键 `env` |
| M2 | `09_uav_realtime/rt_config.yaml:106,133` | `infer.half`、`infer.quantize` 从未被 `rt_common.build_args` 读取 | 静默无效配置，读者误以为生效 | 未消费键启动时打警告，或接上/删除 |
| M3 | `06_deploy/app.py:2131-2133` | `allow_origins` 计算后**从未传入** `launch_kw`（死代码 + 误导注释「必须放行」）；`allowed_paths=None` 也是 no-op | 手机访问的 origin 放行实际没走该变量，注释与行为不符 | 按当前 Gradio 版本核实后真传或删除；删除误导注释 |

### 3.B 无鉴权暴露面（`--share` / `0.0.0.0` 部署面）

| # | 位置 | 表现 | 影响 | 建议 |
|---|---|---|---|---|
| M4 | `06_deploy/app.py:294-299` | 上传图无尺寸上限。空闲内存实测仅 0.10~1.41GB，48MP 图三份拷贝 ≈0.5GB | 单请求可逼近 OOM（系统内存仅 16.88GB 的硬约束下） | 长边超阈值先缩放并声明，或响亮拒绝 |
| M5 | `06_deploy/app.py:1921 + 112-143` | `weight_path` 自由文本框直通 `YOLO()` | `--share`/`0.0.0.0` 下远程访客可让服务端加载任意路径（torch.load 反序列化面） | 沙箱到 `WEIGHTS_DIR` 或改下拉框 |
| M6 | 批量「服务端路径」模式 | 可枚举服务器任意目录 | 信息泄露面 | 目录白名单沙箱 |
| M7 | `06_deploy/phone_live.py:938` | `/push` 无 body 上限 | DoS 面 | Content-Length 上限 |
| M8 | 整体 | `0.0.0.0` 下**全无鉴权**：live_stream `/stream` 摄像头画面全局域网开放 | 隐私面 | 启动日志打暴露面警告 + 沙箱 |

### 3.C 声明与行为不符 / 测量口径

| # | 位置 | 表现 | 影响 | 建议 |
|---|---|---|---|---|
| M9 | `06_deploy/live_stream.py:442` | `run_one(frame, model, sub, image=frame.copy())` 把 numpy 数组当 `image_path`，`res["image"]` 变成上千字符的数组字符串垃圾逐帧入 STATE | 结果数据污染（`phone_live.py:328` 用 `Path("<phone>")` 做对了，两处不一致） | 改为 `run_one(Path("<live>"), ...)` |
| M10 | `06_deploy/live_stream.py:31-37` vs `app.py` | 文件头声称「复用 app.py 的 get_model 缓存、绝不各持一份模型」，但 UI 指引恰恰要求另开进程——跨进程无缓存共享，实际两份 18MB 模型 | 声明与行为不符（本项目「注释即文档」强项的一处破坏点） | 改注释或真共享模型 |
| M11 | `06_deploy/app.py:1194-1204` | 从 `method` 字符串 `re.findall(r"\d+")` 取末位 -1 推 `ks` | 文案一改 P3 区间静默错位（定时炸弹），且 `ks` 直接影响测量精度 | `ks` 进 `Measurement` 字段（其实已有 `m.ks`，直接用它） |
| M12 | `06_deploy/app.py:793-807` | 批量入口硬编码 `distance="20"、focal="24"`，批量 Tab 无对应控件 | 近距离拍的图按 20m 算 GSD，毫米数全错且界面无提示 | 暴露三控件，并在汇总中声明标定假设 |
| M13 | 调色板 | 3+1 份拷贝（`pipeline.py:302`、`app.py:475`、`rt_common.py:388`、live_stream 简化版） | 注释自述 6→7 类扩容时已漏改过一次 | 收进 `common.py` 单一调色板 |

---

## 4. 低危问题

| # | 位置 | 表现 | 建议 |
|---|---|---|---|
| L1 | `09_uav_realtime/uav_probe.py:174` | SSID 以 "WIFI" 开头即判连上无人机并发魔数——家用路由器常见此前缀，门禁没注释自称的那么严 | 收紧 SSID 匹配或加注释 |
| L2 | `09_uav_realtime/uav_receiver.py:189-219` | `H264Decoder._drain` 死代码 + `getattr` 绕 `__init__` + read1 阻塞性依赖 peek | 清理 |
| L3 | `09_uav_realtime/uav_receiver.py:528` | `isOpened()` 布尔式 `(A∧B)∨C∨A` 恒真冗余 | 简化 |
| L4 | `09_uav_realtime/rt_infer.py:105-112` | `OPENCV_FFMPEG_OPEN_TIMEOUT/READ_TIMEOUT` 是否 OpenCV 真实支持存疑（常见的是 `OPENCV_FFMPEG_CAPTURE_OPTIONS`） | 核实，不支持则超时配置静默无效 |
| L5 | 环境类别字面量 | 「二类环境（露天/潮湿）」20 处/13 文件（**刻意保留**，`gsd.ENV_CLASSES/DEFAULT_ENV` 已留） | 长期收口为常量引用（rt_config.yaml 必须写字面量除外） |
| L6 | live_stream / phone_live | 端口占用无友好报错（app.py 有兜底，三入口不一致） | 统一兜底 |
| L7 | `06_deploy/app.py:2124` | 端口非法输入直接 ValueError 堆栈 | 友好提示 |
| L8 | `06_deploy/app.py:1654-1688` vs `pipeline.py:393-429` | compare（两次巡检对比）逻辑第三份拷贝 | 收口到 pipeline.compare_pair |
| L9 | `06_deploy/app.py:1697, 2157` | `_theme_and_style` 调两遍 | 调一次 |
| L10 | 命名口径漂移 | `cls_name` / `cls` / `name`、`env_class` / `env` 在多文件间不统一 | 长期统一（这是 H1/H6 等真 bug 的共同土壤） |

---

## 5. 横切主题索引：单一真源破坏清单

> 本项目核心链路的「单一真源」原则（类别 = common.CLASSES、路径 = common 常量、
> 弃权 = grade.apply_abstention 共享实现）守得很好；破坏集中发生在**部署层参数契约**。

| 破坏形式 | 实例 | 指向 |
|---|---|---|
| 参数键名错位（无 schema/无校验） | live_stream/phone_live args 11 键错 7 | H6 |
| 参数键名错位 | batch_screen env/env_class 双键 | M1 |
| 中文串→内部标识转换层漏写 | 单图入口有（app.py:402-404）、视频入口漏（app.py:636） | H5 |
| 同逻辑双份实现 | analyze() ≈ run_one 副本 | H4 |
| 调色板多份 | pipeline/app/rt_common/live_stream 4 份 | M13 |
| 兜底死代码 | common.py:486 glob、allow_origins、infer.half/quantize | H8/M3/M2 |
| 常量分叉 | 环境类别字面量 20 处 | L5 |

---

## 6. 总体评估总结

**定性：研究纪律一流、工程收口不足。** 核心算法链路（检测→标定 GSD→可判读性自检→
OpenCV 量化→GB50010/JGJ125 分级→弃权）设计严谨，且有罕见的自我验证文化；但部署层的
参数契约与入口实现缺单一真源，多处静默失效。

**强项（写明，防止报告一边倒）**：
- 弃权机制单一共享实现（`grade.apply_abstention`）——恰是 M1 的正面范本；
- `_verify_*` 自检体系 + `rt_verify` 六段断言 + 反向验证纪律（window_ok 285 图实证）；
- 注释即文档：关键决策就地留痕（`live_stream` 文件头一处撒谎，反而衬托其余部分可信）。

**病根（一个）**：**单一真源原则在核心链路守住了，在部署层失守了。**
args 字典无 schema、无校验、4 个入口各自拼键名；UI 中文串与内部英文标识的转换层
单图入口写了、视频入口漏写。

**风险画像（最该担心的 2 类）**：
1. **静默错误结果**（cls 恒 None / 伪中位数 / ks 记录失真）——比崩溃更危险；
2. **入口分叉导致的口径漂移**——四条入口路径，任何口径改进只改一处就制造隐性不一致。

---

## 7. 优先处理关键问题清单

成本档位：**T1** = 一行/一处可修（<30 分钟）；**T2** = 需改主链路（半天级，要跑回归）；
**T3** = 需动架构（>1 天）。**连锁成本**列标注是否触发「改可见界面必须重录实机视频」。

| # | 问题 | 为什么先修 | 成本 | 连锁成本 |
|---|---|---|---|---|
| 1 | compare_pair cls 键名（pipeline.py:421）→ 改写入端为 `cls_name` | 输出静默全 None，正在污染已产出的结果；T1 成本换数据正确性 | T1 | 无界面改动；需重跑历史变化检测批次刷新被污染结果 |
| 2 | `_cert` 私钥出 git + `.gitignore` + 重签 | 成本最低的安全卫生修复，开源前必须做 | T1 | SAN 加新 IP 时更新文档，不动界面 |
| 3 | 视频 JGJ125 漏接（app.py:636）→ 复用单图入口转换 | 用户勾选 100% 无效，功能性失信；先 T1 热修止血 | T1 | **不重录**；报告中注明此前该开关下结果无效 |
| 4 | 新建 `deploy_args.py` 统一参数映射，消 analyze()/run_one 双份 + 键名错位 | 病根修复，一次消掉 M1 全部症状，阻断未来漂移 | T3 | **大概率重录**；排在界面冻结后，或先只抽参数层降为 T2 |
| 5 | med() 伪中位数（video_screen.py:116）→ 改 median 或字段改名 | 一行消除统计语义谎言 | T1 | 改实现会变汇总数值（核对演示截图）；改名字段更稳 |
| 6 | medianBlur 截断后回写 `m.ks`（measure.py:116-119） | 安全闸输入失真，威胁已实证的 window_ok 闸 | T1 | 需重跑 285 图回归确认弃权集不变 |
| 7 | 无鉴权暴露面收口（路径枚举/上传上限/weight_path 白名单） | --share / 0.0.0.0 下的实际攻击面 | T2 | 加白名单不改布局**不重录**；加鉴权页才重录 |
| 8 | 调色板 4 份收口 common + 死配置/死代码清理（M2/M3/M13） | 防漂移收尾，可与 #4 同批做 | T2 | 先 diff 四份调色板是否一致，有色差才重录 |

**节奏建议**：#1/#2/#3/#5/#6 作「T1 热修批」一天内清掉；#7 第二批；
#4/#8 作「架构批」放在演示视频定稿之后、或明确接受重录时再做。

---

## 附录 A：已修复项（本轮审计-修复中已解决，附验证证据指针）

| 项 | 修复 | 证据 |
|---|---|---|
| `window_ok` 默认值 True→False | `measure.py:64` | 285 图零改数证明：`logs/_run_window_ok_probe.py` / `_diff_window_ok.py` |
| 裸 flag 单一真源 | 三入口委托 `common.argv_flag` | `argv_flag` 纯单测通过 |
| 环境类别收口 | `gsd.ENV_CLASSES`/`DEFAULT_ENV`（app UI + rt_verify 引用） | import 冒烟验证 |
| 交付包同步脚本盲区 | `_pack_vs_src.py` 补 `09_uav_realtime` MAP + 通用 `__pycache__` 规则 | `_pack_vs_src.py` 终查 PASS |
| 三能力接线 | `single_oom`→`pipeline`（--single-ood）/`change_detect`→`pipeline.compare_pair`+Web Tab/`stitch_facade`→`pipeline.stitch_facade` | 真调一次验证 |

## 附录 B：行号快照

> ⚠️ 本报告引用的 `文件:行号` 基于 2026-09-29 的代码快照。**T1 热修批（#1/#2/#3/#5/#6）
> 落地后行号会漂移**，届时应用 `logs/_audit_line_refs.py` 的思路做机器校验并统一刷新本附录，
> 避免报告里最精确的话最先过期。

---

*本报告由两个探查代理 + 我本人共同完成：高危项均经我本人二次人工实证（读源码 + 对照读写端），
中低危项来自探查代理的实证扫描。已修复项不计入新发现。*
