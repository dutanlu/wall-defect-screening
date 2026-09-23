# famerL 数据集 —— rebar-seg（露筋）专题统计

## 一、val 集完整性
- `images/train` : 4583 个文件 ✅存在
- `images/val` : 457 个文件 ✅存在
- `labels/train` : 2410 个文件 ✅存在
- `labels/val` : ❌不存在

## 二、★ train/val 图片重叠检查（防泄漏）
- images/train 唯一名: 4583
- images/val   唯一名: 457
- **同名交集: 363**
- 交集占 val 比例: 79.4%

## 三、rebar-seg（露筋）专题
- rebar-seg 标注文件数: 304
- rebar-seg 标注框总数: 6499
- 平均每图框数: 21.38

## 四、全部 7 类的实例总数
| 类别号 | 类名 | 实例数 |
|---|---|---|
| 0 | `crack` | 1556 |
| 1 | `damp-stain` | 834 |
| 2 | `efflorescence` | 2361 |
| 3 | `leakagewaterstain` | 391 |
| 4 | `rebar-seg` | 6499 |
| 5 | `roof-damage` | 1473 |
| 6 | `spalling` | 407 |

**合计 13521 个实例框**

## 五、与外墙项目 7 类的映射建议

| famerL 类名 | 我们的类 | 建议 |
|---|---|---|
| `crack` | crack | 直接对应，已有充足数据 |
| `efflorescence` | efflorescence | 直接对应，已有充足数据 |
| `spalling` | spalling | 直接对应，已有充足数据 |
| **`rebar-seg`** | **exposed_rebar** | **★ 直接补空缺！露筋** |
| `damp-stain` | efflorescence? | 语义近似（潮湿污渍↔泛碱），需看图 |
| `leakagewaterstain` | ? | 渗水痕，我们 7 类无对应，可能不并入 |
| `roof-damage` | ? | 屋面损伤，非外墙立面，可能不并入 |
