# famerL/construction_defect_yolo 结构分析

## 一、目录与数量
- `images/train` : 4583 个文件
- `images/val` : 457 个文件
- `labels/train` : 2410 个文件

## 二、图片文件名前缀分布（images/train）
- `damp-stain` : 963
- `spalling` : 704
- `rebar-seg` : 687
- `roof-damage` : 648
- `efflorescence` : 601
- `crack` : 490
- `leakagewaterstain` : 490

## 三、类别号分布（labels/train）
- 标注文件数: 2410
- 类别 `0` : 1556 个框
- 类别 `1` : 834 个框
- 类别 `2` : 2361 个框
- 类别 `3` : 391 个框
- 类别 `4` : 6499 个框
- 类别 `5` : 1473 个框
- 类别 `6` : 407 个框

## 四、★ 前缀 × 类别号 交叉表（推断类名用）

| 文件名前缀 | 类别号 | 该组合的文件数 |
|---|---|---|
| damp-stain | 1 | 605 |
| efflorescence | 2 | 380 |
| crack | 0 | 305 |
| rebar-seg | 4 | 304 |
| roof-damage | 5 | 301 |
| spalling | 6 | 283 |
| leakagewaterstain | 3 | 231 |

## 五、判读
文件名前缀极可能来自原始数据集（如 CODEBRIM 的 CCC/CSP/CDC 缩写）。
交叉表可确认「前缀 → 类别号」的一一对应关系，从而还原类名。
