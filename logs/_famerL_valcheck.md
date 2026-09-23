# famerL —— val 集标注缺失核实

## 一、文件数
- images/train: 4583
- images/val  : 457
- labels/train: 2410
- labels/val  : 0  ← 目录不存在

## 二、配对情况
- train 图有标注的: 2410 / 4583
- train 图无标注的: 2173
- val   图有标注的: 0 / 457
- val   图无标注的: 457

## 三、★ train/val 泄漏细节
- 同名（同 stem）图片: 363
- 占 train 比例: 7.9%
- 占 val   比例: 79.4%

具体重叠样例（前 15 个 stem）:
- `crack_image-10-ver4_jpg.rf.a65974d989194e7b2752f32ea64dca43`
- `crack_image-100-_jpg.rf.9c8a096e76630d1582e4c95549197931`
- `crack_image-14-_jpg.rf.18cc536a4f31c1d0c87d63adf3b94800`
- `crack_image-15-_jpg.rf.979dfb866c95be9bbcf5aa57fcde1431`
- `crack_image-16-ver5_jpg.rf.9be341519681adc0556eba91bc2a045e`
- `crack_image-19-ver3_jpg.rf.22d7be8aed50835108913fd6882f1949`
- `crack_image-20-ver3_jpg.rf.cb0ed29ff5ca671897bb6388bdc072a6`
- `crack_image-2024-07-31T153418-313_jpg.rf.af8540bf13cd9832e17420c079a307ea`
- `crack_image-2024-07-31T192724-743_jpg.rf.6e90e06a46973ded5b6cff8acd20bd80`
- `crack_image-2024-07-31T192838-448_jpg.rf.252fcee8954969f8389aadf2a656df1e`
- `crack_image-2024-08-01T132955-571_jpg.rf.3b10e71a7d44ed5cde5e48673eca2c15`
- `crack_image-2024-08-01T135124-764_jpg.rf.8dd479c2f07e81beb233c4c053edae95`
- `crack_image-2024-08-01T143945-416_jpg.rf.013497da8719df644cb808302cb0d595`
- `crack_image-2024-08-01T144321-379_jpg.rf.130ca4e5c15dcc76e6b373b0147e6279`
- `crack_image-2024-08-09T175507-318ver2_jpg.rf.9f0fd0095630869f0e4e91ad41274450`

## 四、★ 判读

1. **val 集无标注文件** ⇒ 无法作为独立评估集使用。
2. **train/val 79.4% 重叠** ⇒ 与上一轮 v8s640 的泄漏问题同源，
   该数据集的分割方式不可用于独立评估。
3. **正确用法**：
   - **只把它的 train 图当作新数据源**，纳入我们自己的去重+重划分管线；
   - **绝不采信它自带的 val 划分**；
   - 评估仍用我们自己的独立测试集。

4. **rebar-seg（露筋）304 文件 / 6499 框** 是本次最有价值的收获 ——
   但须注意其「平均每图 21.4 框」异常高，可能标注粒度很细（单根钢筋各一框），
   与我们「露筋区域」的标注粒度可能不同，**需抽样看图确认**。
