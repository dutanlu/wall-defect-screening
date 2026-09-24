# 多种子方差汇总（自动生成，勿手改）

> 生成脚本：`logs/_multiseed_variance.py`；数据源：`04_results/eval/*_test_metrics.json`。
> 口径：独立测试集 285 图 / 1646 实例 / 7 类；std 为样本标准差（n−1）。

```
==============================================================================
多种子方差汇总
==============================================================================

【基线 v11s640（cls_pw=0.0，seed=42 单点）】
  mAP50        0.7230
  mAP50-95     0.4410
  precision    0.7707
  recall       0.6693

【类平衡 v11s640_clsbal（cls_pw=0.5）各随机种子】
  seed=42     mAP50=0.7281  mAP50-95=0.4477  precision=0.7652  recall=0.6585
  seed=123    mAP50=0.7191  mAP50-95=0.4298  precision=0.7566  recall=0.6627
  seed=2024   mAP50=0.7223  mAP50-95=0.4398  precision=0.7436  recall=0.6591

【跨种子 mean ± std (n=3)】
  mAP50        0.7232 ± 0.0046
  mAP50-95     0.4391 ± 0.0090
  precision    0.7551 ± 0.0109
  recall       0.6601 ± 0.0023

【判读：clsbal 与基线的差 vs clsbal 自身种子方差】
  mAP50        base=0.7230  clsbal=0.7232  Δ=+0.0002  seed_std=0.0046  → 差值 < 种子标准差 ⇒ 判不出高下
  mAP50-95     base=0.4410  clsbal=0.4391  Δ=-0.0020  seed_std=0.0090  → 差值 < 种子标准差 ⇒ 判不出高下
  precision    base=0.7707  clsbal=0.7551  Δ=-0.0156  seed_std=0.0109  → 差值 > 种子标准差（仅提示，需多样本检验）
  recall       base=0.6693  clsbal=0.6601  Δ=-0.0092  seed_std=0.0023  → 差值 > 种子标准差（仅提示，需多样本检验）

JSON -> D:\pythonstudy 备份\创新题\外墙缺陷筛查\04_results\eval\multiseed_variance.json
```
