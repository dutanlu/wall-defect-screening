# D 路降采样 · 数据构造核验（逐类实例数）

> 目的：证明「B1/B2 相对 V3 只少了 rust，其余 6 类**一个实例都没动**」
> —— 即「唯一变量 = rust 数量」这句话是**可复算的事实**，不是断言。
> 复算脚本：见本文件末尾命令。
> 数据：`01_data/dataset/labels/train`（V3）vs `01_data/_d_downsample/{B1_rust3000,B2_rust1000}/labels/train`

## 逐类实例数

| class | V3 | B1 | B2 | B1−V3 | B2−V3 |
|---|---:|---:|---:|---:|---:|
| crack | 299 | 299 | 299 | **0** | **0** |
| spalling | 207 | 207 | 207 | **0** | **0** |
| efflorescence | 841 | 841 | 841 | **0** | **0** |
| exposed_rebar | 973 | 973 | 973 | **0** | **0** |
| **rust** | 9433 | 3078 | 977 | **−6355** | **−8456** |
| delamination | 486 | 486 | 486 | **0** | **0** |
| moss | 392 | 392 | 392 | **0** | **0** |
| **合计** | 12631 | 6276 | 4175 | −6355 | −8456 |

## 三条结论

1. **只有 rust 变了**：其余 6 类的实例数**逐个为 0 差异** ⇒ 「唯一变量 = rust 数量」成立。
2. **图像数也一致地减**：V3 train 1995 图 → B1 1425 → B2 1248；
   且 `images/train` 与 `labels/train` 文件数**两两相等**（1425/1425、1248/1248）
   ⇒ 无孤儿标注、无缺标注。
3. **口径交叉验证通过**：本表的 V3 列 `[299,207,841,973,9433,486,392]`
   与项目权威来源 `get_class_counts()` 的实测值**完全一致**（见 MEMORY §2.2）
   ⇒ 计数方法可信。

## ⚠️ 必须一并声明的固有混杂

B1/B2 的**训练样本总量也变小了**（1995 → 1425 / 1248）。

⇒ 「B 组 mAP 变化」**不能单独归因于「rust 减少」**，也可能是「总量减少」。
本实验用 B1/B2 两档**强度趋势**部分缓解：若 AP 随 rust 减少**单调变化**，
两种解释都能成立（信息量有限）；若**非单调或反直觉**，则信息量更大。

若要完全解耦，需再做一个「随机删等量非-rust 图」的对照组（成本 ×2），
**本轮未做**，列为遗留工作。

---

复算命令：
```bash
python -c "
import pathlib; from collections import Counter
def c(d):
    r=Counter()
    for f in pathlib.Path(d).glob('*.txt'):
        for ln in f.read_text(encoding='utf-8').splitlines():
            if ln.strip(): r[int(ln.split()[0])]+=1
    return r
print(c('01_data/dataset/labels/train'))
print(c('01_data/_d_downsample/B1_rust3000/labels/train'))
print(c('01_data/_d_downsample/B2_rust1000/labels/train'))
"
```
