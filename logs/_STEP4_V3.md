# cmd: C:\Users\dutanlu\.workbuddy\binaries\python\versions\3.13.12\python.exe step4_split.py --in=D:\pythonstudy 备份\创新题\外墙缺陷筛查\01_data\_v3_build --out=D:\pythonstudy 备份\创新题\外墙缺陷筛查\01_data\dataset --ratio=0.7,0.2,0.1 --seed=42
# cwd: D:\pythonstudy 备份\创新题\外墙缺陷筛查\02_code
# start: 2026-09-21 09:22:43

[09:22:43] 输入: D:\pythonstudy 备份\创新题\外墙缺陷筛查\01_data\_v3_build
[09:22:43] 图像-标签配对: 2848，无标签图像: 0
[09:24:57] train:  1995 张 /  12631 实例  {'exposed_rebar': 973, 'rust': 9433, 'delamination': 486, 'spalling': 207, 'efflorescence': 841, 'crack': 299, 'moss': 392}
[09:25:37] val  :   568 张 /   3985 实例  {'delamination': 140, 'rust': 3103, 'moss': 103, 'efflorescence': 195, 'crack': 112, 'spalling': 63, 'exposed_rebar': 269}
[09:25:57] test :   285 张 /   1646 实例  {'rust': 1149, 'delamination': 68, 'exposed_rebar': 147, 'efflorescence': 116, 'crack': 58, 'spalling': 43, 'moss': 65}
[09:25:57] ------------------------------------------------------------
[09:25:57] 跨集交集自检（必须全为 0）
[09:26:09]   OK train-val names 交集 0
[09:26:09]   OK train-val md5 交集 0
[09:26:09]   OK train-val dhash 交集 0
[09:26:09]   OK train-test names 交集 0
[09:26:09]   OK train-test md5 交集 0
[09:26:09]   OK train-test dhash 交集 0
[09:26:09]   OK val-test names 交集 0
[09:26:09]   OK val-test md5 交集 0
[09:26:09]   OK val-test dhash 交集 0
[09:26:09] ============================================================
[09:26:09] 划分完成：{'train': 1995, 'val': 568, 'test': 285}
[09:26:09] 数据集配置: D:\pythonstudy 备份\创新题\外墙缺陷筛查\01_data\dataset\wall_defects.yaml
[09:26:09] 无泄漏自检: 通过
[09:26:09] ============================================================

# end  09:26:09  elapsed=206.5s  rc=0
