==============================================================================
V3 组装：多源合并 + L1/L2 两级去重（--near=3）
==============================================================================
输入 D:\pythonstudy 备份\创新题\外墙缺陷筛查\01_data\unified
输出 D:\pythonstudy 备份\创新题\外墙缺陷筛查\01_data\_v3_build

清理旧的 _v3_build ...
[源] greybrick-yolo               图  649  可用配对  649  无标签 0  空/全非法 0  非法框 0
[源] hrcds_yolo_exposed_rebar     图  464  可用配对  464  无标签 0  空/全非法 0  非法框 0
[源] rebar_structural             图   25  可用配对   25  无标签 0  空/全非法 0  非法框 0
[源] urban_yolo_rust_delam        图 1716  可用配对 1716  无标签 0  空/全非法 0  非法框 0

合并后可用配对总数 = 2854

L1 完全重复（MD5）：5 组，冗余 5 张
L1 后唯一内容 2849 张
L2 近似重复（dHash hamming<=3）：1 组，桶内比较 18865 次
最终保留 2848 张唯一图（共丢弃 6）

落盘：images 2848 个 / labels 2848 个

按源：{'greybrick-yolo': 649, 'hrcds_yolo_exposed_rebar': 462, 'rebar_structural': 25, 'urban_yolo_rust_delam': 1712}
按类（图数/框数）：
    crack              206 图 /    469 框
    spalling           159 图 /    313 框
    efflorescence      378 图 /   1152 框
    exposed_rebar      462 图 /   1389 框
    rust              1196 图 /  13685 框
    delamination       518 图 /    694 框
    moss               110 图 /    560 框

报告: D:\pythonstudy 备份\创新题\外墙缺陷筛查\01_data\audit\v3_build_report.json