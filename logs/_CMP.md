================================================================================================
[1] V2（旧）划分报告
================================================================================================
  input = D:\pythonstudy 备份\创新题\外墙缺陷筛查\01_data\unified
  out   = D:\pythonstudy 备份\创新题\外墙缺陷筛查\01_data\dataset
  n_pairs = 674   leak_free = True
  train  n_images=473    n_instances=1734    
  val    n_images=134    n_instances=501     
  test   n_images=67     n_instances=259     
  --- V2 合计实例数 ---
      crack               469
      spalling            313
      efflorescence      1152
      exposed_rebar         0
      rust                  0
      delamination          0
      moss                560

================================================================================================
[2] V3（新）组装报告 01_data/audit/v3_build_report.json
================================================================================================
  per_source:
    greybrick-yolo               图=649    可用配对=649    无标签=0     空/全非法=0     非法框=0
    hrcds_yolo_exposed_rebar     图=464    可用配对=464    无标签=0     空/全非法=0     非法框=0
    rebar_structural             图=25     可用配对=25     无标签=0     空/全非法=0     非法框=0
    urban_yolo_rust_delam        图=1716   可用配对=1716   无标签=0     空/全非法=0     非法框=0
  n_pairs_in=2854  L1 组=5(去 5)  L2 组=1(去 1)  n_final=2848  丢弃合计=6  落盘失败=0
  images_by_source = {'greybrick-yolo': 649, 'hrcds_yolo_exposed_rebar': 462, 'rebar_structural': 25, 'urban_yolo_rust_delam': 1712}
  --- V3 类别（图数 / 框数）---
      crack              206 图 /    469 框
      spalling           159 图 /    313 框
      efflorescence      378 图 /   1152 框
      exposed_rebar      462 图 /   1389 框
      rust              1196 图 /  13685 框
      delamination       518 图 /    694 框
      moss               110 图 /    560 框

================================================================================================
[3] V2 -> V3 类别实例数对照（训练用 7 类）
================================================================================================
  类别                    V2图      V3图      V3框   变化
  crack                 469      206      469   +0
  spalling              313      159      313   +0
  efflorescence        1152      378     1152   +0
  exposed_rebar           0      462     1389   +1389
  rust                    0     1196    13685   +13685
  delamination            0      518      694   +694
  moss                  560      110      560   +0
  （注：V2 列为「实例数」，V3 图列/框列分别为图像数与实例数；两者口径见上表标题）

================================================================================================
[4] 照片收集站 08_photo_collector 现状
================================================================================================
  .                                               files=6         0.02 MB  09-20 18:39
        - _patch_promo_storage.txt
        - _patch_promo_storage2.txt
        - _patch_site_copy.txt
        - _patch_storage_private.txt
        - 发布话术.md
        - 推广文案.md
  site                                            files=4         0.06 MB  09-20 12:48
        - app.js
        - cloud.js
        - config.js
        - index.html
  _backup_before_private                          files=4         0.06 MB  09-20 18:39
        - cloud.js
        - index.html
        - 发布话术.md
        - 推广文案.md