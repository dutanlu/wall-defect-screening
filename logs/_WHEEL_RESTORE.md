==============================================================================
wheel -> site-packages 全量还原
==============================================================================
site-packages = D:\下载\Lib\site-packages
wheelhouse    = C:\Users\dutanlu\AppData\Local\pip\cache\_wheelhouse_lnk
extra wheels  = D:\pythonstudy 备份\创新题\外墙缺陷筛查\logs\_extra_whl
隔离区        = D:\pythonstudy 备份\创新题\外墙缺陷筛查\logs\_quarantine_20260921
DRY = True

==============================================================================
[0] 轮子挑选
==============================================================================
  numpy        现状文件=1212   选中=numpy-2.5.2-cp313-cp313-win_amd64.whl
               理由: mtime 最新 (2026-09-21 08:42:39)；候选 2.5.2(mtime=09-21 08:42), 2.5.3(mtime=09-07 13:45)
  networkx     现状文件=488    选中=networkx-3.6.1-py3-none-any.whl
               理由: mtime 最新 (2026-09-21 08:52:54)
  filelock     现状文件=0      选中=filelock-3.14.0-py3-none-any.whl
               理由: mtime 最新 (2026-09-20 21:43:05)
  colorama     现状文件=0      选中=colorama-0.4.6-py2.py3-none-any.whl
               理由: mtime 最新 (2026-09-21 09:13:58)
  cloudpickle  现状文件=-1     选中=cloudpickle-3.1.2-py3-none-any.whl
               理由: mtime 最新 (2026-09-21 09:13:58)

==============================================================================
[1] 执行还原
==============================================================================

------------------------------------------------------------------------------
还原 numpy  <-  numpy-2.5.2-cp313-cp313-win_amd64.whl
------------------------------------------------------------------------------
  wheel 内顶层条目: ['numpy', 'numpy-2.5.2.dist-info', 'numpy.libs']
  wheel 内文件数  : 945
  移走 D:\下载\Lib\site-packages\numpy      (1212 文件) -> numpy\numpy
  (无旧目录) D:\下载\Lib\site-packages\numpy-2.5.2.dist-info
  移走 D:\下载\Lib\site-packages\numpy.libs (2 文件) -> numpy\numpy.libs
  [dry] 跳过解包

------------------------------------------------------------------------------
还原 networkx  <-  networkx-3.6.1-py3-none-any.whl
------------------------------------------------------------------------------
  wheel 内顶层条目: ['networkx', 'networkx-3.6.1.dist-info']
  wheel 内文件数  : 601
  移走 D:\下载\Lib\site-packages\networkx   (488 文件) -> networkx\networkx
  移走 D:\下载\Lib\site-packages\networkx-3.6.1.dist-info (3 文件) -> networkx\networkx-3.6.1.dist-info
  [dry] 跳过解包

------------------------------------------------------------------------------
还原 filelock  <-  filelock-3.14.0-py3-none-any.whl
------------------------------------------------------------------------------
  wheel 内顶层条目: ['filelock', 'filelock-3.14.0.dist-info']
  wheel 内文件数  : 13
  移走 D:\下载\Lib\site-packages\filelock   (0 文件) -> filelock\filelock
  (无旧目录) D:\下载\Lib\site-packages\filelock-3.14.0.dist-info
  [dry] 跳过解包

------------------------------------------------------------------------------
还原 colorama  <-  colorama-0.4.6-py2.py3-none-any.whl
------------------------------------------------------------------------------
  wheel 内顶层条目: ['colorama', 'colorama-0.4.6.dist-info']
  wheel 内文件数  : 17
  移走 D:\下载\Lib\site-packages\colorama   (0 文件) -> colorama\colorama
  (无旧目录) D:\下载\Lib\site-packages\colorama-0.4.6.dist-info
  [dry] 跳过解包

------------------------------------------------------------------------------
还原 cloudpickle  <-  cloudpickle-3.1.2-py3-none-any.whl
------------------------------------------------------------------------------
  wheel 内顶层条目: ['cloudpickle', 'cloudpickle-3.1.2.dist-info']
  wheel 内文件数  : 7
  (无旧目录) D:\下载\Lib\site-packages\cloudpickle
  (无旧目录) D:\下载\Lib\site-packages\cloudpickle-3.1.2.dist-info
  [dry] 跳过解包

==============================================================================
[2] 文件级验证
==============================================================================
  !! cloudpickle  仍缺: ['__init__.py', 'cloudpickle.py']
  !! colorama     仍缺: ['__init__.py', 'ansi.py', 'initialise.py', 'win32.py', 'winterm.py']
  !! filelock     仍缺: ['__init__.py', '_api.py']
  !! networkx     仍缺: ['__init__.py', 'utils/__init__.py', 'utils/union_find.py', 'algorithms/__init__.py', 'generators/__init__.py']
  !! numpy        仍缺: ['__init__.py', '_typing/_array_like.py', 'version.py', '_core/_multiarray_umath.cp313-win_amd64.pyd', '_core/_multiarray_umath.pyd']
  dist-info: []

==============================================================================
[3] 导入冒烟（子进程，PYTHONPATH 已摘除）
==============================================================================
{
  "numpy": "FAIL ModuleNotFoundError: No module named 'numpy'",
  "cloudpickle": "FAIL ModuleNotFoundError: No module named 'cloudpickle'",
  "colorama": "FAIL ModuleNotFoundError: No module named 'colorama'",
  "filelock": "FAIL ModuleNotFoundError: No module named 'filelock'",
  "loguru": "FAIL ModuleNotFoundError: No module named 'colorama'",
  "natsort": "OK 8.4.0",
  "networkx": "FAIL ModuleNotFoundError: No module named 'networkx'",
  "numpy_smoke": "FAIL ModuleNotFoundError: No module named 'numpy'"
}


==============================================================================
结论
==============================================================================
  numpy          dry
  networkx       dry
  filelock       dry
  colorama       dry
  cloudpickle    dry
  文件级: 仍有缺失