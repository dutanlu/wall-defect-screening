rc.txt = rc=1

--- out (utf16) ---
﻿python.exe : Traceback (most recent call last):
所在位置 行:447 字符: 1
+ & $py "$L\_v3_vs_v2_split.py" *> "$L\_v3_vs_v2_split.out.txt"
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (Traceback (most recent call last) 
   ::String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 
  File "D:\pythonstudy 澶囦唤\鍒涙柊棰榎澶栧缂洪櫡绛涙煡\logs\_v3_vs_v2_split.py", line 139, i
n <module>
    L.append("> 鉂?**鏈户鎵?*锛歏2 test 鐨?%d 寮犲浘閲岋紝鍙湁 %d 寮犵暀鍦?V3 test锛?
             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
             "%d 寮犺閲嶅垎鍒?val/train锛?d 寮犲凡涓嶅湪 V3銆?
             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
             % (len(v2m), rows2[0][0], rows2[1][0] + rows2[2][0], gone2))
             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TypeError: %d format: a real number is required, not str

--- md mtime=12:33:42 size=2379
