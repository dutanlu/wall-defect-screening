@echo off
chcp 936 >nul
title 外墙缺陷智能筛查系统 - 本地演示
cd /d "%~dp0"

echo ============================================================
echo   外墙缺陷智能筛查系统  -  本地演示
echo   （提交的「实机运行视频」用的就是这个程序）
echo ============================================================
echo.

set "PY=D:\下载\python.exe"
if not exist "%PY%" (
  echo [错误] 未找到 Python 解释器：
  echo        %PY%
  echo.
  echo        本演示依赖该解释器里已安装的 torch / ultralytics / gradio。
  echo        若它不存在或改名，请修改本文件里的 PY 变量。
  pause
  exit /b 1
)

echo 正在启动本地服务（首次启动需加载模型，约 5-15 秒）...
echo.
echo   服务地址：http://127.0.0.1:7860
echo   浏览器将在 10 秒后自动打开；没打开就手动访问上面的地址。
echo.
echo   【停止服务】关闭本窗口，或按 Ctrl+C
echo ============================================================
echo.

REM 另起一个最小化窗口：延时约 10 秒后打开默认浏览器
start "" /min cmd /c "ping -n 11 127.0.0.1 >nul & start http://127.0.0.1:7860"

REM 前台运行（窗口保留，日志可见）
"%PY%" app.py

echo.
echo 服务已停止。
pause
