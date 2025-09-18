@echo off
REM 启动 UE4.27 项目（Windows 示例）
REM 请将下面路径替换为实际 UE4 编辑器路径与项目路径
set UE4_EDITOR="C:\Program Files\Epic Games\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe"
set PROJECT_PATH="%~dp0\ue_project\DisasterSwarm.uproject"
echo Launching UE4 project...
%UE4_EDITOR% %PROJECT_PATH%
pause
