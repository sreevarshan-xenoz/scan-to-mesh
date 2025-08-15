@echo off
cd /d %~dp0
%1 mshta vbscript:CreateObject("Shell.Application").ShellExecute("cmd.exe","/c %~s0 ::","","runas",1)(window.close)&&exit
::setlocal enabledelayedexpansion

set fileName= driver.txt
echo Check has installed drivers
del /s/q/f %fileName%
%~dp0/SHCtl.exe STOP -1
pnputil /enum-drivers>>%fileName%
set /a installFlag=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "sh3du3driver.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag = lineKeyword-2
	call rmDrv.bat %fileName%,%%i
)
del /s/q/f %fileName%
pnputil /enum-drivers>>%fileName%
for /f "tokens=1 delims=: " %%i in ('findstr /n "3dscannerdriver.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag = lineKeyword-2
	call rmDrv.bat %fileName%,%%i
)
del /s/q/f %fileName%
if %installFlag% NEQ 0 call delRegNode.bat
