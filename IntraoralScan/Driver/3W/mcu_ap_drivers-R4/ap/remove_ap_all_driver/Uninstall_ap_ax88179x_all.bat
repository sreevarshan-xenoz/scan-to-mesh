@echo off
cd /d "%~dp0"
net session >nul 2>&1
if %errorLevel% == 0 (
	echo Success: Administrative permissions confirmed.
) else (
	echo Failure: Current permissions inadequate.
	::%1 mshta vbscript:CreateObject("Shell.Application").ShellExecute("cmd.exe","/c %~s0 ::","","runas",1)(window.close)&&exit
	::setlocal enabledelayedexpansion
)

set fileName=driver.txt
echo Check has installed drivers
del /s/q/f %fileName%

pnputil /enum-drivers>>%fileName%

set /a installFlag=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "netax88179x_178a_772d.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag = lineKeyword-2
	call rmDrv.bat %fileName%,%%i
)
del /s/q/f %fileName%
pnputil /enum-drivers>>%fileName%

for /f "tokens=1 delims=: " %%i in ('findstr /n "axusbeth.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag = lineKeyword-2
	call rmDrv.bat %fileName%,%%i
)
del /s/q/f %fileName%
