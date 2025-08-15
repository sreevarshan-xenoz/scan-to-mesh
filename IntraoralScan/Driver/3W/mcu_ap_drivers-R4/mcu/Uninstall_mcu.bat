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
if exist %fileName% del /s/q/f %fileName%
::
echo driver is delete method 1, please wait for a few minutes ...
pnputil /enum-drivers>>%fileName%
set /a installFlag=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "ch375wdm.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag = lineKeyword-2
	call rmDrv.bat %fileName%,%%i
)
del /s/q/f %fileName%
::
if exist %fileName% del /s/q/f %fileName%
pnputil /enum-drivers>>%fileName%
set /a installFlag=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "ch375wdm.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag = lineKeyword-2
)
IF not %installFlag% == 0 (
	set /a installFlag_error = 0
	"%~dp0/FILE/SETUP_SHINING_3WMCU.EXE" /S
)

if exist %fileName% del /s/q/f %fileName%
if exist mcu_install_result.txt del /s/q/f mcu_install_result.txt

::pause