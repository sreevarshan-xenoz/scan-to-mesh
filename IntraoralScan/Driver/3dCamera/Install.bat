@echo off
cd /d %~dp0
%1 mshta vbscript:CreateObject("Shell.Application").ShellExecute("cmd.exe","/c %~s0 ::","","runas",1)(window.close)&&exit
::setlocal enabledelayedexpansion

set fileName= driver.txt
echo Check has installed drivers
del /s/q/f %fileName%
del /s/q/f result.version.latest.kk
del /s/q/f result.version.earlier.kk
del /s/q/f result.version.same.kk
echo Close all working device
%~dp0/SHCtl.exe STOP -1
pnputil /enum-drivers>>%fileName%
for /f "tokens=1 delims=: " %%i in ('findstr /n "sh3du3driver.inf" %fileName%') do (
	set lineKeyword=%%i
	call :DriverCheck %fileName% %%i
	if exist result.version.latest.kk call rmDrv.bat %fileName%,%%i
)
del /s/q/f %fileName%
pnputil /enum-drivers>>%fileName%
for /f "tokens=1 delims=: " %%i in ('findstr /n "3dscannerdriver.inf" %fileName%') do (
	set lineKeyword=%%i
	call rmDrv.bat %fileName%,%%i
)

if exist result.version.same.kk exit /b
if exist result.version.earlier.kk exit /b

call delRegNode.bat
echo driver is installing , please wait for a few minutes ...
%~dp0/PsExec64.exe -i -s -realtime /accepteula pnputil -i -a %~dp0/SH3DU3DRIVER.inf
xdevcon.exe rescan
:: check driver install result
echo delete driver.txt
del /s/q/f %fileName%
echo pnputil /enum-drivers
pnputil /enum-drivers>>%fileName%
echo look for sh3du3driver.inf
set /a installFlag=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "sh3du3driver.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag = lineKeyword-2
)
echo check installFlag
IF not %installFlag% == 0 goto FINISHPOS
echo driver is retry installing , please wait for a few minutes ...
pnputil -i -a %~dp0/SH3DU3DRIVER.inf
:FINISHPOS
echo driver install end
exit /b

:DriverCheck
   echo get version line
   set /a skipLines = %2+3
   for /f "skip=%skipLines% delims=: tokens=2" %%k in (%1) do (
      ::echo %%k
      %~dp0/SHCtl.exe DRIVERCHECK 02/08/2023 %%k
      goto EndPos
   )
:EndPos
goto :eof