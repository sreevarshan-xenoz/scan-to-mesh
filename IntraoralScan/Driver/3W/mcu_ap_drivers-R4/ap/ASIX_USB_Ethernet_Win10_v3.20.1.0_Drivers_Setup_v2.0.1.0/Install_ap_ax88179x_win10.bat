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
echo Close all working device

pnputil /enum-drivers>>%fileName%
for /f "tokens=1 delims=: " %%i in ('findstr /n "axusbeth.inf" %fileName%') do (
	set lineKeyword=%%i
	call :DriverCheck %fileName% %%i
	
	call rmDrv.bat %fileName%,%%i
)
del /s/q/f %fileName%

echo driver is installing , please wait for a few minutes ...
"%~dp0/PsExec64.exe" -i -s -realtime /accepteula pnputil -i -a "%~dp0/program files/ASIX\ASIX USB Ethernet Win10_Drivers/64-bit/AxUsbEth.inf"
"%~dp0/program files/ASIX\ASIX USB Ethernet Win10_Drivers/64-bit/dpinst.exe" /PATH "%~dp0/program files/ASIX\ASIX USB Ethernet Win10_Drivers/64-bit/" /F /Q /A
xdevcon.exe rescan
:: check driver install result
echo delete driver.txt
del /s/q/f %fileName%
echo pnputil /enum-drivers
pnputil /enum-drivers>>%fileName%

echo look for axusbeth.inf
set /a installFlag=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "axusbeth.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag = lineKeyword-2
)
echo check installFlag
IF not %installFlag% == 0 goto FINISHPOS
echo driver is retry installing , please wait for a few minutes ...
pnputil -i -a "%~dp0/program files/ASIX\ASIX USB Ethernet Win10_Drivers/64-bit/AxUsbEth.inf"
"%~dp0/program files/ASIX\ASIX USB Ethernet Win10_Drivers/64-bit/dpinst.exe" /PATH "%~dp0/program files/ASIX\ASIX USB Ethernet Win10_Drivers/64-bit/" /F /Q /A
:FINISHPOS
echo driver install end

exit /b

:DriverCheck
   echo get version line
   set /a skipLines = %2+3
   for /f "skip=%skipLines% delims=: tokens=2" %%k in (%1) do (
      echo get_ASIX_axusbeth.inf ver:%%k
	  ::pause
      goto EndPos
   )
:EndPos
goto :eof