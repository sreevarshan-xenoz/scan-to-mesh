@echo off
cd /d "%~dp0"
net session >NULL 2>&1
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
if exist mcu_install_result.txt del /s/q/f mcu_install_result.txt

echo Close all working device

set /a installFlag_error=0

pnputil /enum-drivers>>%fileName%
for /f "tokens=1 delims=: " %%i in ('findstr /n "ch375wdm.inf" %fileName%') do (
	set lineKeyword=%%i
	call :DriverCheck %fileName% %%i
	call rmDrv.bat %fileName%,%%i
)
del /s/q/f %fileName%

::------------------------------------------------------------------------------------
echo driver is installing method 1, please wait for a few minutes ...

"%~dp0/PsExec64.exe" -i -s -realtime /accepteula pnputil -i -a "%~dp0/FILE/CH375WDM.inf"
if exist %fileName% del /s/q/f %fileName%
pnputil /enum-drivers>>%fileName%
set /a installFlag=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "ch375wdm.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag = lineKeyword-2
)
IF not %installFlag% == 0 (
	set /a installFlag_error = 0
	goto FINISHPOS
) ELSE (
	echo check installing method 1 faild!
	set /a installFlag_error = 1
)
::------------------------------------------------------------------------------------
echo driver is installing method 2, please wait for a few minutes ...

"%~dp0/FILE/SETUP_SHINING_3WMCU.EXE" /S
if exist %fileName% del /s/q/f %fileName%
pnputil /enum-drivers>>%fileName%
set /a installFlag=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "ch375wdm.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag = lineKeyword-2
)
IF not %installFlag% == 0 (
	set /a installFlag_error = 0
	goto FINISHPOS
) ELSE (
	echo check installing method 2 faild!
	set /a installFlag_error = 1
)

::------------------------------------------------------------------------------------
echo driver is installing method 3, please wait for a few minutes ...

"%~dp0/FILE/dpinst.exe" /PATH "%~dp0/FILE/" /F /Q /A
if exist %fileName% del /s/q/f %fileName%
pnputil /enum-drivers>>%fileName%
set /a installFlag=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "ch375wdm.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag = lineKeyword-2
)
IF not %installFlag% == 0 (
	set /a installFlag_error = 0
	goto FINISHPOS
) ELSE (
	echo check installing method 3 faild!
	set /a installFlag_error = 1
)

::------------------------------------------------------------------------------------
xdevcon.exe rescan
:: check driver install result
echo delete driver.txt
del /s/q/f %fileName%
echo pnputil /enum-drivers
pnputil /enum-drivers>>%fileName%
echo look for CH375WDM.inf
set /a installFlag=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "ch375wdm.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag = lineKeyword-2
)
echo check installFlag
IF not %installFlag% == 0 (
	set /a installFlag_error = 0
	goto FINISHPOS
) ELSE (
	echo check installing method 4 faild!
	set /a installFlag_error = 1
)

::------------------------------------------------------------------------------------
echo driver is retry installing method 4, please wait for a few minutes ...

pnputil -i -a "%~dp0/FILE/CH375WDM.inf"
if exist %fileName% del /s/q/f %fileName%
pnputil /enum-drivers>>%fileName%
set /a installFlag=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "ch375wdm.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag = lineKeyword-2
)
IF not %installFlag% == 0 (
	set /a installFlag_error = 0
	goto FINISHPOS
) ELSE (
	echo check installing method 4 faild!
	set /a installFlag_error = 1
)

::------------------------------------------------------------------------------------
echo driver is installing method 3, please wait for a few minutes ...

"%~dp0/FILE/dpinst.exe" /PATH "%~dp0/FILE/" /F /Q /A
if exist %fileName% del /s/q/f %fileName%
pnputil /enum-drivers>>%fileName%
set /a installFlag=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "ch375wdm.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag = lineKeyword-2
)
IF not %installFlag% == 0 (
	set /a installFlag_error = 0
	goto FINISHPOS
) ELSE (
	echo check installing method 3 faild!
	set /a installFlag_error = 1
)

:FINISHPOS
if exist mcu_install_result.txt del /s/q/f mcu_install_result.txt
echo driver install end,error_code: %installFlag_error%
pnputil /enum-drivers>>%fileName%
for /f "tokens=1 delims=: " %%i in ('findstr /n "ch375wdm.inf" %fileName%') do (
	set lineKeyword=%%i
	call :DriverCheck %fileName% %%i
)
if exist %fileName% del /s/q/f %fileName%
echo error_code: %installFlag_error% >> mcu_install_result.txt
exit /b %installFlag_error%

:DriverCheck
   echo get version line
   set /a skipLines = %2+3
   for /f "skip=%skipLines% delims=: tokens=2" %%k in (%1) do (
      ::echo %%k
      echo mcu_install_ver:%%k
      if exist mcu_install_result.txt del /s/q/f mcu_install_result.txt
      echo mcu_install_ver:%%k > mcu_install_result.txt
      goto EndPos
   )
:EndPos
goto :eof