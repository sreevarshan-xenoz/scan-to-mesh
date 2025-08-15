@echo off
::%1 mshta vbscript:CreateObject("Shell.Application").ShellExecute("cmd.exe","/c %~s0 ::","","runas",1)(window.close)&&exit
setlocal enabledelayedexpansion
cd /d "%~dp0"

set fileName=driver.txt
echo Check has installed drivers
del /s/q/f %fileName%
if exist ap_ver_temp.txt del /s/q/f ap_ver_temp.txt

echo check_cardle_driver_info start
::--------------------------------------------------------------------------------------------
pnputil /enum-drivers>%fileName%
echo look for CH375WDM.inf
set /a installFlag375=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "ch375wdm.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlag375 = lineKeyword-2
	echo ch375wdm_flag:%installFlag375% %%i
)
::--------------------------------------------------------------------------------------------
echo look for old 88179 driver
set /a installFlagOld88179=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "netax88179x_178a_772d.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlagOld88179 = lineKeyword-2
	echo old_88179_flag:%installFlagOld88179% %%i
)
echo check old 88179 installFlag
IF %installFlagOld88179% == 0 ( 
	echo not_find_driver old_88179_flag:%installFlagOld88179% 
) ELSE (
	echo find_driver old_88179_flag:%installFlagOld88179% 
)
::--------------------------------------------------------------------------------------------
echo look for New 88179 driver
set /a installFlagNew88179=0
for /f "tokens=1 delims=: " %%i in ('findstr /n "axusbeth.inf" %fileName%') do (
	set lineKeyword=%%i
	set /a installFlagNew88179 = lineKeyword-2
	call :DriverCheck %fileName% %%i
	echo New_88179_flag:%installFlagNew88179% %%i
)
if exist ap_ver_temp.txt (
	echo start compare data version ...
	for /f %%t in (ap_ver_temp.txt) do (
		::echo %%k
		set getvertempstr=%%t
		echo getvertempstr !getvertempstr!
		set /a data1=!getvertempstr:~0,2!
		set /a data2=!getvertempstr:~3,2!
		set /a data3=!getvertempstr:~6,4!
		echo data1 !data1! data2 !data2! data3 !data3!
		::data1 01~12  data2 00~31 data3 2000~2100			old drivers 05/17/2022
		if !data1! geq 0 (
			if !data1! leq 12 (
				if !data2! geq 0 (
					if !data2! leq 31 (
						if !data3! geq 2000 (
							if !data3! leq 2100 (
								echo driver data ok,start compare...
								if !data3! gtr 2022 (
									echo new_ax99179 ok 1 ...
								) else (
									if !data3! equ 2022 (
										if !data1! gtr 5 (
											echo new_ax99179 ok 2 ...
										) else (
											if !data1! equ 5 (
												if !data2! gtr 17 (
													echo new_ax99179 ok 3 ...
												) else (
													echo new_ax99179 bad 3 ...
													set /a installFlagNew88179=0
												)
											) else (
												echo new_ax99179 bad 2 ...
												set /a installFlagNew88179=0
											)
										) 
									) else (
										echo new_ax99179 bad 1 ...
										set /a installFlagNew88179=0
										echo 
									)
								)
							)
						)
					)
				)
			)
		) 
	)
	if exist ap_ver_temp.txt del /s/q/f ap_ver_temp.txt
	echo start compare data end, installFlagNew88179:%installFlagNew88179% 
)

echo check New 88179 installFlag
IF %installFlagNew88179% == 0 ( 
	echo not_find_driver New_88179_flag:%installFlagNew88179% 
) ELSE (
	echo find_driver New_88179_flag:%installFlagNew88179% 
)

set /a NORMAL_375_AND_88179=0
set /a ERROR_375=2
set /a ERROR_88179=4
set /a ERROR_375_AND_88179=6
if exist %fileName% del /s/q/f %fileName%
if exist check_driver_return_code.txt del /s/q/f check_driver_return_code.txt
::install check
echo install check
IF %installFlag375% == 0 ( 
::echo not_find_driver ch375wdm_flag:%installFlag375% 	
	IF %installFlagOld88179% == 0 ( 
		IF %installFlagNew88179% == 0 ( 
			echo not_find_driver New_88179 and Old_88179 and ch365
			echo DRIVER_INFO_RETURN %ERROR_375_AND_88179% > check_driver_return_code.txt
			exit /b %ERROR_375_AND_88179%
		) ELSE (
			::echo find_driver New_88179_flag:%installFlagNew88179% 
			echo find_driver New_88179 ,But no found ch365
			echo DRIVER_INFO_RETURN %ERROR_375% > check_driver_return_code.txt
			exit /b %ERROR_375%
		)
	) ELSE (
		echo find_driver old_88179 ,But no found ch365
		echo DRIVER_INFO_RETURN %ERROR_375% > check_driver_return_code.txt
		exit /b %ERROR_375%
	)
	echo 111-1
) ELSE (
::echo find_driver ch375wdm_flag:%installFlag375% 
::echo DRIVER_INFO_RETURN 0>check_driver_return_code.txt
	IF %installFlagOld88179% == 0 ( 
		IF %installFlagNew88179% == 0 ( 
			echo find_driver ch375wdm_flag,But no found New_88179 and Old_88179
			echo DRIVER_INFO_RETURN %ERROR_88179% > check_driver_return_code.txt
			exit /b %ERROR_88179%
		) ELSE (
			::echo find_driver New_88179_flag:%installFlagNew88179% 
			echo find_driver New_88179 AND ch365
			echo DRIVER_INFO_RETURN %NORMAL_375_AND_88179% > check_driver_return_code.txt
			exit /b %NORMAL_375_AND_88179%
		)
	) ELSE (
		echo find_driver old_88179 AND ch365
		echo DRIVER_INFO_RETURN %NORMAL_375_AND_88179% > check_driver_return_code.txt
		exit /b %NORMAL_375_AND_88179%
	)
	echo 111-2
)

::--------------------------------------------------------------------------------------------
:DriverCheck
   echo get version line
   set /a skipLines = %2+3
   for /f "skip=%skipLines% delims=: tokens=2" %%k in (%1) do (
      ::echo %%k
      echo mcu_install_ver:%%k
      if exist ap_ver_temp.txt del /s/q/f ap_ver_temp.txt
      echo %%k > ap_ver_temp.txt
      goto EndPos
   )
:EndPos
goto :eof