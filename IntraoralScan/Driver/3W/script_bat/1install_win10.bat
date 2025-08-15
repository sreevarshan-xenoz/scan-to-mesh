
@echo off
::%1 mshta vbscript:CreateObject("Shell.Application").ShellExecute("cmd.exe","/c %~s0 ::","","runas",1)(window.close)&&exit
::color a
cd /d "%~dp0"
cd ../
start  "SETUP_SHINING_3WMCU.EXE task" cmd /c "%cd%/mcu_ap_drivers-R4/mcu/Install_mcu.bat"
timeout /nobreak /t 5
set num=10
for /l %%i in (1,1,%num%) do (
	if %%i geq 5 (
		tasklist | find /i "SETUP_SHINING_3WMCU.EXE" && (taskkill /im SETUP_SHINING_3WMCU.EXE /f && echo "run..." )
	)
	timeout /nobreak /t 1 > null
	tasklist | find /i "SETUP_SHINING_3WMCU.EXE"
	if errorlevel 1 (
		::exit /b 
		goto :install_ax88179
	) else (
		echo run...
	)
	echo 111 %num% %%i
)

:install_ax88179
echo install new ax99178 drivers
cd /d "%~dp0"
cd ../
start "AX88179 task" cmd /c "%cd%/mcu_ap_drivers-R4/ap/ASIX_USB_Ethernet_Win10_v3.20.1.0_Drivers_Setup_v2.0.1.0/Install_ap_ax88179x_win10.bat"

echo intall end

