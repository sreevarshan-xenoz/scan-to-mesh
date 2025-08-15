@echo off
::%1 mshta vbscript:CreateObject("Shell.Application").ShellExecute("cmd.exe","/c %~s0 ::","","runas",1)(window.close)&&exit
::color a
cd /d "%~dp0"
cd ../
start  "Uninstall SETUP_SHINING_3WMCU.EXE task" cmd /c "%cd%/mcu_ap_drivers-R4/mcu/Uninstall_mcu.bat"
timeout /nobreak /t 5
set num=10
for /l %%i in (1,1,%num%) do (
	if %%i geq 5 (
		tasklist | find /i "SETUP_SHINING_3WMCU.EXE" && (taskkill /im SETUP_SHINING_3WMCU.EXE /f && echo "run..." )
	)
	timeout /nobreak /t 1 > null
	tasklist | find /i "SETUP_SHINING_3WMCU.EXE"
	if errorlevel 1 (
		exit /b 

	) else (
		echo run...
	)
	echo 111 %num% %%i
)

echo intall end

