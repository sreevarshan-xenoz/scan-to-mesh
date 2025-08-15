
@echo off
::%1 mshta vbscript:CreateObject("Shell.Application").ShellExecute("cmd.exe","/c %~s0 ::","","runas",1)(window.close)&&exit
::color a
echo install new ax99178 drivers
cd /d "%~dp0"
cd ../
start "AX88179 task" cmd /c "%cd%/mcu_ap_drivers-R4/ap/remove_ap_all_driver/Uninstall_ap_ax88179x_all.bat"

echo intall end

