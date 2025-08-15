
@echo off
::%1 mshta vbscript:CreateObject("Shell.Application").ShellExecute("cmd.exe","/c %~s0 ::","","runas",1)(window.close)&&exit
::color a
echo install new ax99178 drivers
cd /d "%~dp0"
cd ../
start "AX88179 task" cmd /c "%cd%/mcu_ap_drivers-R4/ap/ASIX_USB_Ethernet_Win10_v3.20.1.0_Drivers_Setup_v2.0.1.0/Install_ap_ax88179x_win10.bat"

echo intall end

