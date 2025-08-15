@echo on
::%1 mshta vbscript:CreateObject("Shell.Application").ShellExecute("cmd.exe","/c %~s0 ::","","runas",1)(window.close)&&exit
cd /d "%~dp0"
echo 1111%~dp0
call "%~dp0/check_cardle_driver_info.bat"
echo errorlevel %errorlevel%

pause