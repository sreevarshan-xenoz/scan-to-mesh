@echo off
reg query HKLM\SYSTEM\CurrentControlSet\Control\Class\{78a1c341-4539-11d3-b88d-00c04fad5171} /s
IF %errorlevel%==1 echo NotExist-{78a1c341-4539-11d3-b88d-00c04fad5171}
IF %errorlevel%==0 %~dp0/psexec64.exe -d -i -s /accepteula reg delete HKLM\SYSTEM\CurrentControlSet\Control\Class\{78a1c341-4539-11d3-b88d-00c04fad5171} /f

reg query HKLM\SYSTEM\CurrentControlSet\Control\Class\{6A0F6C25-2C5E-4F49-A338-556699A7BA8E} /s
IF %errorlevel%==1 echo NotExist-{6A0F6C25-2C5E-4F49-A338-556699A7BA8E}
IF %errorlevel%==0 %~dp0/psexec64.exe -d -i -s /accepteula reg delete HKLM\SYSTEM\CurrentControlSet\Control\Class\{6A0F6C25-2C5E-4F49-A338-556699A7BA8E} /f

reg query HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Setup\PnpLockdownFiles\ /f "*/System32/drivers/SH3DU3DRIVER.sys" /reg:64
IF %errorlevel%==1 echo NotExist SH3DU3Driver.sys
IF %errorlevel%==0 %~dp0/psexec64.exe -i -s /accepteula %~dp0/changeSH3DU3DriverSysAccess.ini
%~dp0/psexec64.exe -d -i -s /accepteula reg delete HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Setup\PnpLockdownFiles\%SystemRoot%/System32/drivers/SH3DU3DRIVER.sys /f

reg query HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Setup\PnpLockdownFiles /f "*/System32/drivers/3dscannerdriver.sys" /reg:64
IF %errorlevel%==1 echo NotExist 3dscannerdriver.inf
IF %errorlevel%==0 %~dp0/psexec64.exe -i -s /accepteula %~dp0/change3DScannerDriverSysAccess.ini
%~dp0/psexec64.exe -d -i -s /accepteula reg delete HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Setup\PnpLockdownFiles\%SystemRoot%/System32/drivers/3dscannerdriver.sys /f