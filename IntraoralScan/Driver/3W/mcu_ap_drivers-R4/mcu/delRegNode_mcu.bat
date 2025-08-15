@echo off
reg query HKLM\SYSTEM\CurrentControlSet\Control\Class\{77989ADF-06DB-4025-92E8-40D902C03B0A} /s
IF %errorlevel%==1 echo NotExist-{77989ADF-06DB-4025-92E8-40D902C03B0A}
IF %errorlevel%==0 "%~dp0/psexec64.exe" -d -i -s /accepteula reg delete HKLM\SYSTEM\CurrentControlSet\Control\Class\{77989ADF-06DB-4025-92E8-40D902C03B0A} /f

reg query HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Setup\PnpLockdownFiles /f "*/System32/drivers/ch375wdm.sys" /reg:64
IF %errorlevel%==1 echo NotExist ch375wdm.inf
IF %errorlevel%==0 "%~dp0/psexec64.exe" -i -s /accepteula "%~dp0/mcuDriverSysAccess.ini"
"%~dp0/psexec64.exe" -d -i -s /accepteula reg delete HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Setup\PnpLockdownFiles\%SystemRoot%/System32/drivers/ch375wdm.sys /f