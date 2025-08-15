@echo on
cd /d "%~dp0"

del /s/q/f "%~dp0/info_all.txt"
echo Check has installed drivers
pnputil /enum-drivers >> "%~dp0/info_all.txt"
