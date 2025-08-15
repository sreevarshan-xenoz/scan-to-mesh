cd /d %~dp0

set /a lineSkip=%2-2
for /f "skip=%lineSkip% delims=: tokens=2" %%i in (%1) do (
	pnputil /uninstall -d %%i /force
	goto :eof
)


::powershell -Command "& {pnputil /uninstall -d %%i /force}"