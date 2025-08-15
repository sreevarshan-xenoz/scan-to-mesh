set regPath=HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\CI\Policy
set keyName=VerifiedAndReputablePolicyState
set valueData=0

reg add "%regPath%" /v "%keyName%" /t REG_DWORD /d "%valueData%" /f