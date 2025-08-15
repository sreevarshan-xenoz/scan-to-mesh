Name "MineClient"
RequestExecutionLevel admin
!define MineClient_NAME "QuickInstaller"
!define MineClient_EXE "MineClient.exe"
!define PRODUCT_NAME "IntraoralScan"
!define PRODUCT_VERSION "${Version}"
!define PRODUCT_PUBLISHER "Shining3D"
!define TEMPRORY_PATH "$APPDATA\Local\Temp\Shining3D\IntraoralScanMineClient"
!define PACKAGE_NAME "${MINECLIENT_NAME}_${PRODUCT_NAME}${PRODUCT_VERSION}.exe"

Icon "installer.ico"
; The file to write
OutFile "${PACKAGE_NAME}"
Unicode True
InstallDir "${TEMPRORY_PATH}"
 
;--------------------------------
VIProductVersion ${PRODUCT_VERSION}
VIAddVersionKey   "ProductName" ${MineClient_NAME}
VIAddVersionKey   "CompanyName" "SHINING3D"
VIAddVersionKey   "LegalCopyright" "SHINING3D"
VIAddVersionKey   "FileDescription" ${MineClient_NAME}
VIAddVersionKey   "FileVersion" ${PRODUCT_VERSION}
VIAddVersionKey   "ProductVersion" ${PRODUCT_VERSION}

;--------------------------------
Function .onInit 
	SetSilent silent
FunctionEnd
; The stuff to install
Section ""

  ; Set output path to the installation directory.
  SetOutPath $INSTDIR
  ; write package path
  WriteRegStr HKCU "SOFTWARE\\Shining3d\\Sn3DInstaller" "path" "$EXEDIR/$EXEFILE"
  ; Put file there
  File /r /x  "HUB\hub.conf"  src\*
  ; run mineclien 
  ExecWait '"${TEMPRORY_PATH}\${MineClient_EXE}"'
  
SectionEnd