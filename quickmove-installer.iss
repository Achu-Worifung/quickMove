; MyAppInstaller.iss
[Setup]
AppName=MyApp
AppVersion=1.0.0
DefaultDirName={pf}\MyApp            ; change to {userappdata}\MyApp for per-user install (no admin)
DefaultGroupName=MyApp
UninstallDisplayIcon={app}\main.exe
OutputDir=install_build
OutputBaseFilename=MyApp-Setup-1.0
Compression=lzma2
SolidCompression=yes
; If you want only x64 installers:
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; Copy everything inside the dist\main folder (recursively)
; Adjust "main" below if your dist folder's executable folder is named differently.
Source: "dist\main\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

; If you prefer to package just the single EXE (onefile build), use:
; Source: "dist\main.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\MyApp"; Filename: "{app}\main.exe"
Name: "{userdesktop}\MyApp"; Filename: "{app}\main.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
Filename: "{app}\main.exe"; Description: "Launch MyApp"; Flags: nowait postinstall skipifsilent

; Optional: run VC++ redistributable before installing your app (uncomment and edit)
; [Run]
; Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/quiet /norestart"; Flags: runhidden waituntilterminated

; Optional: copy redistributable to tmp and run (add the installer to [Files] above)
