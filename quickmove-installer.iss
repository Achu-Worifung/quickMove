#define MyAppName "QuickMove"
#define MyAppVersion "1.0"
#define MyAppPublisher "YourName"
#define AppExe "run.bat"

[Setup]
AppName=hello
AppVersion=1
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=QuickMoveInstaller
Compression=lzma2
SolidCompression=yes

[Files]
Source: "release\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#AppExe}"

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
