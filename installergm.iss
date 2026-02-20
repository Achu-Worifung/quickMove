; Script generated for Quickmove - AI Bible Search & Transcription
#define MyAppName "Quickmove"
#define MyAppVersion "1.0"
#define MyAppPublisher "RHands, Inc."
#define MyAppURL "https://www.example.com/"
#define MyAppExeName "QuickMove.exe"

[Setup]
; NOTE: The AppId uniquely identifies this application.
AppId={{6860CC6A-7ADF-4973-8633-090D21A434D7}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
; Where the installer will be saved
OutputDir=D:\
OutputBaseFilename=Quickmove_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; 1. The main application executable
Source: "C:\Users\achuw\OneDrive\Desktop\quick hsp\dist\QuickMove\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

; 2. The entire _internal folder (This handles all DLLs, Python libs, and models inside it)
Source: "C:\Users\achuw\OneDrive\Desktop\quick hsp\dist\QuickMove\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

; 3. Explicitly include UI folder if it's NOT inside _internal (Safety Check)
; If your UI files are already in _internal, you can comment this out
Source: "C:\Users\achuw\OneDrive\Desktop\quick hsp\ui\*"; DestDir: "{app}\ui"; Flags: ignoreversion recursesubdirs

[Icons]
; Added WorkingDir: "{app}" to ensure the app finds its local files on launch
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; WorkingDir: "{app}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent