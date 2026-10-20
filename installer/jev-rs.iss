; ============================================================================
;  Jev-RS — Windows installer (Inno Setup 6)
;
;  Build:
;    iscc /DAppVersion=0.3.0 installer\jev-rs.iss
;
;  The compiled installer (dist\jev-rs-setup-<version>.exe) is published in
;  the GitHub release. It installs the engine, offers to download the
;  decision model (~2.3 GB) and creates shortcuts.
; ============================================================================

#define MyAppName "Jev-RS"
#define MyAppPublisher "alnyx-dev"
#define MyAppURL "https://github.com/alnyx-dev/jev-rs"
#define MyAppExeName "jev-rs.exe"

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif

[Setup]
AppId={{6E4C1A57-9C3E-4E5B-9F41-6C0A2D8B7E13}
AppName={#MyAppName}
AppVersion={#AppVersion}
AppVerName={#MyAppName} {#AppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={localappdata}\Programs\Jev-RS
DefaultGroupName=Jev-RS
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupIconFile=icon.ico
OutputDir=..\dist
OutputBaseFilename=jev-rs-setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
WizardSizePercent=110
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
DisableDirPage=auto
CloseApplications=yes
RestartApplications=no
UninstallDisplaySize=24576
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} installer
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#AppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "downloadmodel"; Description: "Download the decision model now (~2.3 GB, required for inference)"
Name: "launchapp"; Description: "Start Jev-RS after the installation finishes"

[Files]
Source: "..\target\release\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\scripts\setup_model.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "download-model.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "run-jev-rs.bat"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\Model downloader"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\download-model.ps1"" -DestDir ""{app}\model_onnx"""; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\download-model.ps1"" -DestDir ""{app}\model_onnx"""; WorkingDir: "{app}"; Flags: nowait; Tasks: downloadmodel; Description: "Download the decision model"
Filename: "{app}\run-jev-rs.bat"; WorkingDir: "{app}"; Flags: nowait; Tasks: launchapp; Description: "Start {#MyAppName}"

[Code]
const
  VCRedistUrl = 'https://aka.ms/vs/17/release/vc_redist.x64.exe';

function HasVCRuntime: Boolean;
begin
  Result := FileExists(ExpandConstant('{sys}\vcruntime140.dll')) and
            FileExists(ExpandConstant('{sys}\vcruntime140_1.dll'));
end;

function InitializeSetup: Boolean;
var
  Answer: Integer;
  ErrorCode: Integer;
begin
  Result := True;
  if not HasVCRuntime then
  begin
    Answer := MsgBox(
      'The Microsoft Visual C++ runtime was not found on this system.' + #13#10 + #13#10 +
      'Jev-RS needs it to start. Open the Microsoft download page now?' + #13#10 + #13#10 +
      '(You can also continue the installation and install the runtime later.)',
      mbConfirmation, MB_YESNOCANCEL);
    if Answer = IDYES then
      Exec('open', VCRedistUrl, '', SW_SHOW, ewNoWait, ErrorCode);
    if Answer = IDCANCEL then
      Result := False;
  end;
end;

function FinishPageText: String;
begin
  Result := 'The engine is installed.' + #13#10 + #13#10;
  if WizardIsTaskSelected('downloadmodel') then
    Result := Result +
      'A separate window is downloading the decision model (~2.3 GB).' + #13#10 +
      'The engine can be started right away - it waits for the model to finish.' + #13#10 + #13#10
  else
    Result := Result +
      'The decision model was not downloaded. Run "Model downloader" from the' + #13#10 +
      'Start menu (or run-jev-rs.bat) before using the engine.' + #13#10 + #13#10;
  Result := Result + 'Start Jev-RS with "run-jev-rs.bat" or the shortcut.';
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Answer: Integer;
  ModelDir: String;
begin
  if CurUninstallStep = usUninstall then
  begin
    ModelDir := ExpandConstant('{app}\model_onnx');
    if DirExists(ModelDir) then
    begin
      Answer := MsgBox(
        'Also delete the downloaded decision model (~2.3 GB)?' + #13#10 + #13#10 +
        'Keeping it avoids a 2.3 GB download if you reinstall Jev-RS.',
        mbConfirmation, MB_YESNO);
      if Answer = IDYES then
        DelTree(ModelDir, True, True, True);
    end;
  end;
end;
