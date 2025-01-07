# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[('python312.dll', '.')],
    datas=[('ui/*.ui', 'ui/'), ('images/*.png', 'images/'), ('logo.ico', '.'),('python312.dll', '.')],
    hiddenimports=['PyQt5.sip', 'win32api', 'pynput.keyboard', 'pynput.mouse', 
                   'pyautogui', 'cv2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='QuickMove',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, #do not show console 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon= 'logo.ico'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
