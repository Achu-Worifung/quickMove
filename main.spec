# -*- mode: python ; coding: utf-8 -*-
import os

# Define Tesseract paths
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR'
TESSDATA_PATH = os.path.join(TESSERACT_PATH, 'tessdata')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[
    # Include Tesseract executable and required DLLs
    (r'C:\Program Files\Tesseract-OCR\tesseract.exe', '.'),  # Tesseract executable
    (r'C:\Program Files\Tesseract-OCR\libtesseract-5.dll', '.'),  # Tesseract library
    (r'C:\Program Files\Tesseract-OCR\leptonica-1.82.0.dll', '.'),  # Leptonica library
    (r'C:\Program Files\Tesseract-OCR\zlib1.dll', '.'),  # Zlib
    (r'C:\Program Files\Tesseract-OCR\libwebp-7.dll', '.'),  # WebP
    (r'C:\Program Files\Tesseract-OCR\openjp2.dll', '.'),  # JPEG2000


],

    datas=[
        # Your existing data files
        ('ui/*.ui', 'ui/'),
        ('images/*.png', 'images/'),
        ('logo.ico', '.'),
        # Add Tesseract training data
        (os.path.join(TESSDATA_PATH, 'eng.traineddata'), 'tessdata/'),
        # Add any other language files you need, e.g.:
        # (os.path.join(TESSDATA_PATH, 'deu.traineddata'), 'tessdata/'),
    ],
    hiddenimports=[
        'pytesseract'
    ],
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
    console=True,  # Set to False to hide console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='logo.ico'
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