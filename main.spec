# -*- mode: python ; coding: utf-8 -*-
from glob import glob
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# --- 1. Collect heavy ML packages ---
for pkg in ['transformers', 'optimum', 'onnxruntime', 'sentence_transformers', 
            'huggingface_hub', 'tokenizers', 'tiktoken', 'torch', 'torchaudio']:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# --- 2. Tesseract (for pytesseract) ---
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR"
if os.path.exists(TESSERACT_PATH):
    TESSDATA_PATH = os.path.join(TESSERACT_PATH, "tessdata")
    binaries += [
        (os.path.join(TESSERACT_PATH, "tesseract.exe"), "."),
        (os.path.join(TESSERACT_PATH, "libtesseract-5.dll"), "."),
        (os.path.join(TESSERACT_PATH, "leptonica-1.82.0.dll"), "."),
    ]
    # include ALL tessdata files, not just eng
    for f in glob(os.path.join(TESSDATA_PATH, "*.traineddata")):
        datas.append((f, "tessdata"))
else:
    print("WARNING: Tesseract not found at", TESSERACT_PATH)

# --- 3. Project data files ---
# Use Tree() for folders - it handles wildcards correctly
datas += [
    ("settings.ini", "."),
    ("logo.ico", "."),
    (".env", "."),  # <-- your env file
]

# Add UI files
if os.path.isdir("ui"):
    datas.append(("ui", "ui"))

# Add Icons
if os.path.isdir("Icons"):
    datas.append(("Icons", "Icons"))

# Local model/data folders
for folder in ["classifier", "search", "bibles", "artifacts", "models"]:
    if os.path.exists(folder):
        datas.append((folder, folder))  # simpler than glob loop



# --- 4. Hidden imports ---
hiddenimports += [
    'widgets.Settings', 'widgets.Create', 'widgets.MainPage',
    'widgets.SearchArea', 'widgets.SearchWidget',
    'util.transcription_class', 'util.biblesearch', 'util.classifier',
    'util.event_tracker', 'util.Message', 'util.util',
    'resources_rc',
    'optimum.onnxruntime', 'optimum.onnxruntime.modeling_ort',
    'pytesseract', 'dotenv',
] + collect_submodules('PyQt5.uic')

a = Analysis(
    ['main.py'],
    pathex=[os.path.abspath('.')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],  # we add one below
    excludes=['torch.cuda', 'tensorflow', 'jax', 'tensorrt'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='QuickMove',
    debug=False,
    strip=False,
    upx=False,
    console=True,  # set to False for release
    icon='logo.ico',
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False,
    name='QuickMove',
)