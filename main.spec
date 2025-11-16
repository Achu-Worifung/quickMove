# -*- mode: python ; coding: utf-8 -*-

import os
import glob
import sys

from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_all,
)
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

# ---------------------------------------------------------------------
# 📦 Paths and Core Config
# ---------------------------------------------------------------------
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR'
TESSDATA_PATH = os.path.join(TESSERACT_PATH, 'tessdata')

# ---------------------------------------------------------------------
# 🧠 Collect package metadata / binaries for optional packages
# ---------------------------------------------------------------------
# faster_whisper and ctranslate2 (if present)
fw_datas, fw_binaries, fw_hidden = [], [], []
ct2_datas, ct2_binaries, ct2_hidden = [], [], []
try:
    fw_datas, fw_binaries, fw_hidden = collect_all('faster_whisper')
except Exception:
    print("faster_whisper not found or collect_all failed; continuing...")

try:
    ct2_datas, ct2_binaries, ct2_hidden = collect_all('ctranslate2')
except Exception:
    print("ctranslate2 not found or collect_all failed; continuing...")

# Collect transformers fully (this is critical to avoid lazy-loading issues)
t_datas, t_binaries, t_hidden = [], [], []
try:
    t_datas, t_binaries, t_hidden = collect_all('transformers')
except Exception:
    print("collect_all('transformers') failed; continuing (we'll still attempt other measures).")

# ---------------------------------------------------------------------
# 🧠 Hidden imports - COMPREHENSIVE transformers support
# ---------------------------------------------------------------------
hidden_imports = []

# Collect ALL transformers submodules to avoid lazy loading issues
try:
    hidden_imports += collect_submodules('transformers')
    hidden_imports += collect_submodules('transformers.models')
except Exception:
    print("collect_submodules for transformers failed; continuing...")

# Core dependencies (explicitly included)
hidden_imports += [
    'pytesseract', 'torch', 'numpy', 'tokenizers', 'sentencepiece',
    'requests', 'tqdm', 'regex', 'dataclasses', 'filelock',
    'huggingface_hub', 'safetensors', 'accelerate',
]

# Explicit common model entrypoints
hidden_imports += [
    'transformers.models.auto',
    'transformers.models.bert',
    'transformers.models.distilbert',
    'transformers.models.roberta',
    'transformers.models.gpt2',
    'transformers.models.t5',
    'transformers.models.whisper',
]

# Additional transformers internals that sometimes are missed
hidden_imports += [
    'transformers.tokenization_utils',
    'transformers.tokenization_utils_base',
    'transformers.tokenization_utils_fast',
    'transformers.configuration_utils',
    'transformers.modeling_utils',
    'transformers.file_utils',
]

# Add hidden imports discovered by collect_all for optional packages
hidden_imports += list(fw_hidden) if fw_hidden else []
hidden_imports += list(ct2_hidden) if ct2_hidden else []
hidden_imports += list(t_hidden) if t_hidden else []

# ---------------------------------------------------------------------
# 🧾 Safe collect extra datas
# ---------------------------------------------------------------------
def safe_collect(package_name):
    try:
        return collect_data_files(package_name)
    except Exception as e:
        print(f"Warning: Could not collect data files for {package_name}: {e}")
        return []

# ---------------------------------------------------------------------
# 🔹 Flattening function for datas/binaries
# ---------------------------------------------------------------------
def flatten_entries(entries):
    """Flatten and normalize datas/binaries into (src, dest) tuples."""
    normalized = []
    for e in entries:
        if isinstance(e, (list, tuple)):
            if len(e) >= 2:
                normalized.append((e[0], e[1]))
            else:
                print(f"Skipping invalid entry (too few elements): {e}")
        else:
            print(f"Skipping invalid entry (not tuple/list): {e}")
    return normalized

# ---------------------------------------------------------------------
# 🧩 Binaries and Datas
# ---------------------------------------------------------------------
binaries = [
    (os.path.join(TESSERACT_PATH, 'tesseract.exe'), '.'),
    (os.path.join(TESSERACT_PATH, 'libtesseract-5.dll'), '.'),
    (os.path.join(TESSERACT_PATH, 'leptonica-1.82.0.dll'), '.'),
    (os.path.join(TESSERACT_PATH, 'zlib1.dll'), '.'),
    (os.path.join(TESSERACT_PATH, 'libwebp-7.dll'), '.'),
    (os.path.join(TESSERACT_PATH, 'openjp2.dll'), '.'),
] + list(fw_binaries) + list(ct2_binaries) + list(t_binaries)

datas = [
    ('ui/*.ui', 'ui'),
    ('Icons/*.ico', 'Icons'),
    ('logo.ico', '.'),
    (os.path.join(TESSDATA_PATH, 'eng.traineddata'), 'tessdata'),
    ('.env', '.'),
] + list(fw_datas) + list(ct2_datas) + list(t_datas)

# Add safe collect for widely used packages
for pkg in ['tokenizers', 'torch', 'numpy', 'sentencepiece', 'huggingface_hub', 'safetensors']:
    try:
        datas += safe_collect(pkg)
    except Exception:
        pass

# Recursive inclusion of local model folders (if present)
for folder in ['models', 'finetuned_distilbert_bert', 'bibles']:
    if os.path.exists(folder):
        for file in glob.glob(f'{folder}/**/*', recursive=True):
            if os.path.isfile(file):
                rel_path = os.path.dirname(os.path.relpath(file, '.'))
                datas.append((file, rel_path))

# Normalize datas/binaries
datas = flatten_entries(datas)
binaries = flatten_entries(binaries)

# ---------------------------------------------------------------------
# 🧪 Runtime hook to fix transformers lazy loading
# ---------------------------------------------------------------------
runtime_hook_content = r"""
import sys
import os

# Prefer offline behavior for transformers in the frozen app
os.environ['TRANSFORMERS_OFFLINE'] = '0'
#os.environ['HF_HUB_OFFLINE'] = '0'

# When frozen, force transformers to avoid dynamic disk access that expects packages in normal layout.
if hasattr(sys, 'frozen'):
    try:
        import transformers
        # Mark offline (this helps avoid HF network calls)
        transformers.utils.is_offline_mode = lambda: True
    except Exception:
        # If transformers can't import here, we'll rely on the included datas/hiddenimports.
        pass
"""

hook_path = 'pyinstaller_transformers_hook.py'
online_hook = 'hf_enable_online_runtime_hook.py'
with open(hook_path, 'w', encoding='utf-8') as f:
    f.write(runtime_hook_content)

# ---------------------------------------------------------------------
# Analysis - IMPORTANT: set noarchive=True so modules are extracted to disk
# ---------------------------------------------------------------------
a = Analysis(
    ['main.py'],
    pathex=[os.path.abspath('.')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[hook_path, online_hook],
    excludes=[],
    noarchive=True,        # <- critical: prevents packing modules into the PYZ archive
    optimize=0,
)

pyz = PYZ(a.pure)

# ---------------------------------------------------------------------
# EXE / COLLECT - debug-friendly settings (toggle for release)
# ---------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='QuickMove',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,             # disable UPX for debugging; set to True for release if desired
    console=True,        
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon='logo.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,             # disable UPX at collection stage too
    upx_exclude=[],
    name='main',
)
