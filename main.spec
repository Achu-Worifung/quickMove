# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Define Tesseract paths
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR'
TESSDATA_PATH = os.path.join(TESSERACT_PATH, 'tessdata')

# Comprehensive hidden imports for transformers and related packages
transformers_hidden_imports = collect_submodules('transformers')
transformers_hidden_imports += [
    'pytesseract',
    'torch',
    'numpy',
    'tokenizers',
    'sentencepiece',
    'protobuf',  # Explicitly add missing import
    'protobuf.descriptor',
    'protobuf.json_format',
    'requests',
    'tqdm',
    'regex',
    'sacremotes',
    'dataclasses',
    'scipy.special._cdflib',  # Missing scipy component
    'sip',  # Missing sip
    'whisper',
    'whisper.tokenizer',
]

# Add common model-specific imports
model_specific_imports = [
    'transformers.models.bert',
    'transformers.models.distilbert', 
    'transformers.models.roberta',
    'transformers.models.gpt2',
    'transformers.models.t5',
    'transformers.models.whisper',
]

transformers_hidden_imports += model_specific_imports

a = Analysis(
    ['main.py'],
    pathex=[os.path.abspath('.')],
    binaries=[
        # Include Tesseract executable and required DLLs
        (r'C:\Program Files\Tesseract-OCR\tesseract.exe', '.'),
        (r'C:\Program Files\Tesseract-OCR\libtesseract-5.dll', '.'),
        (r'C:\Program Files\Tesseract-OCR\leptonica-1.82.0.dll', '.'),
        (r'C:\Program Files\Tesseract-OCR\zlib1.dll', '.'),
        (r'C:\Program Files\Tesseract-OCR\libwebp-7.dll', '.'),
        (r'C:\Program Files\Tesseract-OCR\openjp2.dll', '.'),
    ],
    datas=[
        ('ui/*.ui', 'ui'),
        ('Icons/*.ico', 'Icons'),
        ('finetuned_distilbert_bert/*', 'finetuned_distilbert_bert'),
        ('logo.ico', '.'),
        # Create models directory for runtime downloads
        ('models', 'models'),
        # Add Tesseract training data
        (os.path.join(TESSDATA_PATH, 'eng.traineddata'), 'tessdata'),
        # Add .env file
        ('.env', '.'),
    ],
    hiddenimports=transformers_hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# Function to safely collect and format data files
def safe_collect_data_files(package_name):
    try:
        datas = collect_data_files(package_name)
        # Convert to 3-tuples: (src, dest, 'DATA')
        return [(src, dest, 'DATA') for src, dest in datas]
    except Exception as e:
        print(f"Warning: Could not collect data files for {package_name}: {e}")
        return []

# Collect all data files from transformers and related packages
transformers_datas = safe_collect_data_files('transformers')
tokenizers_datas = safe_collect_data_files('tokenizers')
torch_datas = safe_collect_data_files('torch')

a.datas.extend(transformers_datas)
a.datas.extend(tokenizers_datas) 
a.datas.extend(torch_datas)

# Add additional packages that might be needed
additional_packages = ['numpy', 'torch', 'tokenizers', 'sentencepiece', 'protobuf', 'scipy']
for pkg in additional_packages:
    try:
        pkg_datas = safe_collect_data_files(pkg)
        a.datas.extend(pkg_datas)
        a.hiddenimports.extend(collect_submodules(pkg))
    except Exception as e:
        print(f"Warning: Could not process {pkg}: {e}")

# Ensure all datas entries are 3-tuples
def ensure_3_tuples(datas_list):
    fixed_list = []
    for item in datas_list:
        if len(item) == 2:
            # Convert (src, dest) to (src, dest, 'DATA')
            fixed_list.append((item[0], item[1], 'DATA'))
        else:
            fixed_list.append(item)
    return fixed_list

# Fix the datas list
a.datas = ensure_3_tuples(a.datas)

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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='logo.ico',
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