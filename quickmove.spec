# -*- mode: python ; coding: utf-8 -*-

import os
import glob

from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_all,
    copy_metadata,
)

from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT


# =====================================================
# 📦 PATHS
# =====================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR"
TESSDATA_PATH = os.path.join(TESSERACT_PATH, "tessdata")

# =====================================================
# 🧩 HIDDEN IMPORTS - BASE
# =====================================================

hiddenimports = [
    "sklearn",
    "sklearn.metrics",
    "sklearn.utils",
    "sklearn.externals.array_api_compat",
    "sklearn.externals.array_api_compat.numpy",
    "sklearn.externals.array_api_compat.numpy.fft",
    # transformers core for MiniLM
    "transformers",
    "transformers.modeling_utils",
    "transformers.models.bert.modeling_bert",
    "transformers.modeling_outputs",
    "transformers.models.auto",
    "transformers.models.auto.configuration_auto",
    "transformers.models.auto.modeling_auto",
    "transformers.models.auto.tokenization_auto",
    # sentence-transformers
    "sentence_transformers",
    # optimum ONNX (YOUR bible search uses this)
    "optimum",
    "optimum.onnxruntime",
    "optimum.onnxruntime.modeling_ort",
    "optimum.exporters",
    "optimum.exporters.onnx",
    # torch._dynamo needs this
    "optree",
]

# =====================================================
# 📦 COLLECT CORE PACKAGES
# =====================================================

# Scikit-learn
sk_datas, sk_binaries, sk_hidden = [], [], []
try:
    sk_datas, sk_binaries, sk_hidden = collect_all("sklearn")
    print("✓ sklearn collected")
except Exception as e:
    print(f"sklearn collection issue: {e}")

# Transformers
datas_t, bins_t, hidden_t = collect_all("transformers")

# Sentence-Transformers
st_datas, st_binaries, st_hidden = collect_all("sentence_transformers")

# Faiss
faiss_datas, faiss_binaries, faiss_hidden = collect_all("faiss")

# Torch
torch_datas, torch_binaries, torch_hidden = collect_all("torch")

# Optree (torch._dynamo dependency)
optree_datas, optree_binaries, optree_hidden = collect_all("optree")

# Optimum (YOUR bible search needs this)
optimum_datas, optimum_binaries, optimum_hidden = collect_all("optimum")

# ONNX Runtime (optimum backend)
ort_datas, ort_binaries, ort_hidden = [], [], []
try:
    ort_datas, ort_binaries, ort_hidden = collect_all("onnxruntime")
    print("✓ onnxruntime collected")
except Exception as e:
    print(f"onnxruntime collection issue: {e}")

# Combine all hidden imports
hiddenimports += (
    hidden_t + sk_hidden + st_hidden 
    + faiss_hidden + torch_hidden + optree_hidden
    + optimum_hidden + ort_hidden
)


# =====================================================
# 🧾 DATA FILES
# =====================================================

datas = [
    ("ui/*.ui", "ui"),
    ("Icons/*.ico", "Icons"),
    ("logo.ico", "."),
    (os.path.join(TESSDATA_PATH, "eng.traineddata"), "tessdata"),
    (".env", "."),
    ("settings.ini", "."),
]

# add collected data
datas += (
    datas_t + sk_datas + st_datas 
    + faiss_datas + torch_datas + optree_datas
    + optimum_datas + ort_datas
)

# metadata
for pkg in ['transformers', 'sentence-transformers', 'torch', 'numpy', 'sklearn', 
            'tokenizers', 'huggingface-hub', 'safetensors', 'pyyaml', 'tqdm',
            'regex', 'requests', 'packaging', 'filelock', 'optree',
            'optimum', 'onnxruntime']:
    try:
        datas += copy_metadata(pkg)
    except:
        pass

# ---- local model/data folders ----
for folder in ["classifier", "search", "bibles", "artifacts"]:
    if os.path.exists(folder):
        for f in glob.glob(f"{folder}/**/*", recursive=True):
            if os.path.isfile(f):
                rel = os.path.dirname(os.path.relpath(f, "."))
                datas.append((f, rel))


# =====================================================
# 🧩 BINARIES
# =====================================================

binaries = [
    (os.path.join(TESSERACT_PATH, "tesseract.exe"), "."),
    (os.path.join(TESSERACT_PATH, "libtesseract-5.dll"), "."),
    (os.path.join(TESSERACT_PATH, "leptonica-1.82.0.dll"), "."),
    (os.path.join(TESSERACT_PATH, "zlib1.dll"), "."),
    (os.path.join(TESSERACT_PATH, "libwebp-7.dll"), "."),
    (os.path.join(TESSERACT_PATH, "openjp2.dll"), "."),
]

binaries += (
    bins_t + sk_binaries + st_binaries 
    + faiss_binaries + torch_binaries + optree_binaries
    + optimum_binaries + ort_binaries
)


# =====================================================
# 🧪 RUNTIME HOOKS
# =====================================================

# metadata version spoof
metadata_hook = r"""
import sys
if hasattr(sys, 'frozen'):
    import importlib.metadata as metadata
    _real_version = metadata.version
    SAFE_VERSIONS = {
        "tqdm": "4.66.0", "tokenizers": "0.15.0", "huggingface-hub": "0.20.0",
        "safetensors": "0.4.0", "numpy": "1.24.0", "regex": "2023.10.0",
        "requests": "2.31.0", "packaging": "23.0", "pyyaml": "6.0", "filelock": "3.12.0",
    }
    def safe_version(name):
        try: return _real_version(name)
        except: return SAFE_VERSIONS.get(name.lower(), "99.0.0")
    metadata.version = safe_version
"""
metadata_hook_path = "pyinstaller_metadata_hook.py"
open(metadata_hook_path, "w", encoding="utf-8").write(metadata_hook)

# transformers env + ONNX path fix
env_hook = r"""
import os, sys
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
if hasattr(sys, "_MEIPASS"):
    os.environ["PATH"] = sys._MEIPASS + ";" + os.environ.get("PATH", "")
"""
env_hook_path = "pyinstaller_env_hook.py"
open(env_hook_path, "w", encoding="utf-8").write(env_hook)


# =====================================================
# 🛠 ANALYSIS
# =====================================================

a = Analysis(
    ["main.py"],
    pathex=[os.path.abspath(".")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    runtime_hooks=[metadata_hook_path, env_hook_path],
    excludes=[
        # Whisper stack ONLY - GONE
        "faster_whisper", "ctranslate2", "silero_vad", "av",
        # Stuff you don't use
        "torchvision", "torchaudio", "torch.utils.tensorboard",
        "matplotlib", "notebook", "IPython", "pytest",
        # Heavy ONNX stuff you don't need
        "onnxruntime.quantization", "onnxruntime.training",
    ],
    noarchive=True,
)


# =====================================================
# BUILD
# =====================================================

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="QuickMove",
    console=True,
    icon="logo.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="QuickMove",
)