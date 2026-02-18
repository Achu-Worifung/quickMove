# -*- mode: python ; coding: utf-8 -*-

import os
import glob

from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_all,
    copy_metadata,
)

from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT


# =====================================================
# 📦 PATHS
# =====================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR"
TESSDATA_PATH = os.path.join(TESSERACT_PATH, "tessdata")

hiddenimports = []
hiddenimports += [
    "sklearn",
    "sklearn.metrics",
    "sklearn.utils",
    "sklearn.externals",
    "sklearn.externals.array_api_compat",
    "sklearn.externals.array_api_compat.numpy",
    "sklearn.externals.array_api_compat.numpy.fft",
]


# -------------------------------
# 📦 Scikit-learn FULL COLLECTION
# -------------------------------
sk_datas, sk_binaries, sk_hidden = [], [], []

try:
    sk_datas, sk_binaries, sk_hidden = collect_all("sklearn")
    print("sklearn collected successfully")
except Exception as e:
    print(f"sklearn collection issue: {e}")


# =====================================================
# 🧠 COLLECT CORE PACKAGES
# =====================================================

# ---- Transformers full collection ----
datas_t, bins_t, hidden_t = collect_all("transformers")

# ---- Optional packages ----
fw_datas, fw_binaries, fw_hidden = [], [], []
ct2_datas, ct2_binaries, ct2_hidden = [], [], []
optimum_datas, optimum_binaries, optimum_hidden = [], [], []

for pkg in ["faster_whisper", "ctranslate2", "optimum"]:
    try:
        d, b, h = collect_all(pkg)
        if pkg == "faster_whisper":
            fw_datas, fw_binaries, fw_hidden = d, b, h
        elif pkg == "ctranslate2":
            ct2_datas, ct2_binaries, ct2_hidden = d, b, h
        else:
            optimum_datas, optimum_binaries, optimum_hidden = d, b, h
        print(f"{pkg} collected")
    except Exception:
        print(f"{pkg} not found")


# =====================================================
# 🧩 HIDDEN IMPORTS
# =====================================================

hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("transformers")

hiddenimports += [
    "transformers.models.auto",
    "transformers.models.auto.configuration_auto",
    "transformers.models.auto.modeling_auto",
    "transformers.models.auto.tokenization_auto",
    "transformers.models.auto.feature_extraction_auto",
    "transformers.models.auto.processing_auto",
    "transformers.pipelines",
]

hiddenimports += hidden_t + fw_hidden + ct2_hidden + optimum_hidden
hiddenimports += sk_hidden


# =====================================================
# 🧾 DATA FILES
# =====================================================

datas = [
    ("ui/*.ui", "ui"),
    ("Icons/*.ico", "Icons"),
    ("logo.ico", "."),
    (os.path.join(TESSDATA_PATH, "eng.traineddata"), "tessdata"),
    (".env", "."),
]

datas += datas_t + fw_datas + ct2_datas + optimum_datas
datas += sk_datas



# ---- metadata ----
for pkg in ["optimum", "onnxruntime", "onnxruntime-gpu"]:
    try:
        datas += copy_metadata(pkg)
    except:
        pass


# ---- local model folders ----
for folder in ["finetuned_distilbert_bert", "bibles"]:
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

binaries += bins_t + fw_binaries + ct2_binaries + optimum_binaries
binaries += sk_binaries



# =====================================================
# 🧪 RUNTIME HOOKS
# =====================================================

# ---------- metadata version spoof ----------
metadata_hook = r"""
import sys

if hasattr(sys, 'frozen'):
    import importlib.metadata as metadata

    _real_version = metadata.version

    SAFE_VERSIONS = {
        "tqdm": "4.66.0",
        "tokenizers": "0.15.0",
        "huggingface-hub": "0.20.0",
        "safetensors": "0.4.0",
        "numpy": "1.24.0",
        "regex": "2023.10.0",
        "requests": "2.31.0",
        "packaging": "23.0",
        "pyyaml": "6.0",
        "filelock": "3.12.0",
    }

    def safe_version(name):
        try:
            return _real_version(name)
        except Exception:
            return SAFE_VERSIONS.get(name.lower(), "99.0.0")

    metadata.version = safe_version
    sys.modules["importlib.metadata"].version = safe_version
"""

metadata_hook_path = "pyinstaller_metadata_hook.py"
open(metadata_hook_path,"w",encoding="utf-8").write(metadata_hook)


# ---------- transformers env ----------
transformers_env = r"""
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"]="1"
"""
transformers_env_path = "pyinstaller_transformers_env.py"
open(transformers_env_path,"w",encoding="utf-8").write(transformers_env)


# ---------- ONNX CUDA ----------
onnx_hook = r"""
import os, sys
if hasattr(sys,"_MEIPASS"):
    os.environ["PATH"]=sys._MEIPASS+";"+os.environ.get("PATH","")
"""
onnx_hook_path = "onnx_cuda_hook.py"
open(onnx_hook_path,"w",encoding="utf-8").write(onnx_hook)


# ---------- ultra early torch ----------
torch_patch = r"""
import os
os.environ["PYTORCH_JIT"]="0"
"""
torch_patch_path = "torch_early_patch.py"
open(torch_patch_path,"w",encoding="utf-8").write(torch_patch)


# ---------- CRITICAL transformers generation fix ----------
generation_fix = r"""
try:
    import transformers.generation as g
    from transformers.generation.utils import GenerationMixin
    from transformers.generation.configuration_utils import GenerationConfig
    g.GenerationMixin = GenerationMixin
    g.GenerationConfig = GenerationConfig
except Exception:
    pass
"""
generation_fix_path = "runtime_hook_transformers_generation_fix.py"
open(generation_fix_path,"w",encoding="utf-8").write(generation_fix)


# =====================================================
# 🛠 ANALYSIS
# =====================================================

a = Analysis(
    ["main.py"],
    pathex=[os.path.abspath(".")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    runtime_hooks=[
        torch_patch_path,
        metadata_hook_path,
        transformers_env_path,
        onnx_hook_path,
        generation_fix_path,
    ],
    excludes=[
        "onnxruntime.quantization",
        "onnxruntime.training",
        "onnx.reference",
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
    console=False,
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
