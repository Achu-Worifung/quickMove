import os
import subprocess
import sys

# =====================================================
# CONFIG
# =====================================================

MAIN_FILE = "main.py"   # <-- change if different
APP_NAME = "QuickMove"

VENV = r"C:\Users\achuw\OneDrive\Desktop\quick hsp\.venv\Lib\site-packages"
TESSERACT = r"C:\Program Files\Tesseract-OCR"

# =====================================================
# HELPER
# =====================================================

def add(cmd, value):
    cmd.append(value)

# =====================================================
# BASE COMMAND
# =====================================================

cmd = [
    sys.executable,
    "-m",
    "nuitka",
    MAIN_FILE,
]

# =====================================================
# CORE BUILD SETTINGS
# =====================================================

add(cmd, "--standalone")
add(cmd, "--assume-yes-for-downloads")
# add(cmd, "--output-dir=build")
add(cmd, f"--output-filename={APP_NAME}.exe")


add(cmd, "--nofollow-import-to=scipy.integrate._lebedev")
# add(cmd, "--nofollow-import-to=torch")

# UI
add(cmd, "--enable-plugin=pyqt5")
# add(cmd, "--disable-console")  # Disable console for GUI apps

# Performance / stability
add(cmd, "--jobs=1")
# add(cmd, "--compiler-choice=gcc")
# add(cmd, "--no-verbose")
# add(cmd, "--c-compiler-options=-O1")
# add(cmd, "--show-progress")
# add(cmd, "--show-scons")
# add(cmd, "--verbose")


# IMPORTANT — prevents linker memory crash
add(cmd, "--lto=no")

# =====================================================
# ML RUNTIME SETTINGS
# =====================================================

# Torch runtime stability
add(cmd, "--module-parameter=torch-disable-jit=yes")

# numba unsupported standalone
add(cmd, "--noinclude-numba-mode=nofollow")

# =====================================================
# REMOVE TEST / DEV MODULES (CRITICAL)
# =====================================================

add(cmd, "--nofollow-import-to=torch.testing")
add(cmd, "--nofollow-import-to=transformers.testing_utils")
add(cmd, "--nofollow-import-to=transformers.commands")
add(cmd, "--nofollow-import-to=sympy.testing")
add(cmd, "--nofollow-import-to=optimum.conftest")

# =====================================================
# REQUIRED PACKAGES
# =====================================================

# Compile these (pure Python, fine)
add(cmd, "--include-package=transformers")
add(cmd, "--include-package=optimum")

# Ship these as-is (heavy native libs)
for pkg in ["torch", "faiss", "ctranslate2", "onnxruntime", "faster_whisper"]: #skiping these ones because they are too heavy
    pkg_path = os.path.join(VENV, pkg)
    if not os.path.exists(pkg_path):
        print(f"WARNING: {pkg_path} does not exist, skipping!")
        continue
    add(cmd, f"--nofollow-import-to={pkg}")
    add(cmd, f"--include-data-dir={pkg_path}={pkg}")

add(cmd, "--report=build-report.xml")
add(cmd, "--report-diffable")


# =====================================================
# PROJECT DATA
# =====================================================

add(cmd, "--include-data-dir=classifier=classifier")
add(cmd, "--include-data-dir=bibles=bibles")
add(cmd, "--include-data-dir=artifacts=artifacts")
add(cmd, "--include-data-dir=search=search")
add(cmd, "--include-data-dir=ui=ui")
add(cmd, "--include-data-dir=Icons=Icons")

add(cmd, "--include-data-file=logo.ico=logo.ico")
add(cmd, "--include-data-file=.env=.env")

# =====================================================
# ONNX GPU PROVIDERS
# =====================================================

onnx_capi = os.path.join(VENV, "onnxruntime", "capi")

# add(cmd, f"--include-data-file={onnx_capi}\\onnxruntime_providers_cuda.dll=onnxruntime_providers_cuda.dll")
# add(cmd, f"--include-data-file={onnx_capi}\\onnxruntime_providers_shared.dll=onnxruntime_providers_shared.dll")
# add(cmd, f"--include-data-file={onnx_capi}\\onnxruntime_providers_tensorrt.dll=onnxruntime_providers_tensorrt.dll")

# =====================================================
# TESSERACT
# =====================================================

add(cmd, f'--include-data-file={TESSERACT}\\tesseract.exe=tesseract.exe')
add(cmd, f'--include-data-file={TESSERACT}\\libtesseract-5.dll=libtesseract-5.dll')
add(cmd, f'--include-data-file={TESSERACT}\\leptonica-1.82.0.dll=leptonica-1.82.0.dll')
add(cmd, f'--include-data-file={TESSERACT}\\zlib1.dll=zlib1.dll')
add(cmd, f'--include-data-file={TESSERACT}\\libwebp-7.dll=libwebp-7.dll')
add(cmd, f'--include-data-file={TESSERACT}\\openjp2.dll=openjp2.dll')
add(cmd, f'--include-data-dir={TESSERACT}\\tessdata=tessdata')


add(cmd, r"--output-dir=D:\QuickMoveBuild")


# =====================================================
# TORCH CUDA DLLS (AUTO INCLUDE)
# =====================================================

# torch_lib = os.path.join(VENV, "torch", "lib")

# # Keep track of what DLLs are already added
# included_dlls = {
#     entry.split('=')[-1].lower() 
#     for entry in cmd 
#     if entry.startswith("--include-data-file=")
# }


# if you already have other DLLs added manually, populate this:
# included_dlls.update(["asmjit.dll", "other.dll"])

# for dll in os.listdir(torch_lib):
#     dll_lower = dll.lower()
    
#     if not dll_lower.endswith(".dll"):
#         continue

#     if dll_lower in included_dlls:
#         print(f"Skipping already included DLL: {dll}")
#         continue

#     src = os.path.join(torch_lib, dll)
#     add(cmd, f"--include-data-file={src}={dll}")
#     included_dlls.add(dll_lower)

# =====================================================
# BUILD
# =====================================================

import subprocess
import re
import time
from tqdm import tqdm
from colorama import Fore, Style, init

init(autoreset=True)

# =====================================================
# ADVANCED BUILD MONITOR
# =====================================================

def run_with_progress(cmd):
    print(Fore.CYAN + "\n🚀 Starting Nuitka Production Build\n")

    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    stage = "Initializing"
    module_total = None
    module_bar = None
    scons_total = None
    scons_bar = None

    module_regex = re.compile(r"Progress:\s+(\d+)/(\d+)")
    scons_regex = re.compile(r"(\d+)/(\d+)\s+.*\.o")

    spinner = tqdm(
        total=0,
        bar_format="{desc}",
        leave=False
    )

    for line in process.stdout:

        line = line.rstrip()

        # ---------------------------
        # Detect stages
        # ---------------------------
        if "Compiling module" in line:
            stage = "🐍 Python compilation"
        elif "Running C compilation" in line or "Scons" in line:
            stage = "⚙️  C compilation"
        elif "Linking" in line:
            stage = "🔗 Linking"
        elif "Completed Python level compilation" in line:
            stage = "✅ Python compile finished"
        elif "Generating source code" in line:
            stage = "🧠 Generating C code"
        elif "data composer tool" in line:
            stage = "📦 Optimizing constants"

        spinner.set_description_str(Fore.YELLOW + f"[{stage}]")

        # ---------------------------
        # Module progress tracking
        # ---------------------------
        m = module_regex.search(line)
        if m:
            current = int(m.group(1))
            total = int(m.group(2))

            if module_bar is None:
                module_total = total
                module_bar = tqdm(
                    total=module_total,
                    desc="Modules",
                    unit="mod",
                    colour="green"
                )

            module_bar.n = current
            module_bar.refresh()
            continue

        # ---------------------------
        # SCons object compilation
        # ---------------------------
        s = scons_regex.search(line)
        if s:
            current = int(s.group(1))
            total = int(s.group(2))

            if scons_bar is None:
                scons_total = total
                scons_bar = tqdm(
                    total=scons_total,
                    desc="C Objects",
                    unit="obj",
                    colour="blue"
                )

            scons_bar.n = current
            scons_bar.refresh()
            continue

        # ---------------------------
        # Important messages only
        # ---------------------------
        if any(word in line for word in [
            "WARNING",
            "ERROR",
            "FATAL",
            "Nuitka:",
            "Scons:",
        ]):
            print(Fore.MAGENTA + line)

    process.wait()

    elapsed = time.time() - start_time

    if module_bar:
        module_bar.close()
    if scons_bar:
        scons_bar.close()
    spinner.close()

    print("\n" + "=" * 50)

    if process.returncode == 0:
        print(Fore.GREEN + "✅ BUILD SUCCESSFUL")
    else:
        print(Fore.RED + "❌ BUILD FAILED")

    print(Fore.CYAN + f"⏱ Total time: {elapsed/60:.2f} minutes")
    print("=" * 50)


# =====================================================
# RUN BUILD
# =====================================================

print("\n".join(cmd))
run_with_progress(cmd)


