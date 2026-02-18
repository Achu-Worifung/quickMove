# Custom hook for onnxruntime to avoid subprocess crashes
# This hook prevents PyInstaller from trying to import problematic modules

from PyInstaller.utils.hooks import collect_data_files
import os

# Collect data files without triggering imports
datas = []
binaries = []

try:
    datas = collect_data_files('onnxruntime', include_py_files=True)
except:
    pass

# Manually specify hidden imports - DO NOT use collect_submodules or collect_all
hiddenimports = [
    'onnxruntime.capi._pybind_state',
    'onnxruntime.capi.onnxruntime_pybind11_state',
]

# Exclude the problematic modules that cause subprocess crashes
excludedimports = [
    'onnx.reference',
    'onnx.reference.ops',
]