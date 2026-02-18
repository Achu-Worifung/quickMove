# Custom hook for onnx to prevent importing problematic reference modules

# Explicitly exclude the modules that crash during analysis
excludedimports = [
    'onnx.reference',
    'onnx.reference.ops',
    'onnx.reference.ops.aionnx_preview_training',
    'onnx.reference.ops.aionnxml_preview_training', 
]

# Don't collect anything from onnx - we only need onnxruntime
datas = []
binaries = []
hiddenimports = []