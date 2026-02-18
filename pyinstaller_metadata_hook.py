
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
