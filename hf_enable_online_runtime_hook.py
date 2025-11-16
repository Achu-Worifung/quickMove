# hf_enable_online_runtime_hook.py
# This is a PyInstaller runtime hook. It runs very early in the frozen executable
# before your application code and before imports of transformers/huggingface_hub.

import os
import sys

# Clear any HF offline env flags
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

# Ensure a writable cache_dir exists (optional but useful)
try:
    from pathlib import Path
    preferred = Path("D:/my_app_cache")
    fallback = Path(os.environ.get("LOCALAPPDATA", sys._MEIPASS if hasattr(sys, '_MEIPASS') else '.')) / "my_app_cache"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        cache_dir = str(preferred) if os.access(preferred, os.W_OK) else str(fallback)
    except Exception:
        cache_dir = str(fallback)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")
except Exception:
    pass

# Force transformers/hf-hub out of offline behavior if already imported
try:
    import transformers
    # transformers.utils.is_offline_mode is a function; override to return False
    if hasattr(transformers, "utils") and hasattr(transformers.utils, "is_offline_mode"):
        transformers.utils.is_offline_mode = lambda: False
except Exception:
    # transformer may not be importable in the runtime hook; that's fine
    pass
