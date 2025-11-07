
import sys
import os

# Prefer offline behavior for transformers in the frozen app
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# When frozen, force transformers to avoid dynamic disk access that expects packages in normal layout.
if hasattr(sys, 'frozen'):
    try:
        import transformers
        # Mark offline (this helps avoid HF network calls)
        transformers.utils.is_offline_mode = lambda: True
    except Exception:
        # If transformers can't import here, we'll rely on the included datas/hiddenimports.
        pass
