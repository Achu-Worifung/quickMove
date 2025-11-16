
import sys
import os

# Disable transformers lazy loading in frozen app
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_OFFLINE'] = '0'

# Fix transformers import structure
if hasattr(sys, 'frozen'):
    try:
        import transformers
        # Force eager loading of models
        transformers.utils.is_offline_mode = lambda: False
    except Exception:
        pass
