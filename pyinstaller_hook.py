
import sys
import os

# Disable transformers lazy loading in frozen app
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# Fix transformers import structure
if hasattr(sys, 'frozen'):
    import transformers
    # Force eager loading of models
    transformers.utils.is_offline_mode = lambda: True
