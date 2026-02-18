
import sys
import os

# Set PyTorch JIT environment variable IMMEDIATELY
os.environ['PYTORCH_JIT'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

if hasattr(sys, 'frozen'):
    # Patch inspect module to return empty strings instead of crashing
    import inspect
    _orig_getsource = inspect.getsource
    _orig_getsourcelines = inspect.getsourcelines
    
    def _patched_getsource(obj):
        try:
            return _orig_getsource(obj)
        except:
            return ""
    
    def _patched_getsourcelines(obj):
        try:
            return _orig_getsourcelines(obj)
        except:
            return ([], 0)
    
    inspect.getsource = _patched_getsource
    inspect.getsourcelines = _patched_getsourcelines
    
    # Also patch findsource which is called by getsourcelines
    _orig_findsource = inspect.findsource
    
    def _patched_findsource(obj):
        try:
            return _orig_findsource(obj)
        except:
            return ([], 0)
    
    inspect.findsource = _patched_findsource
