
import os, sys, inspect
os.environ["PYTORCH_JIT"]="0"
try:
    import torch._jit_internal as jit
except Exception: jit=None
if jit:
    if not hasattr(jit,'boolean_dispatch'): 
        jit.boolean_dispatch = lambda *a, **k: a[2] if len(a)>=3 else a[0]
    if not hasattr(jit,'is_scripting'): jit.is_scripting=lambda:False
    if not hasattr(jit,'is_tracing'): jit.is_tracing=lambda:False
    if not hasattr(jit,'_overload'): jit._overload=lambda f:f
    if not hasattr(jit,'_overload_method'): jit._overload_method=lambda f:f
    if not hasattr(jit,'_check_overload_body'): jit._check_overload_body=lambda *a,**k:None
inspect.getsource=lambda *a,**k:""
inspect.getsourcelines=lambda *a,**k:([],0)
inspect.findsource=lambda *a,**k:([],0)
