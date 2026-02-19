
import sys

# 1. FORCE TORCH TO INITIALIZE NORMALLY FIRST
try:
    import torch
except ImportError:
    pass

# 2. NOW INJECT STUBS IF DYNAMO IS MISSING
if "torch" in sys.modules:
    import types
    import torch
    
    if not hasattr(torch, "_dynamo"):
        def make_stub_module(name):
            m = types.ModuleType(name)
            m.__path__ = []
            return m

        # Create the stub
        stub = make_stub_module("torch._dynamo")
        
        # Universal decorator stub
        def identity_decorator(*args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            return lambda fn: fn
        
        # Attach required attributes
        stub.allow_in_graph = identity_decorator
        stub.disallow_in_graph = identity_decorator
        stub.disable = identity_decorator
        stub.mark_static_address = identity_decorator
        stub.is_compiling = lambda: False
        stub.config = make_stub_module("torch._dynamo.config")
        
        sys.modules["torch._dynamo"] = stub
        sys.modules["torch._dynamo.config"] = stub.config
        
        # Patch compiler
        if hasattr(torch, "compiler"):
            torch.compiler.disable = identity_decorator
