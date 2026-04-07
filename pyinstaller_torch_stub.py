
import sys
import types

# 1. Create fake torch.compiler.disable NOTHING decorator
def _dummy_decorator(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    return lambda f: f

# 2. Inject stub BEFORE torch is imported
compiler_stub = types.ModuleType("torch.compiler")
compiler_stub.disable = _dummy_decorator
compiler_stub.compile = _dummy_decorator
sys.modules["torch.compiler"] = compiler_stub

# 3. Also stub torch._dynamo to prevent optree import
dynamo_stub = types.ModuleType("torch._dynamo")
dynamo_stub.disable = _dummy_decorator
dynamo_stub.config = types.ModuleType("torch._dynamo.config")
sys.modules["torch._dynamo"] = dynamo_stub
sys.modules["torch._dynamo.config"] = dynamo_stub.config

# 4. Environment vars
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
