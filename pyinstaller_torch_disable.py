
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # <-- KILLS torch.compile/torch._dynamo
os.environ["PYTORCH_JIT"] = "0"
