# Custom hook for PyTorch to handle distributed imports
from PyInstaller.utils.hooks import collect_submodules

# Include torch.distributed submodules that are actually needed
hiddenimports = [
    'torch.distributed',
     'torch.multiprocessing',
    'torch.distributed.distributed_c10d',
    'torch.distributed.rpc',  # Include but will be mocked at runtime
    'torch.distributed.rpc.api',
]

# Don't exclude these - PyTorch checks for them even if not used
excludedimports = []