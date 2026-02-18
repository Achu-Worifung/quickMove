# Custom hook for transformers to ensure all required modules are collected
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all generation submodules
hiddenimports = collect_submodules('transformers.generation')

# Also collect modeling_utils which contains GenerationMixin
hiddenimports += [
    'transformers.generation.utils',
    'transformers.generation.configuration_utils',
    'transformers.generation.stopping_criteria',
    'transformers.generation.logits_process',
    'transformers.generation.streamers',
    'transformers.modeling_utils',
]

# Collect data files (configs, etc.)
datas = collect_data_files('transformers', include_py_files=False)