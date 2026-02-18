# Custom hook for transformers to ensure all required modules are collected
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata

# Collect all generation submodules
hiddenimports = collect_submodules('transformers.generation')

# Explicitly add the generation modules where GenerationMixin is defined
hiddenimports += [
    'transformers.generation.utils',
    'transformers.generation.configuration_utils',
    'transformers.generation.stopping_criteria',
    'transformers.generation.logits_process',
    'transformers.generation.streamers',
    'transformers.modeling_utils',
    # Add the specific files that define generation classes
    'transformers.generation.beam_search',
    'transformers.generation.beam_constraints',
    'transformers.generation.flax_utils',
    'transformers.generation.tf_utils',
]

# Collect data files (configs, etc.) including Python files this time
# This ensures __init__.py and other module files are properly included
datas = collect_data_files('transformers', include_py_files=True)
datas += collect_data_files('transformers.generation', include_py_files=True)

# Collect package metadata for version checking
try:
    datas += copy_metadata('transformers')
    datas += copy_metadata('tqdm')
    datas += copy_metadata('tokenizers')
    datas += copy_metadata('huggingface-hub')
    datas += copy_metadata('safetensors')
    datas += copy_metadata('numpy')
    datas += copy_metadata('regex')
    datas += copy_metadata('requests')
    datas += copy_metadata('filelock')
    datas += copy_metadata('packaging')
except Exception:
    pass  # Metadata collection is best-effort