"""
Pre-build script to patch transformers.__init__.py and generation.__init__.py
Run this BEFORE building with PyInstaller
"""
import sys
import os

def patch_file(filepath, patch_marker, patch_content):
    """Patch a file if not already patched"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if patch_marker in content:
            print(f"  Already patched: {filepath}")
            return True
        
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write('\n' + patch_content)
        
        print(f"  ✅ Patched: {filepath}")
        return True
    except Exception as e:
        print(f"  ❌ Error patching {filepath}: {e}")
        return False

try:
    import transformers
    transformers_path = os.path.dirname(transformers.__file__)
    
    print(f"Found transformers at: {transformers_path}")
    print("\nPatching transformers for PyInstaller...")
    
    # Patch 1: transformers/generation/__init__.py
    gen_init_path = os.path.join(transformers_path, 'generation', '__init__.py')
    gen_patch = '''

# PyInstaller patch: Explicit exports for frozen builds
try:
    from .utils import GenerationMixin
    from .configuration_utils import GenerationConfig
    __all__ = ['GenerationMixin', 'GenerationConfig']
except ImportError as e:
    print(f"Warning: Could not import generation classes: {e}")
'''
    
    # Patch 2: transformers/__init__.py for AutoModel classes
    main_init_path = os.path.join(transformers_path, '__init__.py')
    main_patch = '''

# PyInstaller patch: Force eager loading of Auto classes
if getattr(sys, 'frozen', False):
    try:
        from transformers.models.auto import (
            AutoModel,
            AutoTokenizer,
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoModelForQuestionAnswering,
        )
    except ImportError as e:
        print(f"PyInstaller: Could not eagerly load Auto classes: {e}")
'''
    
    success = True
    success &= patch_file(gen_init_path, "PyInstaller patch", gen_patch)
    success &= patch_file(main_init_path, "PyInstaller patch", main_patch)
    
    if success:
        print("\n✅ Successfully patched transformers!")
        print("Now run: pyinstaller --clean quickmove_fixed_v2.spec")
    else:
        print("\n⚠️ Some patches failed, but continuing...")
        sys.exit(1)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)