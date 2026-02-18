"""
Patch PyTorch's _jit_internal.py to disable JIT overload decorator
This fixes the "Expected a single top-level function" error in frozen builds
"""
import sys
import os

try:
    import torch
    torch_path = os.path.dirname(torch.__file__)
    jit_internal_path = os.path.join(torch_path, '_jit_internal.py')
    
    print(f"Found PyTorch at: {torch_path}")
    print(f"Patching: {jit_internal_path}")
    
    # Read the current content
    with open(jit_internal_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if 'PyInstaller patch' in content:
        print("✅ Already patched!")
        sys.exit(0)
    
    # Find the _overload function and replace it with a no-op version
    patch = '''

# PyInstaller patch: Disable JIT overload decorator in frozen builds
import sys
if getattr(sys, 'frozen', False):
    # In frozen mode, make _overload a no-op decorator
    def _overload_patch(func):
        """No-op overload for frozen builds"""
        return func
    
    # Replace the original _overload with our no-op version
    _overload = _overload_patch
'''
    
    # Add the patch at the end
    with open(jit_internal_path, 'a', encoding='utf-8') as f:
        f.write(patch)
    
    print("✅ Successfully patched PyTorch _jit_internal.py")
    print("\nNow rebuild:")
    print("  pyinstaller --clean quickmove_fixed_v2.spec")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)