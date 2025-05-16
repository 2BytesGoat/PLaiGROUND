import os
import shutil
import importlib

def patch_library_file(library_name, file_path, your_modified_file):
    """
    Patch a file in an installed library with your modified version.
    
    Args:
        library_name (str): Name of the library to patch
        file_path (str): Path to the file within the library (relative to library root)
        your_modified_file (str): Path to your modified version of the file
    """
    try:
        # Get the library's installation directory
        library = importlib.import_module(library_name)
        library_dir = os.path.dirname(library.__file__)
        
        # Full path to the target file
        target_file = os.path.join(library_dir, file_path)
        
        # Create backup of original file
        backup_file = target_file + '.backup'
        if not os.path.exists(backup_file):
            shutil.copy2(target_file, backup_file)
            print(f"Created backup at: {backup_file}")
        
        # Copy your modified file
        shutil.copy2(your_modified_file, target_file)
        print(f"Successfully patched: {target_file}")
        
    except Exception as e:
        print(f"Error patching file: {str(e)}")

if __name__ == "__main__":
    patch_library_file('godot_rl', 'core/godot_env.py', 'patches/godot_env.py')