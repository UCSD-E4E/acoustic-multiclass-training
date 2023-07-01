"""Used as a helper file for testing audio file moves
"""
import os

def clear_files(path):
    """Recursively remove files in a folder
    """
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    for s in subfolders:
        subfolders_type = [f.path for f in os.scandir(s) if f.is_dir()]
        for s_type in subfolders_type:
            files = [f.path for f in os.scandir(s_type) if f.is_file()]
            for file in files:
                os.remove(file)
