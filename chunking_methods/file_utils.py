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

def make_dirs(species, train_path, validation_path):
    """If directories do not exist for species, make them
    """
    s_train_path = os.path.join(train_path, species)
    s_validation_path = os.path.join(validation_path, species)
    
    if not os.path.exists(s_train_path):
        os.makedirs(s_train_path)
    if not os.path.exists(s_validation_path):
        os.makedirs(s_validation_path)
    
    return s_train_path, s_validation_path
