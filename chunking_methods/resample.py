"""To fix long-tail distribution training problem, limit classes to 50-500 samples
"""
import os
import shutil
from random import shuffle
from random import choice
from file_utils import clear_files

chunk_path_old = '/share/acoustic_species_id/pretraining_combined'
chunk_path_new = '/share/acoustic_species_id/pretraining_combined_resampled'
down_sample_from = 50
up_sample_to = 0
filetype = ".mp3"

def distribute_files():
    """Upsample classes with fewer than 50 samples to 50, limit classes to 500 samples
    """
    # get list of folders
    #file_path = os.path.join(share_path, 'BirdCLEF2023_train_audio')
    subfolders = [f.path for f in os.scandir(chunk_path_old) if f.is_dir()]
    upsampled_classes = 0
    downsampled_classes = 0

    # def contains_chunks(f):
    #     file_name = f.path.split('/')[-1][:-4]
    #     species = f.path.split('/')[-2]
    #     chunk_path = os.path.join(chunk_path_old, species)
    #     return os.path.exists(chunk_path) and len([fn for fn in os.scandir(chunk_path) if file_name in fn.path]) > 0

    for s in subfolders:
        species = s.split('/')[-1]
        s_path = os.path.join(chunk_path_new, species)
        files = [f.path.split('/')[-1] for f in os.scandir(s) if f.path.endswith(filetype)] #and contains_chunks(f)]
        if len(files) == 0:
            continue
        if not os.path.exists(s_path):
            os.makedirs(s_path)
        if len(files) > down_sample_from:
            shuffle(files)
            copy_files = files[1:down_sample_from]
            downsampled_classes += 1
        elif len(files) >= up_sample_to:
            # copy everything over
            copy_files = files
        else:
            copy_files = []
            for _ in range(up_sample_to):
                copy_files.append(choice(files))
            #print(f"upsampled {species} from {len(files)} to {up_sample_to}")
            upsampled_classes += 1

        for file in copy_files:
            shutil.copyfile(os.path.join(s, file), os.path.join(s_path, file.split('/')[-1]))
    print(downsampled_classes)
    print(upsampled_classes)

if __name__ == '__main__':
    clear_files(chunk_path_new)
    distribute_files()
