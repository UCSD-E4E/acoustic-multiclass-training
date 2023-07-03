"""Split data into training and validation sets.
Distributes files using a 4:1 split, then distributes chunks
according to those files to avoid leakage between sets
"""
import os
import shutil
import sys
from random import shuffle
from math import floor
from file_utils import clear_files, make_dirs

VALIDATION_DISTR = 0.2

def distribute_files(path):
    """Shuffle splits the original audio files (non-chunked).
    Only move files that chunks were successfully generated for.
    """
    file_path = os.path.join(path, 'BirdCLEF2023_train_audio')
    wav_path = os.path.join(path, 'BirdCLEF2023_split_audio')
    chunk_path_old = os.path.join(path, 'BirdCLEF2023_train_audio_chunks')
    subfolders = [f.path for f in os.scandir(file_path) if f.is_dir()]

    train_path = os.path.join(wav_path, 'training')
    validation_path = os.path.join(wav_path, 'validation')

    def contains_chunks(f):
        file_name = f.path.split('/')[-1][:-4]
        species = f.path.split('/')[-2]
        chunk_path = os.path.join(chunk_path_old, species)
        return os.path.exists(chunk_path) and len([fn for fn in os.scandir(chunk_path) if file_name in fn.path]) > 0

    for s in subfolders:
        species = s.split('/')[-1]
        num_files = int(len(os.listdir(s)) / 2)
        s_train_path, s_validation_path = make_dirs(species, train_path, validation_path)
        files = [f.path.split('/')[-1] for f in os.scandir(s) if f.path.endswith('.wav') and contains_chunks(f)]

        if len(files) == 0:
            continue
        # if the number of files is 1, we'll keep the same file in both training and validation
        if len(files) == 1:
            file = files[0]
            file_name = file.split('/')[-1]

            shutil.copyfile(os.path.join(s, file), os.path.join(s_train_path, file_name))
            shutil.copyfile(os.path.join(s, file), os.path.join(s_validation_path, file_name))

        else:
            num_validation = max(1, floor(num_files * VALIDATION_DISTR))            
            shuffle(files)

            train_files = files[:-num_validation]
            validation_files = files[-num_validation:]

            for file in train_files:
                shutil.copyfile(os.path.join(s, file), os.path.join(s_train_path, file.split('/')[-1]))
            
            for file in validation_files:
                shutil.copyfile(os.path.join(s, file), os.path.join(s_validation_path, file.split('/')[-1]))


def distribute_chunks(path):
    """Move chunks into respective training/validation folders
    """
    wav_path = os.path.join(path, 'BirdCLEF2023_split_audio')
    chunk_path_old = os.path.join(path, 'BirdCLEF2023_train_audio_chunks')
    chunk_path_new = os.path.join(path, 'BirdCLEF2023_split_chunks')
    subfolders = [f.path for f in os.scandir(wav_path) if f.is_dir()]

    # for training and validation folders:
    for s in subfolders:
        # s: /share/acoustic_species_id/BirdCLEF2023_split_audio/validation
        species_folders = [f.path for f in os.scandir(s) if f.is_dir()]

        for species_path in species_folders:
            # species_path: /share/acoustic_species_id/BirdCLEF2023_split_audio/validation/rerswa1

            species = species_path.split('/')[-1]
            chunk_path_dst = os.path.join(chunk_path_new, species_path.split('/')[-2], species)
            if not os.path.exists(chunk_path_dst):
                os.makedirs(chunk_path_dst)

            chunk_path_src = os.path.join(chunk_path_old, species)
            if not os.path.exists(chunk_path_src):
                continue

            wav_file_path = os.path.join(wav_path, species_path.split('/')[-2], species)
            wav_files = [f.path for f in os.scandir(wav_file_path) if f.is_file()]

            for wav_file in wav_files:

                chunks = [f.path for f in os.scandir(chunk_path_src) if wav_file.split('/')[-1][:-4] in f.path]

                for chunk in chunks:
                    shutil.copyfile(chunk, os.path.join(chunk_path_dst, chunk.split('/')[-1]))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Incorrect number of args", file=sys.stderr)
        print("USAGE: python distribute_chunks.py /path", file=sys.stderr)
        sys.exit(1)
    distribute_files(sys.argv[1])
    distribute_chunks(sys.argv[1])
    clear_files(os.path.join(sys.argv[1], 'BirdCLEF2023_split_audio'))
