"""Create "raw" end-to-end audio chunks from an original .wav file.
ie. no centering, sliding window, etc. 
"""
import os
from math import ceil, floor
from random import shuffle
import pydub
from file_utils import make_dirs

VALIDATION_DISTR = 0.2

#wav_path = '/share/acoustic_species_id/BirdCLEF2023_train_audio'
#chunk_path = '/share/acoustic_species_id/BirdCLEF2023_raw_chunks'
wav_path = '/home/sprestrelski/TEST'
chunk_path = '/home/sprestrelski/TEST_raw_chunks'
chunk_dict = {}

def gen_raw_chunks(pad=False): 
    """Create 5 second chunks from a wav file
    """  
    subfolders = [f.path for f in os.scandir(wav_path) if f.is_dir()]

    for s in subfolders:
        wavs = [f.path for f in os.scandir(s) if f.path[-4:] == '.wav']
        species = s.split('/')[-1]
        chunk_dict[species] = []
        
        for wav in wavs:
            wav_file = pydub.AudioSegment.from_wav(wav)
            wav_file_name = wav.split('/')[-1][:-4]

            wav_length = len(wav_file) # in ms
            num_chunks = ceil(wav_length / 5000)

            for i in range(num_chunks):
                start = i * 5000
                end = start + 5000 

                if end > wav_length and pad:
                    pass
                else:
                    chunk_dict[species].append((wav_file_name, wav_file[start : end]))
    
def distribute_raw_chunks():
    """Distribute previously created chunks into training and validation folders.
    """
    train_path = os.path.join(chunk_path, 'training')
    validation_path = os.path.join(chunk_path, 'validation')
    chunk_items = chunk_dict.items()
    for species in chunk_items:
        # items is organized [species, [chunks]]
        chunks = species[1]

        # remove duplicate chunks
        files = list({chunk[0] for chunk in chunks})
        s_train_path, s_validation_path = make_dirs(species[0], train_path, validation_path)

        if len(files) == 0:
            continue        
        if len(files) == 1:
            file_name = files[0]
            chunk_wavs = [chunk[1] for chunk in chunks]

            for i, chunk_wav in enumerate(chunk_wavs):
                chunk_name = file_name + '_' + str(i+1) + '.wav'
                chunk_wav.export(os.path.join(s_train_path, chunk_name), format='wav')
                chunk_wav.export(os.path.join(s_validation_path, chunk_name), format='wav')
        
        else:
            num_validation = max(1, floor(len(files) * VALIDATION_DISTR))

            shuffle(files)

            train_files = files[:-num_validation]
            validation_files = files[-num_validation:]

            for file in train_files:
                chunk_wavs = [chunk[1] for chunk in chunks if chunk[0] == file]

                for i, chunk_wav in enumerate(chunk_wavs):
                    chunk_name = file + '_' + str(i+1) + '.wav'
                    chunk_wav.export(os.path.join(s_train_path, chunk_name), format='wav')

            for file in validation_files:
                chunk_wavs = [chunk[1] for chunk in chunks if chunk[0] == file]

                for i, chunk_wav in enumerate(chunk_wavs):
                    chunk_name = file + '_' + str(i+1) + '.wav'
                    chunk_wav.export(os.path.join(s_validation_path, chunk_name), format='wav')


if __name__ == "__main__":
    gen_raw_chunks()
    distribute_raw_chunks()