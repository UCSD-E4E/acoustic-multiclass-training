import os
import pydub

from math import ceil, floor
from random import shuffle

VALIDATION_DISTR = 0.2

wav_path = '/share/acoustic_species_id/BirdCLEF2023_train_audio'
chunk_path = '/share/acoustic_species_id/BirdCLEF2023_raw_chunks'

chunk_dict = {}

def gen_raw_chunks(pad=False):
    
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
    
    train_path = os.path.join(chunk_path, 'training')
    validation_path = os.path.join(chunk_path, 'validation')
    
    for species in chunk_dict:
        
        chunks = chunk_dict[species]

        files = list(set([chunk[0] for chunk in chunks]))

        s_train_path = os.path.join(train_path, species)
        s_validation_path = os.path.join(validation_path, species)

        if len(files) == 0:
            continue        
        
        if not os.path.exists(s_train_path):
          os.makedirs(s_train_path)
        if not os.path.exists(s_validation_path):
            os.makedirs(s_validation_path)

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

