import os
import shutil
from random import shuffle
from math import floor


VALIDATION_DISTR = 0.2

share_path = '/share/acoustic_species_id'
wav_path = '/share/acoustic_species_id/BirdCLEF2023_split_audio'
chunk_path_old = '/share/acoustic_species_id/BirdCLEF2023_train_audio_chunks'
chunk_path_new = '/share/acoustic_species_id/BirdCLEF2023_split_chunks_new'


def distribute_files():
    
  file_path = os.path.join(share_path, 'BirdCLEF2023_train_audio')
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
      s_train_path = os.path.join(train_path, species)
      s_validation_path = os.path.join(validation_path, species)

      files = [f.path.split('/')[-1] for f in os.scandir(s) if f.path.endswith('.wav') and contains_chunks(f)]

      if len(files) == 0:
          continue

      if not os.path.exists(s_train_path):
          os.makedirs(s_train_path)
      if not os.path.exists(s_validation_path):
          os.makedirs(s_validation_path)

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
        
      


def distribute_chunks():
    
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


def clear_files(path):

  subfolders = [f.path for f in os.scandir(path) if f.is_dir()]

  for s in subfolders:
      
      subfolders_type = [f.path for f in os.scandir(s) if f.is_dir()]

      for s_type in subfolders_type:
      
        files = [f.path for f in os.scandir(s_type) if f.is_file()]

        for file in files:
            os.remove(file)



if __name__ == '__main__':
    clear_files(wav_path)
    clear_files(chunk_path_new)
    distribute_files()
    distribute_chunks()