import os
import pydub
import pandas as pd
from WTS_chunking import *


path = '/share/acoustic_species_id'
unchunked_df = pd.read_csv(os.path.join(path, 'BirdCLEF2023_TweetyNet_StrongLabels.csv'))
chunk_path = os.path.join(path, 'BirdCLEF2023_train_audio_chunks')

def generate_chunked_df(unchunked_df, save_chunks=True):
  chunked_df = dynamic_yan_chunking(unchunked_df, chunk_count=5, chunk_duration=5, only_slide=False)
  if save_chunks:
      chunked_df.to_csv(os.path.join(path, 'BirdCLEF2023_TweetyNet_Chunks.csv'))
  return chunked_df


def generate_wavs_from_chunks(chunk_duration):
  
  if not os.path.exists(chunk_path):
     os.makedirs(chunk_path)

  chunked_df = pd.read_csv(os.path.join(path, 'BirdCLEF2023_TweetyNet_Chunks.csv'))
  file_name = ''
  label = ''
  folder_path = ''
  wav_file = None
  chunk_count = 0
  chunk_duration *= 1000

  test_count = 0

  for _, row in chunked_df.iterrows():
     
    if row['STRONG LABEL'] != label:
        
      label = row['STRONG LABEL']
      folder_path = os.path.join(chunk_path, label)

      if not os.path.exists(folder_path):
        os.makedirs(folder_path)
      
      test_count += 1

    if row['IN FILE'] != file_name:
        
      file_name = row['IN FILE']
      wave_file_path = os.path.join(path, 'BirdCLEF2023_train_audio', label, file_name)
      wav_file = pydub.AudioSegment.from_wav(wave_file_path)
      chunk_count = 1

    # splice wav file and save chunk
    offset = float(row['OFFSET']) * 1000 # pydub splices in milliseconds, so multiply by 1000
    

    chunk = wav_file[offset : offset + chunk_duration]

    try:
      assert len(chunk) == chunk_duration, f"Chunk of length {chunk_duration / 1000}s could not be generated from {file_name}. \n \
                                             Got chunk of length {len(chunk) / 1000}s. Check chunking script."
      chunk.export(os.path.join(folder_path, file_name[:-4] + '_' + str(chunk_count) + '.wav'), format='wav')
    except AssertionError as e:
      print(e)

    chunk_count += 1


def delete_chunks_with_len(length):
  length *= 1000
  subfolders = [f.path for f in os.scandir(chunk_path) if f.is_dir() ]   
  for subfolder in subfolders:
    chunks = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]
    for chunk in chunks:
      wav_file_path = os.path.join(subfolder, chunk)
      wav_file = pydub.AudioSegment.from_wav(wav_file_path)
      if len(wav_file) == length:
        os.remove(wav_file_path)


      


if __name__ == '__main__':
  # generate_chunked_df(unchunked_df)
  # generate_wavs_from_chunks(5)
  delete_chunks_with_len(3)