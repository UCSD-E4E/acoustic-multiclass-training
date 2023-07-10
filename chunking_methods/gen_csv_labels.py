"""Generates binary annotations using TweetyNet from weakly labeled audio.
This file should be run from inside the PyHa directory. It also requires
WTS_chunking.py to be added to the PyHa directory. 
Input:     A folder with audio files
Output:    A csv with chunked, strongly-labeled annotations
"""

## TODO: rewrite with pathlib

import os
import sys
import pandas as pd
from math import ceil
from pydub import AudioSegment
from PyHa.IsoAutio import generate_automated_labels
from WTS_chunking import dynamic_yan_chunking

ISOLATION_PARAMETERS = {
    "model" : "tweetynet",
    "tweety_output": True,
    "verbose" : True
}

CHUNK_DURATION = 5
# produce a wav for each chunk
GENERATE_WAVS = False
# incoming filetype
FILETYPE = ".mp3"
# use sliding window or raw chunks
SLIDING_CHUNKS = False

# weak annotations
METADATA_PATH = "/home/sprestrelski/amabaw1/metadata.csv"
# output for strong annotations
STRONG_LABELS_CSV = "/home/sprestrelski/amabaw1/strong_labels.csv"
# output for chunked annotations
CHUNKED_CSV = "/home/sprestrelski/amabaw1/raw-chunks.csv"

def convert2wav(path):
    """Convert audio files to .wav files with PyDub. Used to ensure
    that TweetyNet can read the files for predictions.
    Args:
        path (string)
            - folder path containing audio files without subfolders
    """
    # conversion not needed for tweetynet processing
    if SLIDING_CHUNKS and FILETYPE in [".wav", ".mp3"]:
        return
    for fn in os.listdir(path):
        if fn.endswith(FILETYPE):
            x = AudioSegment.from_file(os.path.join(path, fn))
            x.export(fn.replace(FILETYPE, '.wav'), format='wav')

def generate_labels(path):
    """Generate binary time-specific labels using PyHa
    Args:
        path (string)
            - folder path containing audio files with at most one 
            subdirectory level
    Returns a PyHa-formatted DataFrame
    """
    if not os.path.exists(os.path.join(path)):
        print(f"Directory not found in path {path}", file=sys.stderr)
        sys.exit(1)

    # generate labels at a top level
    convert2wav(path)
    automated_df = generate_automated_labels(path, ISOLATION_PARAMETERS)

    # check one-level deep in case files organized by class
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    if subfolders:
        subfolders.sort()
        for s in subfolders:
            convert2wav(path)
            temp_df = generate_automated_labels(s, ISOLATION_PARAMETERS)
            if temp_df.empty:
                continue
            automated_df = pd.concat([automated_df, temp_df], ignore_index=True, sort=False)
    
    if automated_df.empty:
        print("no labels generated")
    
    return automated_df

def attach_labels():
    """ Attach the primary label from original metadata as a strong label
    for each chunk and reformat the columns.
    Args:
        None
    Returns a stripped DataFrame with only columns necessary for training 
    """
    metadata_df = pd.read_csv(METADATA_PATH)
    binary_df = pd.read_csv(STRONG_LABELS_CSV)
    strong_df = metadata_df.merge(binary_df, left_on="filename", right_on="IN FILE")
    strong_df = strong_df[["Species eBird Code", "Scientific Name", "IN FILE", "FOLDER", "OFFSET", 
                           "DURATION", "CLIP LENGTH"]]
    strong_df = strong_df.rename(columns={"IN FILE": "FILE NAME",
                                          "Species eBird Code": "SPECIES",
                                          "Scientific Name": "SCIENTIFIC"})
    return strong_df

def generate_sliding_chunks():
    """
    Args: 
        None
    Returns a DataFrame with sliding window chunked annotations
    """
    unchunked_df = pd.read_csv(STRONG_LABELS_CSV)
    return dynamic_yan_chunking(unchunked_df, chunk_duration=CHUNK_DURATION, only_slide=False)

def generate_wavs_from_labels(path, chunk_duration):
    """Create wav files based on a .csv with annotations
    Args:
        path (string)
            - folder path containing audio files with at most one 
            subdirectory level
        chunk_duration (int)
            - length of desired file chunks
    """
    chunk_path = os.path.join(path, 'chunks')
    if not os.path.exists(chunk_path):
        os.makedirs(chunk_path)
    chunked_df = pd.read_csv(CHUNKED_CSV)
    file_name = ''
    label = ''
    folder_path = ''
    wav_file = None
    chunk_count = 0
    chunk_duration *= 1000

    for _, row in chunked_df.iterrows():
        # make new folder for each species
        if row['SPECIES'] != label:
            label = row['SPECIES']
            folder_path = os.path.join(chunk_path, label)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        
        # access the original file
        if row['FILE NAME'] != file_name:
            file_name = row['FILE NAME']
            wav_file_path = os.path.join(path, file_name)
            wav_file = AudioSegment.from_file(wav_file_path)
            chunk_count = 1

        # splice wav file and save chunk
        offset = float(row['OFFSET']) * 1000 # pydub splices in milliseconds, so multiply by 1000
        chunk = wav_file[offset : offset + chunk_duration]

        try:
            assert len(chunk) == chunk_duration, f"Chunk of length {chunk_duration / 1000}s could not be \
                generated from {file_name}. Got chunk of length {len(chunk) / 1000}s. Check chunking script."
            chunk.export(os.path.join(folder_path, file_name[:-4], '_', str(chunk_count), '.wav'), format='wav')
        except AssertionError as e:
            print(e)
        chunk_count += 1

def generate_raw_chunks(path, chunk_duration):
    """Create .csv annotations for specified second chunks
    Args:
        path (string)
            - folder path containing audio files with at most one 
            subdirectory level
        chunk_duration (int)
            - length of desired file chunks
    """  
    chunked_df = pd.DataFrame(columns=["FILE NAME", "FOLDER", "OFFSET", "DURATION", "CLIP LENGTH"])
    chunk_length = chunk_duration * 1000

    files = [f.path for f in os.scandir(path) if f.path.endswith(FILETYPE)]
    for f in files:
        audio = AudioSegment.from_file(f)
        filename = f.split('/')[-1][:-4]
        file_length = len(audio) # in ms
        num_chunks = ceil(file_length / (chunk_length))

        for i in range(num_chunks):
            start = i * chunk_length
            end = start + chunk_length

            # cut off ending chunks that won't be 5s long
            if end > file_length:
                pass
            else:
                temp_df = {"FILE NAME" : f"{filename}_{i}{FILETYPE}",
                           "FOLDER" : path,
                           "OFFSET" : start / 1000,
                           "DURATION": chunk_duration,
                           "CLIP LENGTH" : file_length}
                chunked_df = chunked_df.append(temp_df, ignore_index=True)  
    return chunked_df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect number of args", file=sys.stderr)
        print("USAGE: python gen_labels_csv.py /path", file=sys.stderr)
        sys.exit(1)
    
    path = sys.argv[1]
    if SLIDING_CHUNKS:
        convert2wav(path)
        generate_labels(path).to_csv(STRONG_LABELS_CSV)
        attach_labels().to_csv(STRONG_LABELS_CSV)
        generate_sliding_chunks().to_csv(CHUNKED_CSV)
    else:
        generate_raw_chunks(path, CHUNK_DURATION).to_csv(CHUNKED_CSV)

    if GENERATE_WAVS:
        generate_wavs_from_labels(path, CHUNK_DURATION)
