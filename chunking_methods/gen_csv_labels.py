"""Generates binary annotations using TweetyNet from unlabeled audio.
This file should be run from inside the PyHa directory.
Inputs:     A folder with audio files
Outputs:    A csv with chunked, strongly-labeled annotations
"""

## TODO: rewrite with pathlib

import os
import sys
import pandas as pd
from pydub import AudioSegment
from PyHa.IsoAutio import generate_automated_labels
from WTS_chunking import dynamic_yan_chunking

ISOLATION_PARAMETERS = {
    "model" : "tweetynet",
    "tweety_output": True,
    "verbose" : True
}

FILETYPE = ".mp3"
SLIDING_CHUNKS = True
GENERATE_WAVS = True
CHUNK_DURATION = 5
METADATA_PATH = "/home/sprestrelski/amabaw1/metadata.csv"
STRONG_LABELS_CSV = "/home/sprestrelski/amabaw1/test.csv"
CHUNKED_CSV = "/home/sprestrelski/amabaw1/chunks.csv"

def convert2wav(path):
    """Convert audio files to .wav files with PyDub
    
    Args:
        path (string)
            - folder path containing audio files without subfolders
    """
    # conversion not needed for tweetynet processing
    if SLIDING_CHUNKS and FILETYPE in [".wav", ".mp3"]:
        return
    # conversion needed for generating raw chunks
    if not SLIDING_CHUNKS and FILETYPE == ".wav":
        return
    for fn in os.listdir(path):
        if fn.endswith(FILETYPE):
            x = AudioSegment.from_file(os.path.join(path, fn))
            x.export(fn.replace(FILETYPE, '.wav'), format='wav')

def generate_labels(path):
    """Generate binary automated time-specific labels using PyHa

    Args:
        path (string)
            - folder path containing audio files with at most one 
            subdirectory level
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
    else:
        automated_df.to_csv(STRONG_LABELS_CSV)

def attach_labels():
    """Add the primary label from original metadata as a strong label > bird
    """
    metadata_df = pd.read_csv(METADATA_PATH)
    binary_df = pd.read_csv(STRONG_LABELS_CSV)
    strong_df = metadata_df.merge(binary_df, left_on="filename", right_on="IN FILE")
    strong_df = strong_df[["Species eBird Code", "Scientific Name", "IN FILE", "FOLDER", "OFFSET", 
                           "DURATION"]]
    strong_df = strong_df.rename(columns={"IN FILE": "FILE NAME",
                                          "Species eBird Code": "SPECIES",
                                          "Scientific Name": "SCIENTIFIC"})
    strong_df.to_csv(STRONG_LABELS_CSV)

def generate_sliding_chunks():
    """Return a dataframe with sliding window chunked annotations
    """
    unchunked_df = pd.read_csv(STRONG_LABELS_CSV)
    chunked_df = dynamic_yan_chunking(unchunked_df, chunk_duration=CHUNK_DURATION, only_slide=False)
    chunked_df.to_csv(CHUNKED_CSV)

def generate_wavs_from_labels(path, chunk_duration):
    """Create wav files based on a .csv with annotations
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
    test_count = 0

    for _, row in chunked_df.iterrows():
        # make new folder for each species
        if row['SPECIES'] != label:
            label = row['SPECIES']
            folder_path = os.path.join(chunk_path, label)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            test_count += 1
        
        # access the original file
        if row['FILE NAME'] != file_name:
            file_name = row['FILE NAME']
            wave_file_path = os.path.join(path, label, file_name)
            wav_file = pydub.AudioSegment.from_wav(wave_file_path)
            chunk_count = 1

        # splice wav file and save chunk
        offset = float(row['OFFSET']) * 1000 # pydub splices in milliseconds, so multiply by 1000
        chunk = wav_file[offset : offset + chunk_duration]

        try:
            # pylint: disable=line-too-long, just an assertion
            assert len(chunk) == chunk_duration, f"Chunk of length {chunk_duration / 1000}s could not be generated from {file_name}. \
                \n Got chunk of length {len(chunk) / 1000}s. Check chunking script."
            chunk.export(os.path.join(folder_path, file_name[:-4] + '_' + str(chunk_count) + '.wav'), format='wav')
        except AssertionError as e:
            print(e)
        chunk_count += 1

def generate_raw_chunks(path):
    """Create 5 second chunks from a wav file
    """  

    # find all files 
    # for each, split into 5 seconds and add a row for the chunk
    # return dataframe
    chunked_df = pd.DataFrame(columns=["IN FILE", "FOLDER", "OFFSET", "DURATION", "CLIP LENGTH"])
    
    
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    for s in subfolders:
        wavs = [f.path for f in os.scandir(s) if f.path.endswith('.wav')]
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect number of args", file=sys.stderr)
        print("USAGE: python gen_labels_csv.py /path", file=sys.stderr)
        sys.exit(1)
    
    path = sys.argv[1]
    # if SLIDING_CHUNKS:
    #     generate_labels(path)
    #     attach_labels()
    #     generate_sliding_chunks()
    # else:
    #     convert2wav(path)
    #     generate_raw_chunks(path)
    
    if GENERATE_WAVS:
        generate_wavs_from_labels(path, CHUNK_DURATION)