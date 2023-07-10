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

def gen_labels(path):
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
                           "DURATION", "CLIP LENGTH"]]
    strong_df = strong_df.rename(columns={"Species eBird Code": "SPECIES",
                                          "Scientific Name": "SCIENTIFIC"})
    strong_df.to_csv(STRONG_LABELS_CSV)

def generate_sliding_chunks():
    """Return a dataframe with sliding window chunked annotations
    """
    unchunked_df = pd.read_csv(STRONG_LABELS_CSV)
    chunked_df = dynamic_yan_chunking(unchunked_df, chunk_duration=5, only_slide=False)
    chunked_df.to_csv(CHUNKED_CSV)

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
    if SLIDING_CHUNKS:
        gen_labels(path)
        attach_labels(path)
        generate_sliding_chunks()
    else:
        convert2wav(path)
        generate_raw_chunks(path)