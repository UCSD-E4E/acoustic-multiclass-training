"""Generates binary annotations using TweetyNet from weakly labeled audio.
This file should be run from inside the PyHa directory. It also requires
WTS_chunking.py to be added to the PyHa directory. 
Input:     A folder with audio files
Output:    A csv with chunked, strongly-labeled annotations
"""

import os
import sys
from math import ceil
import pandas as pd
from pydub import AudioSegment
# pylint: disable=import-error #this file gets put into PyHa
from PyHa.IsoAutio import generate_automated_labels
from WTS_chunking import dynamic_yan_chunking

ISOLATION_PARAMETERS = {
    "model" : "tweetynet",
    "tweety_output": True,
    "verbose" : True
}

CHUNK_DURATION = 5
# incoming filetype
FILETYPE = ".mp3"
# use sliding window or raw chunks
SLIDING_CHUNKS = False

# weak annotations
METADATA_PATH = "~/metadata.csv"
# output for strong annotations
STRONG_LABELS_CSV = "~/132PeruXC_Labels.csv"
# output for chunked annotations
CHUNKED_CSV = "~/132PeruXC_RawChunks.csv"

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
    """Generate binary automated time-specific labels using PyHa

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

    metadata_df = pd.read_csv(METADATA_PATH)

    files = [f.path for f in os.scandir(path) if f.path.endswith(FILETYPE)]
    files.sort()
    for f in files:
        try:
            audio = AudioSegment.from_file(f)
        except RuntimeError as e:
            print("Audio conversion failed for ", filename + FILETYPE)
            print(e)
            continue
        filename = f.split('/')[-1][:-4]
        file_length = len(audio) # in ms
        num_chunks = ceil(file_length / (chunk_length))
        try: 
            scientific = metadata_df.loc[metadata_df["filename"] == (filename + FILETYPE), 'Scientific Name'].iloc[0]
            species = metadata_df.loc[metadata_df["filename"] == (filename + FILETYPE), 'Species eBird Code'].iloc[0]
        except IndexError as e:
            print("Scientific name or species lookup failed for ", filename + FILETYPE)
            print(e)
            continue
        for i in range(num_chunks):
            start = i * chunk_length
            end = start + chunk_length
            # cut off ending chunks that won't be 5s long
            if end > file_length:
                pass
            else:
                temp = [
                    species,
                    scientific,
                    f"{filename}_{i}{FILETYPE}",
                    path,
                    start / 1000,
                    chunk_duration,
                    file_length
                ]
                chunked_df.append(temp)
    return pd.DataFrame(chunked_df, columns=["SPECIES", "SCIENTIFIC", 
                                             "FILE NAME", "FOLDER", 
                                             "OFFSET", "DURATION", "CLIP LENGTH"])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect number of args", file=sys.stderr)
        print("USAGE: python gen_csv_labels.py /path", file=sys.stderr)
        sys.exit(1)
    
    filepath = sys.argv[1]
    if SLIDING_CHUNKS:
        convert2wav(filepath)
        generate_labels(filepath).to_csv(STRONG_LABELS_CSV)
        attach_labels().to_csv(STRONG_LABELS_CSV)
        generate_sliding_chunks().to_csv(CHUNKED_CSV)
    else:
        generate_raw_chunks(filepath, CHUNK_DURATION).to_csv(CHUNKED_CSV)
