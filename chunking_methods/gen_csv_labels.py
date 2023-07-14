"""Generates binary annotations using TweetyNet from weakly labeled audio.
This file should be run from inside the PyHa directory. It also requires
WTS_chunking.py to be added to the PyHa directory. 
Input:     A path to a folder with audio files
Output:    A csv with chunked, strongly-labeled annotations
"""

import os
import sys
from math import ceil

import pandas as pd
from config import get_config
from pydub import AudioSegment, exceptions
# pylint: disable=import-error #this file gets put into PyHa
from PyHa.IsoAutio import generate_automated_labels
from WTS_chunking import dynamic_yan_chunking

# This could be changed to use Microfaune or BirdNET, but the parameters are
# somewhat different and TweetyNet should be the default.
ISOLATION_PARAMETERS = {
    "model" : "tweetynet",
    "tweety_output": True,
    "verbose" : True
}


def convert_audio(path, filetype=".wav"):
    """Convert audio files to .wav files with PyDub. Used to ensure
    that TweetyNet can read the files for predictions.
    Args:
        path (string)
            - Path to folder containing audio files without subfolders
        filetype (str)
            - File extension for incoming audio files
    """
    # conversion not needed for tweetynet processing
    if filetype in [".wav", ".mp3"]:
        print(f"Conversion from {filetype} not required for TweetyNet processing")
        return
    for file in os.listdir(path):
        if file.endswith(filetype):
            x = AudioSegment.from_file(os.path.join(path, file))
            x.export(file.replace(filetype, '.wav'), format='wav')

def generate_labels(path):
    """Generate binary automated time-specific labels using TweetyNet as 
    implemented in PyHa.
    Args:
        path (string)
            - Path to folder containing audio files with at most one 
            subdirectory level
    Returns a PyHa-formatted DataFrame
    """
    if not os.path.exists(os.path.join(path)):
        print(f"Directory not found in path {path}", file=sys.stderr)
        sys.exit(1)

    # generate labels at a top level
    convert_audio(path)
    automated_df = generate_automated_labels(path, ISOLATION_PARAMETERS)

    # check one-level deep in case files organized by class
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    if subfolders:
        subfolders.sort()
        for s in subfolders:
            convert_audio(path)
            temp_df = generate_automated_labels(s, ISOLATION_PARAMETERS)
            if temp_df.empty:
                continue
            automated_df = pd.concat([automated_df, temp_df], ignore_index=True, sort=False)
    
    if automated_df.empty:
        print("no labels generated")
    
    return automated_df

def attach_labels(metadata, strong_labels):
    """ Attach the primary label from original metadata as a strong label
    for each chunk and reformat the columns for the training pipeline.
    Args:
        metadata (str)
            - Path to .csv with original audio clip information. Assumes 
            Xeno-canto formatting.
        strong_labels (str)
            - Path to .csv with time-specific labels. Assumes PyHa formatting.  
    Returns a DataFrame with minimum required columns
    """
    metadata_df = pd.read_csv(metadata)
    binary_df = pd.read_csv(strong_labels)
    strong_df = metadata_df.merge(binary_df, left_on="filename", right_on="IN FILE")
    strong_df = strong_df[["Species eBird Code", "Scientific Name", "IN FILE", "FOLDER", "OFFSET", 
                           "DURATION", "CLIP LENGTH"]]
    strong_df = strong_df.rename(columns={"IN FILE": "FILE NAME",
                                          "Species eBird Code": "SPECIES",
                                          "Scientific Name": "SCIENTIFIC"})
    return strong_df

def generate_sliding_chunks(strong_labels, chunk_duration=5):
    """Creates sliding window chunks out of previously made annotations to 
    create more data for training and better capture calls.
    Args: 
        strong_labels (str)
            - Path to .csv with time-specific labels.
        chunk_duration (int)
            - Length of desired file chunks
        
    Returns a DataFrame with sliding window chunked annotations
    """
    unchunked_df = pd.read_csv(strong_labels)
    return dynamic_yan_chunking(unchunked_df, chunk_duration=chunk_duration, only_slide=False)

def generate_raw_chunks(path, metadata, chunk_duration=5, filetype=".wav"):
    """Create simple chunks by dividing the file into equal length
    segments. Used as a baseline comparison to PyHa's pseudo-labeling.
    Args:
        path (string)
            - Path to folder containing audio files
        metadata (string)
            - Path to .csv with original audio clip information. Assumes 
            Xeno-canto formatting.
        chunk_duration (int)
            - Length of desired file chunks
        filetype (string)
            - File extension for incoming audio files
    Returns a DataFrame with end-to-end chunked annotations
    """  
    chunked_df = []
    chunk_length = chunk_duration * 1000

    metadata_df = pd.read_csv(metadata)

    files = [f.path for f in os.scandir(path) if f.path.endswith(filetype)]
    files.sort()
    for f in files:
        try:
            audio = AudioSegment.from_file(f)
        except exceptions.CouldntDecodeError as e:
            # catch ffmpeg error
            print("Audio conversion failed for ", filename + filetype)
            print(e)
            continue
        
        basepath = os.path.splitext(os.path.basename(f))[0] # only want basepath
        filename = basepath.split('.')[0]
        file_length = len(audio) # in ms
        num_chunks = ceil(file_length / (chunk_length))

        # attempt to match file with scientific name and ebird code
        try: 
            scientific = metadata_df.loc[metadata_df["filename"] == (filename + filetype),
                                         'Scientific Name'].iloc[0]
            species = metadata_df.loc[metadata_df["filename"] == (filename + filetype),
                                      'Species eBird Code'].iloc[0]
        except IndexError as e:
            print("Scientific name or species lookup failed for ", filename + filetype)
            print(e)
            continue
        
        # create chunks and add to dataframe
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
                    f"{filename}{filetype}",
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
    CONFIG = get_config()

    if CONFIG.sliding_chunks:
        print("converting audio...?")
        convert_audio(
            path=CONFIG.data_path,
            filetype=CONFIG.filetype)
        # saved to csv in case attaching labels fails as generating labels takes more time
        print("generating labels...")
        generate_labels(CONFIG.data_path).to_csv(CONFIG.strong_labels)
        attach_labels(
            metadata=CONFIG.metadata,
            strong_labels=CONFIG.strong_labels).to_csv(CONFIG.strong_labels)
        print("generating sliding chunks...")
        generate_sliding_chunks(
            strong_labels=CONFIG.strong_labels,
            chunk_duration=CONFIG.chunk_duration).to_csv(CONFIG.chunk_path)
    else:
        print("generating raw chunks...")
        generate_raw_chunks(
            path=CONFIG.data_path,
            metadata=CONFIG.metadata,
            chunk_duration=CONFIG.chunk_duration,
            filetype=CONFIG.filetype).to_csv(CONFIG.chunk_path)
