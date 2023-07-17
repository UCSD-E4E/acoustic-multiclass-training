"""Generates binary annotations using TweetyNet from weakly labeled audio.
This file should be run from inside the PyHa directory. It also requires
config.py and WTS_Chunking.py to be added to the PyHa directory. 
Input:     A path to a folder with audio files
Output:    A csv with chunked, strongly-labeled annotations
"""
import sys
from math import ceil
from pathlib import Path

import pandas as pd
from chunks_config import get_config
from pydub import AudioSegment, exceptions
# pylint: disable=import-error #this file gets put into PyHa
# pylint: disable=no-name-in-module
from PyHa.IsoAutio import generate_automated_labels  # pyright: ignore
from sliding_chunks import dynamic_yan_chunking

# This could be changed to use Microfaune or BirdNET, but the parameters are
# somewhat different and TweetyNet should be the default.
ISOLATION_PARAMETERS = {
    "model" : "tweetynet",
    "tweety_output": True,
    "verbose" : True
}

def convert_audio(directory: str, filetype: str) -> None:
    """Convert audio files to .wav files with PyDub. Used to ensure
    that TweetyNet can read the files for predictions.
    Args:
        directory (str)
            - Path to folder containing audio files
        filetype (str)
            - File extension for incoming audio files
    Returns:
        None
    """
    # conversion not needed for tweetynet processing
    if filetype in [".wav", ".mp3"]:
        print(f'Conversion from {filetype} not required for TweetyNet processing')
        return
    print(f'Converting audio for {directory}')
    file_list = [f for f in Path(directory).glob('**/*') if f.is_file()]
    for path in file_list:
        if path.suffix == filetype:
            audio = AudioSegment.from_file(path)
            audio.export(path.with_suffix('.wav'), format='wav')
        
def generate_labels(path: str) -> None:
    """Generate binary automated time-specific labels using TweetyNet as 
    implemented in PyHa.
    Args:
        path (str)
            - Path to folder containing audio files with at most one 
            subdirectory level
    Returns:
        PyHa-formatted DataFrame
    """
    rootdir = Path(path)
    if not rootdir.is_dir():
        print(f'Directory not found in path {path}', file=sys.stderr)
        sys.exit(1)

    # generate labels at a top level
    automated_df = generate_automated_labels(path, ISOLATION_PARAMETERS)

    # check subdirectories in case files organized by class
    subfolders = [str(f) for f in rootdir.rglob('*') if f.is_dir()]
    for folder in sorted(subfolders):
        temp_df = generate_automated_labels(folder, ISOLATION_PARAMETERS)
        if temp_df.empty:
            continue
        automated_df = pd.concat([automated_df, temp_df], ignore_index=True, sort=False)
    
    if automated_df.empty:
        print('No labels generated')
    
    return automated_df

def attach_labels(metadata_df: pd.DataFrame, binary_df: pd.DataFrame) -> pd.DataFrame:
    """ Attach the primary label from original metadata as a strong label
    for each chunk and reformat the columns for the training pipeline.
    Args:
        metadata_df (DataFrame)
            - DataFrame with original audio clip information. Assumes 
            Xeno-canto formatting.
        binary_df (DataFrame)
            - DataFrame with time-specific labels. Assumes PyHa formatting.  
    Returns:
        DataFrame with minimum required columns for training
    """
    if 'filename' not in metadata_df.columns:
        raise KeyError("This function merges .csvs on filename. Check your metadata columns!")
    strong_df = metadata_df.merge(binary_df, left_on='filename', right_on='IN FILE')
    strong_df = strong_df[['Species eBird Code',
                           'Scientific Name',
                           'IN FILE',
                           'FOLDER',
                           'OFFSET',
                           'DURATION',
                           'CLIP LENGTH']]
    strong_df = strong_df.rename(columns={'IN FILE': 'FILE NAME',
                                          'Species eBird Code': 'SPECIES',
                                          'Scientific Name': 'SCIENTIFIC'})
    return strong_df

def generate_sliding_chunks(strong_df: pd.DataFrame, chunk_length_s: int=5) -> pd.DataFrame:
    """Wrapper function. Creates sliding window chunks out of previously made annotations
    to make more training data and better capture calls.
    Args: 
        strong_df (DataFrame)
            - DataFrame with time-specific labels
        chunk_length_s (int)
            - Length of desired file chunks in seconds
        
    Returns a DataFrame with sliding window chunked annotations
    """
    return dynamic_yan_chunking(strong_df, chunk_length_s, only_slide=False)

def generate_raw_chunks(directory: str, metadata_df: pd.DataFrame, chunk_length_s: int=5,
                        filetype: str='.wav') -> pd.DataFrame:
    """Create simple chunks by dividing the file into equal length
    segments. Used as a baseline comparison to PyHa's pseudo-labeling.
    Args:
        directory (str)
            - Path to folder containing audio files
        metadata_df (DataFrame)
            - DataFrame original audio clip information. Assumes 
            Xeno-canto formatting.
        chunk_length_s (int)
            - Length of desired file chunks in seconds
        filetype (str)
            - File extension for incoming audio files
    Returns a DataFrame with end-to-end chunked annotations
    """  
    if 'filename' not in metadata_df.columns:
        raise KeyError("This function merges .csvs on filename. Check your metadata columns!")
    chunks = []
    chunk_length_ms = chunk_length_s * 1000
    file_list = [f for f in Path(directory).glob('**/*') if f.is_file() and f.suffix == filetype]
    for path in sorted(file_list):
        try:
            audio = AudioSegment.from_file(path)
        except exceptions.CouldntDecodeError as ex:
            # catch ffmpeg error
            print('Audio conversion failed for ', path)
            print(ex)
            continue
    
        file_length_ms = len(audio)
        num_chunks = ceil(file_length_ms / (chunk_length_ms))

        # attempt to match file with scientific name and ebird code
        try:
            row = metadata_df.loc[metadata_df['filename'] == path.name]
            scientific = row['Scientific Name'].iloc[0]
            species = row['Species eBird Code'].iloc[0]
        except IndexError as ex:
            print('Scientific name or species lookup failed for ', path.name)
            print(ex)
            continue
        
        # create chunks and add to dataframe
        for i in range(num_chunks):
            start = i * chunk_length_ms
            end = start + chunk_length_ms
            # cut off ending chunks that won't be 5s long
            if end <= file_length_ms:
                temp = {
                    'SPECIES' : species,
                    'SCIENTIFIC' : scientific,
                    'FILE NAME' : path.name,
                    'FOLDER' : path.parent,
                    'OFFSET' : start / 1000,
                    'DURATION' : chunk_length_s,
                    'CLIP LENGTH' : file_length_ms
                }
                chunks.append(temp)
    return pd.DataFrame(chunks)

def main():
    """Generates binary annotations using TweetyNet from weakly labeled audio.
    Args:
        None
    Returns:
        None
    """
    cfg = get_config()
    metadata = pd.read_csv(cfg.metadata)

    if cfg.sliding_window:
        # saved to csv in case attaching labels fails as generating labels takes more time
        print('Generating labels...')
        convert_audio(cfg.audio_path, cfg.filetype)
        labels = generate_labels(cfg.audio_path)
        labels.to_csv(cfg.strong_labels)

        print('Attaching strong labels...')
        strong_labels = attach_labels(metadata, labels)
        strong_labels.to_csv(cfg.strong_labels)
        
        print('Generating sliding chunks...')
        chunks_df = generate_sliding_chunks(strong_labels, cfg.chunk_length_s)
        chunks_df.to_csv(cfg.chunk_labels)
    else:
        print('Generating raw chunks...')
        chunks_df = generate_raw_chunks(
            directory=cfg.audio_path,
            metadata_df=metadata,
            chunk_length_s=cfg.chunk_length_s,
            filetype=cfg.filetype)
        chunks_df.to_csv(cfg.chunk_labels)
    print("Wrote chunks to", cfg.chunk_labels)

if __name__=="__main__":
    main()
