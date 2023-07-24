""" Splits longer audio files into smaller ones """

CONFIG = {
    "metadata_csv": "annotations_chunked.csv",
    "metadata_output": "annotations_split.csv",

    "audio_dir": "input",
    "sample_rate": "error", # Only use if input format is pt
    "output_dir": "output",
    "output_format": "flac", # Supports "wav" or "pt"

    "chunk_length_s": 60 * 5, # Length of each clip in seconds
    "overlap_s": 10, # Overlap to add to each file in seconds

    "file_name_col": "FILE NAME",
    "offset_col": "OFFSET",

}

import os
import pandas as pd
import torch
import torchaudio

def output_file_name(path: str, index: int, format: str) -> str:
    """ Returns the output file name for a given input file name and index """
    return os.path.basename(path).split('.')[0] + "_" + str(index) + "." + format

def split_audio_file(path: str):
    """ Splits audio file into smaller chunks """
    print("splitting", path)
    split_len = CONFIG["chunk_length_s"]
    
    # Load audio file
    if path.endswith(".pt"):
        audio = torch.load(path)
        sr = CONFIG["sample_rate"]
    else:
        audio, sr = torchaudio.load(path)
        audio = audio[0]
    
    file_len = len(audio)/float(sr)
    num_splits = int(file_len / split_len)
    
    for i in range(num_splits):
        # Create slice
        slice = audio[i*split_len*sr:((i+1)*split_len+CONFIG["overlap_s"])*sr]
        if CONFIG["output_format"] == "pt":
            torch.save(slice, os.path.join(CONFIG["output_dir"], output_file_name(path, i, "pt")))
        else:
            torchaudio.save(os.path.join(CONFIG["output_dir"],output_file_name(path,i,CONFIG["output_format"])),torch.unsqueeze(slice,0), sr)

def edit_row(row: pd.Series) -> pd.Series:
    """ Edits a row of the metadata csv to reflect the new audio files
    Changes file name and offset
    """
    offset = row[CONFIG["offset_col"]]
    file_index = int(offset/CONFIG["chunk_length_s"])
    # Update file name
    row[CONFIG["file_name_col"]] = \
        output_file_name(row[CONFIG["file_name_col"]], file_index, CONFIG["output_format"])
    # Shift offset
    row[CONFIG["offset_col"]] -= file_index * CONFIG["chunk_length_s"]
    return row

def edit_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """ Edits metadata to reflect the new audio files """
    return df.apply(edit_row, axis=1)

def split_all(input_dir: str):
    """ Splits all audio files in the input directory """
    input_dir = os.path.abspath(input_dir)
    for path in os.listdir(input_dir):
        audio_path = os.path.join(input_dir, path)
        split_audio_file(audio_path)

def verify_config():
    """ Verify that the config is correct """
    #if CONFIG["output_format"]!="wav" and CONFIG["output_format"]!="pt":
        #raise ValueError("Output format must be wav or pt")

if __name__ == "__main__":
    verify_config()
    df = pd.read_csv(CONFIG["metadata_csv"], index_col=0)
    split_all(CONFIG["audio_dir"])
    edit_metadata(df)
    df.to_csv(CONFIG["metadata_output"])