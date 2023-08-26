""" Splits longer audio files into smaller ones """

import os
from typing import List
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

CONFIG = {
    "metadata_csv": "annotations.csv",
    "metadata_output": "annotations_split.csv",

    "audio_dir": "input",
    "sample_rate": "error", # Only use if input format is pt
    "output_dir": "output",
    "output_format": "flac", # Supports torch audio formats

    "chunk_length_s": 60 * 5, # Length of each clip in seconds

    "file_name_col": "FILE NAME",
    "offset_col": "OFFSET",
    "duration_col": "DURATION"

}

def output_file_name(path: str, index: int, file_format: str) -> str:
    """ Returns the output file name for a given input file name and index """
    return os.path.basename(path).split('.')[0] + "_" + str(index) + "." + file_format

def split_audio_file(path: str):
    """ Splits audio file into smaller chunks """
    split_len = CONFIG["chunk_length_s"]
    
    # Load audio file
    if path.endswith(".pt"):
        audio = torch.load(path)
        sample_rate = CONFIG["sample_rate"]
    else:
        audio, sample_rate = torchaudio.load(path) # type: ignore
        audio = audio[0]
    
    file_len = len(audio)/float(sample_rate)
    num_splits = int(file_len / split_len)
    
    for i in range(num_splits):
        # Create slice
        aud_slice = audio[i*split_len*sample_rate:((i+1)*split_len+CONFIG["overlap_s"])*sample_rate]
        torchaudio.save(os.path.join(CONFIG["output_dir"], # type: ignore
                                     output_file_name(path,i,CONFIG["output_format"])),
                        torch.unsqueeze(aud_slice,0), sample_rate)

def edit_row(row: pd.Series) -> List[pd.Series]:
    """ Edits a row of the metadata csv to reflect the new audio files
    Changes file name and offset
    """
    chunk_len = CONFIG["chunk_length_s"]
    file_index = int(row[CONFIG["offset_col"]]/chunk_len)
    # Update file name
    cur_file_name = str(row[CONFIG["file_name_col"]])
    row[CONFIG["file_name_col"]] = \
        output_file_name(cur_file_name, file_index, CONFIG["output_format"])
    # Shift offset
    end = row[CONFIG["offset_col"]] + row[CONFIG["duration_col"]]
    row[CONFIG["offset_col"]] -= file_index * chunk_len
    row[CONFIG["duration_col"]] = min(chunk_len - row[CONFIG["offset_col"]], row[CONFIG["duration_col"]])
    row["CLIP LENGTH"] = chunk_len
    out = [row]
    while end > (file_index+1) * chunk_len:
        file_index += 1 
        row_cpy = row.copy()
        row_cpy[CONFIG["file_name_col"]] = \
            output_file_name(cur_file_name, file_index, CONFIG["output_format"])
        row_cpy[CONFIG["offset_col"]] = 0
        row_cpy[CONFIG["duration_col"]] = min(chunk_len, end - file_index * chunk_len)
        out.append(row_cpy)
    return out

def edit_metadata(df: pd.DataFrame):
    """ Edits metadata to reflect the new audio files """
    out_list = []
    for _, row in df.iterrows():
        out_list.extend(edit_row(row))
    out = pd.DataFrame(out_list)
    out.reset_index(drop=True,inplace=True)
    return out

def split_all(input_dir: str):
    """ Splits all audio files in the input directory """
    input_dir = os.path.abspath(input_dir)
    for path in tqdm(os.listdir(input_dir)):
        audio_path = os.path.join(input_dir, path)
        split_audio_file(audio_path)

def main():
    """ Main function """
    df = pd.read_csv(CONFIG["metadata_csv"], index_col=0)
    #split_all(CONFIG["audio_dir"])
    df = edit_metadata(df)
    df.to_csv(CONFIG["metadata_output"])

if __name__ == "__main__":
    main()
