"""Create a single annotation for each clip that covers the entire duration"""

import pandas as pd

CONFIG = {
    "input_csv" : "pyha_annotations.csv",
    "output_csv" : "all_interest.csv",

    "clip_length_col" : "CLIP LENGTH",
    "duration_col" : "DURATION",
    "file_name_col": "FILE NAME",
    "offset_col": "OFFSET",
    
}

def all_interest():
    df = pd.read_csv(CONFIG["input_csv"], index_col=0)
    df = df.drop_duplicates(subset=[CONFIG["file_name_col"]],keep="first")
    df[CONFIG["offset_col"]] = 0
    df[CONFIG["duration_col"]] = df[CONFIG["clip_length_col"]]
    df.to_csv(CONFIG["output_csv"])

    
if __name__ == "__main__":
    all_interest()
