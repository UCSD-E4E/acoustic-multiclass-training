"""
    Small script that can generally be used to clean up an audio metadata csv file
"""
import os
import pandas as pd

ARGUMENTS = {
    # INPUT
    "input_path": "../example_dataset/metadata.csv",
    "has_index_col": True,
    "file_name_col": "file_location",

    # PROCESSING
    # Specify these if there is a start and end time for the clip
    # Offset and duration will be calculated from these
    "start_time": "",
    "end_time": "",
    # Columns
    "column_renames": {
        "MANUAL ID": "Species eBird Code",
        "Scientific Name": "SCIENTIFIC",
        "Common Name": "COMMON",
        "Offset": "OFFSET",
        "Duration": "DURATION",
    },
    
    # OUTPUT
    "cols_to_save": [
        "FILE NAME",
        #"Species eBird Code",
        "OFFSET",
        "DURATION",
        "SCIENTIFIC",
        "COMMON"
    ],
    "output_path": "../example_dataset/metadata_cleaned.csv",
    
}

def main():
    """ Main function """
    if ARGUMENTS["input_path"] == "":
        raise ValueError("Input path not specified")
    
    if ARGUMENTS["has_index_col"]:
        df = pd.read_csv(ARGUMENTS["input_path"], index_col=0)
    else:
        df = pd.read_csv(ARGUMENTS["input_path"])
    # Rename columns
    col_renames = ARGUMENTS["column_renames"]
    for col in col_renames.items():
        if col in df.columns:
            df = df.rename(columns={col: col_renames[col]})
        elif col_renames[col] in df.columns:
            pass # Already renamed
        else:
            print("Warning: column", col, "not found in dataset")
    
    # Delete missing files
    df = df[df[ARGUMENTS["file_name_col"]].apply(lambda x: isinstance(x,str))]
    df = df[df[ARGUMENTS["file_name_col"]]!=""]
    
    # Fix file name column
    df["FILE NAME"] = df[ARGUMENTS["file_name_col"]].apply(os.path.basename)
    
    # Turn start and end to offset and duration
    if ARGUMENTS["start_time"] != "":
        df["OFFSET"] = df[ARGUMENTS["start_time"]]
        df["DURATION"] = df[ARGUMENTS["end_time"]] - df[ARGUMENTS["start_time"]]

    df = df.reset_index(drop=True)
    df = df[ARGUMENTS["cols_to_save"]]
    df.to_csv(ARGUMENTS["output_path"])

if __name__ == "__main__":
    main()
