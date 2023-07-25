""" Takes in a dataframe csv containing chunked data (from config.yml)
Recognizes overlapping targets and creates a new dataframe with mixed targets
In the "TARGET" column, there will be a dictionary with the target for each class
Example target:
    {
        "class1": 1,
        "other_class": 0.4,
        "other_class2": 0.2
    }
    where the other number is the overlapping proportion with the original annotation
    This may be loaded using ast.literal_eval to get the dictionary from the CSV
"""
import os.path

import pandas as pd
from pyha_analyzer import config

cfg = config.cfg

def apply_overlapping_target(row: pd.Series, group: pd.DataFrame):
    """ Iterates over other rows in group and returns row with target dictionary """
    manual_id = str(row[cfg.manual_id_col])
    target = dict({manual_id: 1})
    start = row[cfg.offset_col]
    end = row[cfg.offset_col] + row[cfg.duration_col]
    # Iterate over other rows in group and check if they overlap
    for _, other_row in group.iterrows():
        other_id = str(other_row[cfg.manual_id_col])
        if other_id == manual_id:
            continue
        other_start = other_row[cfg.offset_col]
        other_end = other_row[cfg.offset_col] + other_row[cfg.duration_col]
        # Calculate overlap
        overlap = 0
        if start < other_start < end:
            overlap = (end - other_start) / row[cfg.duration_col]
        if start < other_end < end:
            overlap = (other_end - start) / row[cfg.duration_col]
        if overlap > 0:
            target[other_id] = round(max(overlap, target.get(other_id, 0)),2)
    row["TARGET"] = target
    return row

def get_targets(df: pd.DataFrame):
    """
    Get the targets for each group and returns a new dataframe
    """
    by_file = df.groupby(cfg.file_name_col, as_index=False)
    unique_files = df[cfg.file_name_col].unique()
    groups = []
    groups = [by_file.get_group(file) for file in unique_files]
    processed_groups = [group.apply(
        lambda row,g=group: apply_overlapping_target(row,g),axis=1
        ) for group in groups]
    return pd.concat(processed_groups)

def main():
    """ Main function """
    df = pd.read_csv(cfg.dataframe_csv, index_col=0)
    df_mixed = get_targets(df)
    # Delete manual id column so it is not accidentally used
    del df_mixed[cfg.manual_id_col]
    df_mixed.to_csv(os.path.basename(cfg.dataframe_csv)+"_mixed.csv")
    
if __name__ == "__main__":
    main()
