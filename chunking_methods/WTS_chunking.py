"""Chunking script from PyHa to convert weak labels to strong labels.
"""
import pandas as pd

def create_chunk_row(row, rows_to_add, new_start, new_end):
    """Add a new row with specified duration and offset
    """
    chunk_row = row.copy()
    chunk_row["DURATION"] = new_end
    chunk_row["OFFSET"] = new_start
    rows_to_add.append(chunk_row.to_frame().T)
    return rows_to_add

def convolving_chunk(row, chunk_duration=3, only_slide=False):
    """Sliding window to cover different parts of the same call
    """
    chunk_df = pd.DataFrame(columns=row.to_frame().T.columns)
    rows_to_add = []
    offset = row["OFFSET"]
    duration = row["DURATION"]
    chunk_half_duration = chunk_duration / 2
    
    #Ignore small duration (could be errors, play with this value)
    if duration < 0.4:
        return chunk_df

    if duration <= chunk_duration and not only_slide:
        #Put the original bird call at...
        #1) Start of clip
        if (offset+chunk_duration) < row["CLIP LENGTH"]:
            rows_to_add = create_chunk_row(row, rows_to_add, offset, chunk_duration)
            
        #2) End of clip
        if offset+duration-chunk_duration > 0: 
            rows_to_add = create_chunk_row(row, rows_to_add, offset+duration-chunk_duration, chunk_duration)
            
        #3) Middle of clip
        if offset+duration-chunk_half_duration>0 and (offset+duration+chunk_half_duration)< row["CLIP LENGTH"]:
            #Could be better placed in middle, maybe with some randomization?
            rows_to_add = create_chunk_row(row, rows_to_add, (offset+duration-chunk_half_duration), chunk_duration)
            
    #Longer than chunk duration
    else:
        #Perform Yan's Sliding Window operation
        clip_num=int(duration/(chunk_half_duration))
        for i in range(clip_num-1):
            new_start = offset+i*chunk_half_duration
            #new_end = offset + chunk_duration+i*chunk_half_duration
            if ((offset+chunk_duration)+i*chunk_half_duration) < row["CLIP LENGTH"]:
                rows_to_add = create_chunk_row(row, rows_to_add, new_start, chunk_duration) 
    #Add all new rows to our return df
    if len(rows_to_add) == 0:
        return chunk_df
    chunk_df = pd.concat(rows_to_add, ignore_index=True)
    return chunk_df

def dynamic_yan_chunking(df, chunk_duration=3, only_slide=False):
    """TODO
    """
    return_dfs = []
    for _, row in df.iterrows():
        chunk_df = convolving_chunk(row, chunk_duration=chunk_duration, only_slide=only_slide)
        return_dfs.append(chunk_df)
    return pd.concat(return_dfs,  ignore_index=True)