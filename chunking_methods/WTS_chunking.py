"""Chunking script from PyHa to convert weak labels to strong labels.
"""
import pandas as pd

def create_chunk_row(row, rows_to_add, new_start, new_end):
    """
    Helper function that takes in a Dataframe containing annotations 
    and appends a single row to the Dataframe before returning it.
    Args:
        row (Series)
            - Row of a single annotation
        rows_to_add (Dataframe)
            - Dataframe of labels
        
        new_start (float)
            - The start time of the annotation in row
        duration (int)
            - The duration of the annotation in row
    Returns:
        Dataframe of labels with the newly appended row
    """
    chunk_row = row.copy()
    chunk_row["DURATION"] = new_end
    chunk_row["OFFSET"] = new_start
    rows_to_add.append(chunk_row.to_frame().T)
    return rows_to_add

def convolving_chunk(row, chunk_duration=3, only_slide=False):
    """
    Helper function that converts a Dataframe row containing a binary
    annotation to uniform chunks of chunk_length. 
    Note: Annotations of length shorter than min_length are ignored. Annotations
    that are shorter than or equal to chunk_length are chopped into three chunks
    where the annotation is placed at the start, middle, and end. Annotations
    that are longer than chunk_length are chunked used a sliding window.
    Args:
        row (Series)
            - Row of a single annotation
        chunk_length (int)
            - duration in seconds to set all annotation chunks
        
        min_length (float)
            - duration in seconds to ignore annotations shorter in length
        only_slide (bool)
            - If True, only annotations greater than chunk_length are chunked
    Returns:
        Dataframe of labels with chunk_length duration 
        (elements in "OFFSET" are divisible by chunk_length).
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
    """
    Function that converts a Dataframe containing binary
    annotations to uniform chunks of chunk_length. 
    Note: Annotations shorter than min_length are ignored. Annotations
    shorter than or equal to chunk_length are chopped into three chunks
    where the annotation is placed at the start, middle, and end. Annotations
    longer than chunk_length are chunked used a sliding window.
    Args:
        df (Dataframe)
            - Dataframe of annotations 
        chunk_length (int)
            - duration in seconds to set all annotation chunks
        
        min_length (float)
            - duration in seconds to ignore annotations shorter than
        only_slide (bool)
            - If True, only annotations greater than chunk_length are chunked
    Returns:
        Dataframe of labels with chunk_length duration 
        (elements in "OFFSET" are divisible by chunk_length).
    """
    return_dfs = []
    for _, row in df.iterrows():
        chunk_df = convolving_chunk(row, chunk_duration=chunk_duration, only_slide=only_slide)
        return_dfs.append(chunk_df)
    return pd.concat(return_dfs,  ignore_index=True)