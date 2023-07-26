"""Chunking script from PyHa to convert weak labels to strong labels.

    Based on trying to center the chunk on an annotation and avoid overlaps
"""
from typing import Dict, List

import pandas as pd


def center_small_anno(row, chunk_length_s=3, 
                     min_length_s=0.4) -> List[Dict]:
    """
    Helper function that converts a binary annotation row to uniform chunks. 
    Note: Annotations of length shorter than min_length are ignored. Attempts to center annotation
    in chunk by moving the offset down by half the time between annotation duration and chunk length
    Args:
        row (dict)
            - Single annotation row represented as a dict
        chunk_length_s (int)
            - Duration in seconds to set all annotation chunks
        min_length_s (float)
            - Duration in seconds to ignore annotations shorter in length
        only_slide (bool)
            - If True, only annotations greater than chunk_length_s are chunked
    Returns:
        Array of labels of chunk_length_s duration
    """

    offset_s = row['OFFSET']        # start time of original clip
    duration_s = row['DURATION']    # length of annotation
    clip_len_s = row['CLIP LENGTH']

    #If the clip cannot be centered, return it and let future preprocessing handle it
    if duration_s < min_length_s:
        return []
    
    if clip_len_s < chunk_length_s:
        return [row]

    # Get the new offset based on centering the annotation in the chunk
    off_offset_s = float(chunk_length_s - duration_s)  / 2
    new_offset_s = offset_s - off_offset_s

    #Handle case of offset close to start of clip
    new_offset_s = max(new_offset_s, 0) 

    #Handle case of offset close to end of clip
    if new_offset_s + chunk_length_s > clip_len_s:
        new_offset_s = clip_len_s - chunk_length_s

    new_row = row.copy()
    new_row['OFFSET'] = new_offset_s
    new_row['DURATION'] = chunk_length_s
    return [new_row]

def center_large_anno(row,  chunk_length_s=3, include_last=True) -> List[Dict]:
    """
    Helper function that converts a binary annotation row to uniform chunks. 
    Note: Annotations of length shorter than min_length are ignored. Starts a naive
    chunking at the offset of the row
    Args:
        row (dict)
            - Single annotation row represented as a dict
        chunk_length_s (int)
            - Duration in seconds to set all annotation chunks
        min_length_s (float)
            - Duration in seconds to ignore annotations shorter in length
        only_slide (bool)
            - If True, only annotations greater than chunk_length_s are chunked
    Returns:
        Array of labels of chunk_length_s duration
    """
    
    offset_s = row['OFFSET']        # start time of original clip
    duration_s = row['DURATION']    # length of annotation
    clip_len_s = row['CLIP LENGTH']

    #determine number of possible chunks
    chunk_count = int(duration_s // chunk_length_s)

    #Create naive chunks from here
    print(offset_s)
    chunk_starts = [(offset_s + i * chunk_length_s) for i in range(chunk_count)]
    chunk_durats = [5 for _ in range(chunk_count)]

    print(len(chunk_starts), len(chunk_durats))

    #Check end of annotation case
    if include_last:
        last_offset = offset_s + chunk_length_s * chunk_count
        
        # If the chunk goes beyond length of the clip
        # set the duration to the distance between end of clip and offset
        # otherwise normal behavior
        amount_over = last_offset + chunk_length_s - clip_len_s
        if amount_over > 0:
            last_durati = clip_len_s - last_offset
        else:
            last_durati = chunk_length_s


        chunk_starts.append(last_offset)
        chunk_durats.append(last_durati)
        print(len(chunk_starts), len(chunk_durats))

    # create new rows
    assert len(chunk_starts) == len(chunk_durats)
    rows = []
    for new_offset, new_duration in zip(chunk_starts, chunk_durats):
        new_row = row.copy()
        new_row['OFFSET'] = new_offset
        new_row['DURATION'] = new_duration
        rows.append(new_row)
    return rows

def make_center_chunk(row:dict, 
                     chunk_length_s=3, 
                     min_length_s=0.4,
                     include_last=True 
                     )->List[Dict]:
    """
    Helper function that converts a binary annotation row to uniform chunks. 
    Note: Annotations of length shorter than min_length are ignored. If annotation
    is less than chunking length, center the chunk. Otherwise do a naive chunk starting at
    the row offset
    Args:
        row (dict)
            - Single annotation row represented as a dict
        chunk_length_s (int)
            - Duration in seconds to set all annotation chunks
        min_length_s (float)
            - Duration in seconds to ignore annotations shorter in length
        only_slide (bool)
            - If True, only annotations greater than chunk_length_s are chunked
    Returns:
        Array of labels of chunk_length_s duration
        Empty if annotation is still too small
    """
    duration_s = row['DURATION']    # length of annotation
    
    #Ignore small duration (could be errors, play with this value)
    if duration_s <= chunk_length_s:
        return center_small_anno(
            row,
            chunk_length_s=chunk_length_s, 
            min_length_s=min_length_s
        )
    
    #Else clip is longer than chunk length
    return center_large_anno(
        row,
        chunk_length_s=chunk_length_s, 
        include_last=include_last
    )


def center_chunking(df: pd.DataFrame,
                         chunk_length_s:int=5,
                         min_length_s:float=0.4,
                         include_last:bool=True)-> pd.DataFrame:
    """
    Function that converts a Dataframe containing binary annotations 
    to uniform chunks by centering chunks on annotations
    Args:
        df (Dataframe)
            - Dataframe of annotations 
        chunk_length_s (int)
            - Duration in seconds to set all annotation chunks
        min_length_s (float)
            - Duration in seconds to ignore annotations shorter in length
        only_slide (bool)
            - If True, only annotations greater than chunk_length_s are chunked
    Returns:
        Dataframe of labels with chunk_length duration 
    """
    return_dicts = []
    for _, row in df.iterrows():
        rows_dict = make_center_chunk(row.to_dict(), chunk_length_s, min_length_s, include_last)
        return_dicts.extend(rows_dict)
    print(return_dicts)
    return pd.DataFrame(return_dicts)
