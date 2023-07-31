"""Chunking script from PyHa to convert weak labels to strong labels.
"""
from typing import Dict, List

import pandas as pd


# pylint: disable-next=too-many-arguments
def convolving_chunk(row:dict, 
                     chunk_length_s: int, 
                     min_length_s: float, 
                     overlap: float,
                     chunk_margin_s: float,
                     only_slide=False) -> List[Dict]:
    """
    Helper function that converts a binary annotation row to uniform chunks. 
    Note: Annotations of length shorter than min_length are ignored. Annotations
    that are shorter than or equal to chunk_length are chopped into three chunks
    where the annotation is placed at the start, middle, and end. Annotations
    that are longer than chunk_length are chunked used a sliding window.
    Args:
        row (dict)
            - Single annotation row represented as a dict
        chunk_length_s (int)
            - Duration in seconds to set all annotation chunks
        min_length_s (float)
            - Duration in seconds to ignore annotations shorter in length
        overlap (float)
            - Percentage of overlap between chunks
        chunk_margin_s (float)
            - Duration to pad chunks on either side
        only_slide (bool)
            - If True, only annotations greater than chunk_length_s are chunked
    Returns:
        Array of labels of chunk_length_s duration
    """
    starts = []
    offset_s = max(row['OFFSET']-chunk_margin_s, 0)
    duration_s = row['DURATION']    # length of annotation
    duration_s += 2 * chunk_margin_s
    end_s = min(offset_s + duration_s, row["CLIP LENGTH"])
    chunk_self_time = chunk_length_s * (1 - overlap)
    
    #Ignore small duration (could be errors, play with this value)
    if duration_s < min_length_s:
        return []
    
    # calculate valid offsets for short annotations
    if (duration_s <= chunk_length_s) and not only_slide:
        # start of clip
        if (offset_s + chunk_length_s) < row['CLIP LENGTH']:
            starts.append(offset_s)
        # middle of clip
        if end_s - chunk_length_s/2.0 > 0 and end_s + chunk_length_s/2.0 < row['CLIP LENGTH']:
            starts.append(end_s - chunk_length_s/2.0)
        # end of clip
        if end_s - chunk_length_s > 0:
            starts.append(end_s - chunk_length_s)
    # calculate valid offsets for long annotations
    else:
        clip_num = int(duration_s / chunk_self_time)
        for i in range(clip_num-1):
            if (offset_s + chunk_length_s) + (i * chunk_self_time) < row['CLIP LENGTH']:
                starts.append(offset_s + i * chunk_self_time)
    
    # create new rows
    rows = []
    for value in starts:
        new_row = row.copy()
        new_row['OFFSET'] = value
        new_row['DURATION'] = chunk_length_s
        rows.append(new_row)
    return rows

def convolving_chunk_old(row:dict, 
                     chunk_length_s=3, 
                     min_length_s=0.4, 
                     only_slide=False)->List[Dict]:
    """ Helper function that converts a binary annotation row to uniform chunks.  """
    starts = []
    offset_s = row['OFFSET']        # start time of original clip
    duration_s = row['DURATION']    # length of annotation
    end_s = offset_s + duration_s
    half_chunk_s = chunk_length_s / 2

    if duration_s < min_length_s:
        return []
    # calculate valid offsets for short annotations
    if (duration_s <= chunk_length_s) and not only_slide:
        if (offset_s + chunk_length_s) < row['CLIP LENGTH']:
            starts.append(offset_s) # start of clip
        if end_s - half_chunk_s > 0 and end_s + half_chunk_s < row['CLIP LENGTH']:
            starts.append(end_s - half_chunk_s) # middle of clip
        if end_s - chunk_length_s > 0:
            starts.append(end_s - chunk_length_s) # end of clip
    # calculate valid offsets for long annotations
    else:
        clip_num = int(duration_s / half_chunk_s)
        for i in range(clip_num-1):
            if (offset_s + chunk_length_s) + (i * half_chunk_s) < row['CLIP LENGTH']:
                starts.append(offset_s + i * half_chunk_s)
    rows = []
    for value in starts:
        new_row = row.copy()
        new_row['OFFSET'] = value
        new_row['DURATION'] = chunk_length_s
        rows.append(new_row)
    return rows

# pylint: disable-next=too-many-arguments
def dynamic_yan_chunking(df: pd.DataFrame,
                         chunk_length_s: int,
                         min_length_s: float,
                         overlap: float,
                         chunk_margin_s: float,
                         only_slide: bool=False) -> pd.DataFrame:
    """
    Function that converts a Dataframe containing binary annotations 
    to uniform chunks using a sliding window
    Args:
        df (Dataframe)
            - Dataframe of annotations 
        chunk_length_s (int)
            - Duration in seconds to set all annotation chunks
        min_length_s (float)
            - Duration in seconds to ignore annotations shorter in length
        overlap (float)
            - Percentage of overlap between chunks
        chunk_margin_s (float)
            - Duration to pad chunks on either side
        only_slide (bool)
            - If True, only annotations greater than chunk_length_s are chunked
    Returns:
        Dataframe of labels with chunk_length duration 
    """
    return_dicts = []
    for _, row in df.iterrows():
        rows_dict = convolving_chunk(row.to_dict(),
                                     chunk_length_s,
                                     min_length_s,
                                     overlap,
                                     chunk_margin_s,
                                     only_slide)
        return_dicts.extend(rows_dict)
    return pd.DataFrame(return_dicts)

def dynamic_yan_chunking_old(df: pd.DataFrame,
                         chunk_length_s:int=3,
                         min_length_s:float=0.4,
                         only_slide:bool=False)-> pd.DataFrame:
    """ Function that converts a Dataframe containing binary annotations
    to uniform chunks using a sliding window """
    return_dicts = []
    for _, row in df.iterrows():
        rows_dict = convolving_chunk_old(row.to_dict(), chunk_length_s, min_length_s, only_slide)
        return_dicts.extend(rows_dict)
    return pd.DataFrame(return_dicts)
