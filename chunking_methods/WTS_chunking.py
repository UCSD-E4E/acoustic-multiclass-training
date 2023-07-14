"""Chunking script from PyHa to convert weak labels to strong labels.
"""
import math

import numpy as np
import pandas as pd


def annotation_chunker_no_duplicates(
        kaleidoscope_df,
        chunk_length,
        include_no_bird=False):
    """
    Function that converts a Kaleidoscope-formatted Dataframe to 
    uniform chunks of chunk_length. If there are multiple species 
    in the same clip, create chunks for more confident species.

    Note: if all or part of an annotation covers the last < chunk_length
    seconds of a clip it will be ignored. If two annotations overlap in 
    the same 3 second chunk, both are represented in that chunk
    Args:
        kaleidoscope_df (Dataframe)
            - Dataframe of annotations in kaleidoscope format
        chunk_length (int)
            - duration to set all annotation chunks
        include_no_bird (boolean)
            - create annotations for "no bird" segments
    Returns:
        Dataframe of labels with chunk_length duration 
        (elements in "OFFSET" are divisible by chunk_length).
    """
    #Init list of clips to cycle through and output dataframe
    kaleidoscope_df['FILEPATH'] = kaleidoscope_df.loc[:,['FOLDER','IN FILE']].sum(axis=1)
    clips = kaleidoscope_df["FILEPATH"].unique()
    df_columns = {'FOLDER': 'str', 'IN FILE' :'str', 'CLIP LENGTH' : 'float64', 'CHANNEL' : 'int64',
                  'OFFSET' : 'float64','DURATION' : 'float64', 'SAMPLE RATE' : 'int64','MANUAL ID' : 'str'}
    output_df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in df_columns.items()})

    # going through each clip
    for clip in clips:
        clip_df = kaleidoscope_df[kaleidoscope_df["FILEPATH"] == clip]
        path = clip_df["FOLDER"].unique()[0]
        file = clip_df["IN FILE"].unique()[0]
        sr = clip_df["SAMPLE RATE"].unique()[0]
        clip_len = clip_df["CLIP LENGTH"].unique()[0]

        # quick data sanitization to remove very short clips
        # do not consider any chunk that is less than chunk_length
        if clip_len < chunk_length:
            continue
        potential_annotation_count = int(clip_len)//int(chunk_length)

        # going through each species that was ID'ed in the clip
        arr_len = int(clip_len*1000)
        species_df = clip_df
        human_arr = np.zeros((arr_len))
        # looping through each annotation
        for annotation in species_df.index:
            minval = int(round(species_df["OFFSET"][annotation] * 1000, 0))
            # Determining the end of a human label
            maxval = int(
                round(
                    (species_df["OFFSET"][annotation] +
                        species_df["DURATION"][annotation]) *
                    1000,
                    0))
            # Placing the label relative to the clip
            human_arr[minval:maxval] = 1
        # performing the chunk isolation technique on the human array

        for index in range(potential_annotation_count):
            chunk_start = index * (chunk_length*1000)
            chunk_end = min((index+1)*chunk_length*1000,arr_len)
            chunk = human_arr[int(chunk_start):int(chunk_end)]

            #Get row data
            row = pd.DataFrame(index = [0])
            annotation_start = chunk_start / 1000

            if max(chunk) >= 0.5:
                #Handle birdnet output edge case
                if sum(clip_df["DURATION"] == 3)/clip_df.shape[0] == 1:
                    overlap = (clip_df["OFFSET"]+0.5 >= (annotation_start)) \
                            & (clip_df["OFFSET"]-0.5 <= (annotation_start))
                    annotation_df = clip_df[overlap]
                else:
                    overlap = is_overlap(clip_df["OFFSET"], 
                                         clip_df["OFFSET"] + clip_df["DURATION"], 
                                         annotation_start,
                                         annotation_start + chunk_length)
                    annotation_df = clip_df[overlap]
                
                #updating the dictionary
                if 'CONFIDENCE' in clip_df.columns:
                    annotation_df = annotation_df.sort_values(by="CONFIDENCE", ascending=False)
                    row["CONFIDENCE"] = annotation_df.iloc[0]["CONFIDENCE"]
                else:
                    #The case of manual id, or there is an annotation with no known confidence
                    row["CONFIDENCE"] = 1
                
                row["MANUAL ID"] = annotation_df.iloc[0]["MANUAL ID"] 
               
            elif include_no_bird:
                #updating the dictionary
                row["CONFIDENCE"] = 0
                row["MANUAL ID"] = "no bird"
                
            row["FOLDER"] = path
            row["IN FILE"] = file
            row["CLIP LENGTH"] = clip_len
            row["OFFSET"] = annotation_start
            row["DURATION"] = chunk_length
            row["SAMPLE RATE"] = sr
            row["CHANNEL"] = 0
            output_df = pd.concat([output_df,row], ignore_index=True)

    return output_df


def is_overlap(offset_df, end_df, chunk_start, chunk_end):
    """Check for overlap between two chunks
    """
    is_both_before = (chunk_end < offset_df) & (chunk_start < offset_df)
    is_both_after = (end_df < chunk_end) & (end_df < chunk_start)
    return (~is_both_before) & (~is_both_after)

def exact_split(df, chunk_len=3):
    """TODO
    """
    def gen_list(a, b, chunk):
        # generate multiples of 3 between a and b
        multiplist = [i for i in range(math.ceil(a), math.ceil(b)) if i % chunk == 0]
        # if there are multiples
        if multiplist:
            # insert start and end times
            multiplist.insert(0,a)
            multiplist.append(b)
            # create a list of list ranges
            multiplist = [[multiplist[i], multiplist[i+1]] for i in range(len(multiplist) - 1)]
        return multiplist

    # create an end time column
    df["END_TIME"] = df["OFFSET"] + df["DURATION"]
    df["SPLIT"] = df.apply(lambda x: gen_list(x["OFFSET"], x["END_TIME"], chunk_len), axis=1)
    df_split = df.copy()
    df_split = df_split.explode("SPLIT", ignore_index=True)

    # assign start time to index 0 in "SPLIT" range
    df_split["START_TIME"] = df_split["SPLIT"].dropna().map(lambda x: x[0])
    df_split["START_TIME"] = df_split["START_TIME"].fillna(df_split["OFFSET"])
    # assign end time to index 1 in "SPLIT" range
    df_split["END_TIME"] = df_split["SPLIT"].dropna().map(lambda x: x[1])
    df_split["END_TIME"] = df_split["END_TIME"].fillna(df_split["OFFSET"] + df_split["DURATION"])

    df_split["OFFSET"] = df_split["START_TIME"]
    df_split["DURATION"] = df_split["END_TIME"] - df_split["START_TIME"]
    return df_split

def fill_no_class(df, chunk):
    """Helper method to create "no class" annotations for times that 
    do not already have annotations. 
    """
    new_df = df.copy()
    count = 0
    files = np.unique(df["IN FILE"])
    for file in files:
        print(count, len(files))
        count += 1
        sub_df = df[df["IN FILE"] == file]
        clip_length = sub_df.iloc[0]["CLIP LENGTH"]
        chunks = list(range(int(clip_length // chunk + 1)))
        chunks_with_annotations = np.array(sub_df["CHUNK_ID"].apply(int))
        no_class_chunks = np.setdiff1d(chunks,chunks_with_annotations)
        tmp_row = sub_df.iloc[0]
        for chunk_off in no_class_chunks:
            tmp_row["OFFSET"] = chunk_off
            tmp_row["START_TIME"] = chunk_off
            tmp_row["END_TIME"] = chunk_off + chunk
            tmp_row["DURATION"] = chunk
            tmp_row["MANUAL ID"] = "no class"
            tmp_row["CHUNK_ID"] = chunk_off
            tmp_row["SPLIT"] = []
            new_df = new_df.append(tmp_row)
    df = new_df.sort_values(["IN FILE", "OFFSET"])
    return df

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

def combine_chunked_df(df):
    """
    Note: Assumes all labels are the same.
    """
    dfs = []
    for file in df["IN FILE"].unique():
        file_df = df[df["IN FILE"] == file]
        data_df = file_df.copy()

        file_df = file_df.sort_values(by="OFFSET")
        file_df["END"] = np.array(file_df["OFFSET"] + file_df["DURATION"])
        file_df["NEXT START"] = file_df["OFFSET"].shift(-1)
        file_df["LAST END"] = file_df["END"].shift(1)

        #IF TRUE, THEN THIS ROW IS SEPERATED ABOVE AND BELOW FROM ANY OTHER ANNOTATION
        temp = ~((file_df['END'] == file_df['NEXT START']) | (file_df['OFFSET'] == file_df['LAST END']))

        #SEE IF WE HAVE THE START OF A GROUPED ANNOTATION OR WE HAVE A INDEPETENT ANNOTATION
        file_df["GROUPS"] = ((temp.shift() != temp) | temp).cumsum()

        file_df = file_df[["OFFSET", "END", "NEXT START", "LAST END", "GROUPS"]]
        file_df = file_df.groupby("GROUPS").agg({"OFFSET": "min", "END": "max"}).reset_index(drop=True)
        
        file_df["DURATION"] = file_df["END"] - file_df["OFFSET"]
        file_df["IN FILE"] = file
        file_df["FOLDER"] = data_df.iloc[0]["FOLDER"]
        file_df["CLIP LENGTH"] = data_df.iloc[0]["CLIP LENGTH"]
        file_df["CHANNEL"] = data_df.iloc[0]["CHANNEL"]
        file_df["MANUAL ID"] = data_df.iloc[0]["MANUAL ID"]
        file_df["SAMPLE RATE"] = data_df.iloc[0]["SAMPLE RATE"]
        dfs.append(file_df)
    return pd.concat(dfs).reset_index(drop=True)
