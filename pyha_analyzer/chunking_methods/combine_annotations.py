""" Combines short frequent annotations into a longer call annotation """

import argparse

import pandas as pd
from tqdm import tqdm

def combine_annotations(df: pd.DataFrame, max_gap_s: float = 0.5) -> pd.DataFrame:
    """ Combine any annotations that have a gap length less than max_gap seconds"""
    groups = df.groupby("IN FILE")
    out_groups = []
    for _, group in tqdm(groups):
        group.reset_index(drop=True, inplace=True)
        df = group.sort_values(by=["OFFSET"])
        i = 0
        off_col = df.columns.get_loc("OFFSET")
        dur_col = df.columns.get_loc("DURATION")
        # Can't use a for loop because combining annotations changes the length of the dataframe :(
        while i < len(df.index) - 1:
            gap_length = df.iloc[i+1,off_col] - df.iloc[i,off_col] - df.iloc[i,dur_col]
            if gap_length < max_gap_s:
                # Combine the two annotations by increasing the firsts duration
                # and deleting the second
                df.iloc[i, dur_col] += gap_length + df.iloc[i+1,dur_col]
                df.drop(df.iloc[i+1].name, inplace=True)
            else:
                i += 1

        df.reset_index(drop=True, inplace=True)
        out_groups.append(df)
    return pd.concat(out_groups, ignore_index=True, sort=False)

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input', type=str)
    argparser.add_argument('-o', '--output', type=str)
    argparser.add_argument('-g', '--max_gap', type=float, default=0.3)
    args = argparser.parse_args()
    assert args.input is not None, "Input file not specified"
    assert args.output is not None, "Output file not specified"

    dataframe = pd.read_csv(args.input, index_col=0)
    combined = combine_annotations(dataframe, args.max_gap)
    combined.to_csv(args.output)
    print("Old annotation count:",len(dataframe.index))
    print("New annotation count:",len(combined.index))
