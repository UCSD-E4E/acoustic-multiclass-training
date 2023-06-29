from PyHa.IsoAutio import *
import pandas as pd
import os 
import sys


isolation_parameters_tweety = {
    "model" : "tweetynet",
    "tweety_output": True,
    "verbose" : True
}


# for converting to wav files
from pydub import AudioSegment
def ogg2wav(ofn):
    wfn = ofn.replace('.ogg', '.wav')
    x = AudioSegment.from_file(ofn)
    x.export(wfn, format='wav')

def gen_labels(path):
    if not os.path.exists(os.path.join(path, "train_audio")):
        print(f"Directory \"train_audio\" not found in path {path}", file=sys.stderr)
        sys.exit(1)
    audio_path = os.path.join(path, "train_audio")
    subfolders = [f.path for f in os.scandir(audio_path) if f.is_dir() ]
    automated_df = pd.DataFrame()
    subfolders.sort()
    print("starting predictions...")
    for s in subfolders:
        for fn in os.listdir(s):
            if fn.endswith('.ogg'):
                f = os.path.join(s,fn)
                ogg2wav(f)
        print(s + "/")
        temp_df = generate_automated_labels(s+ "/", isolation_parameters_tweety)
        if temp_df.empty:
            continue
        automated_df = pd.concat([automated_df, temp_df], ignore_index=True, sort=False)
    automated_df.to_csv(os.path.join(path, "BirdCLEF2023_TweetyNet_Labels.csv"))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect number of args", file=sys.stderr)
        print("USAGE: python gen_tweety_labels.py /path", file=sys.stderr)
        sys.exit(1)
    gen_labels(sys.argv[1])
    