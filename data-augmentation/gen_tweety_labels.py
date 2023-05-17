from PyHa.PyHa.IsoAutio import *
import pandas as pd
import os 

# path = "../train_audio/"
path = "/share/acoustic_species_id/BirdCLEFPeru_Audio/"

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

subfolders = [f.path for f in os.scandir(path) if f.is_dir() ]
automated_df = pd.DataFrame()
subfolders.sort()
print("starting predictions...")
#print(subfolders)
#for s in subfolders:
#    for fn in os.listdir(s):
#        if fn.endswith('.ogg'):
#            f = os.path.join(s,fn)
#            ogg2wav(f)
#     print(s + "/")
# temp_df = generate_automated_labels(s+ "/", isolation_parameters_tweety)
#     if temp_df.empty:
#        sys.exit() 
#    automated_df = pd.concat([automated_df, temp_df], ignore_index=True, sort=False)

# for i in range(len(subfolders)):
#     print(str(i) + " " + subfolders[i] + "/")
#     temp_df = generate_automated_labels(subfolders[i] + "/", isolation_parameters_tweety)
#     automated_df = pd.concat([automated_df, temp_df], ignore_index=True, sort=False)
#     # in case your script crashes
#     # automated_df.to_csv("BirdCLEF2023_TweetyNet_Labels.csv")
#     automated_df.to_csv("TestIsolation_TweetyNet_Labels.csv")

temp_df = generate_automated_labels(path, isolation_parameters_tweety)
automated_df = pd.concat([automated_df, temp_df], ignore_index=True, sort=False)
# in case your script crashes
# automated_df.to_csv("BirdCLEF2023_TweetyNet_Labels.csv")
automated_df.to_csv("Test_Peru_labels.csv")