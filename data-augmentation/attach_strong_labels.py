import pandas as pd
import os

# NOTE: When copying the unlabeled rows to the new csv,
#       the floating point values are slightly different. 
#       We believe the differences are negligable, wo we'll 
#       ignore it.

path = "/share/acoustic_species_id/"

metadata_file = "train_metadata.csv"
binary_labels_file = "BirdCLEF2023_TweetyNet_Labels.csv"
strong_labels_file = "BirdCLEF2023_TweetyNet_StrongLabels.csv"

def attach_labels(binary_labels_file, replace_file=False, new_file_name=strong_labels_file):

  if replace_file:
     print('Replacing file not implemented')
     return
  
  else:
    metadata_path = os.path.join(path, metadata_file)
    binary_path = os.path.join(path, binary_labels_file)
    strong_path = os.path.join(path, strong_labels_file)

    metadata_df = pd.read_csv(metadata_path)
    binary_df = pd.read_csv(binary_path)
    strong_df = pd.DataFrame()

    # for each row in our metadata, find the corresponding rows in the binary labeled df
    for i in range(len(metadata_df)):
        metadata_row = metadata_df.iloc[i]

        ogg_file_name = metadata_row["filename"].split("/")[-1]
        wav_file_name = ogg_file_name.split('.')[0] + '.wav'

        matching_rows = binary_df.loc[binary_df['IN FILE'] == wav_file_name].copy()

        if len(matching_rows) == 0:
            continue
        
        strong_label = metadata_row["primary_label"]
        matching_rows['STRONG LABEL'] = [strong_label] * len(matching_rows)

        strong_df = pd.concat([strong_df, matching_rows], ignore_index=True, sort=False)
        #in case script crashes
        strong_df.to_csv(strong_path)

if __name__ == "__main__":
  attach_labels(binary_labels_file)