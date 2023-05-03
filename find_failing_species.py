import pandas as pd
import numpy as np
import os

path = "/share/acoustic_species_id"

failed_files = {}
missing_files = {}

failed_folder = ''
failed_file = ''

binary_df = pd.read_csv(os.path.join(path, 'BirdCLEF2023_TweetyNet_Labels.csv'))
metadata_df = pd.read_csv(os.path.join(path, 'train_metadata.csv'))

#find the files that failed in TweetyNet
with open(os.path.join(path, 'nohup.txt'), 'r') as f:
    for line in f:
        if line[:5] == 'Error':
            reason = line.split(' ')[2][:-1]
            failed_file = line[line.index('XC'):-5]

            if (failed_file, reason) not in failed_files[failed_folder]:
                failed_files[failed_folder].append((failed_file, reason))                

            
        else:
            if 'starting' in line:
              continue

            failed_folder = line.split('/')[-2]
            failed_files[failed_folder] = []
        
        

#find the missing files
for i in range(len(metadata_df)):
    metadata_row = metadata_df.iloc[i]

    ogg_file_name = metadata_row["filename"].split("/")[-1]
    wav_file_name = ogg_file_name.split('.')[0] + '.wav'

    matching_rows = binary_df.loc[binary_df['IN FILE'] == wav_file_name]

    if len(matching_rows) == 0:
      folder_name = metadata_row["primary_label"]

      if folder_name not in missing_files:
          missing_files[folder_name] = []
      
      missing_files[folder_name].append(wav_file_name[:-4])
    
    

folders_df = pd.DataFrame()
failing_df = pd.DataFrame()
missing_df = pd.DataFrame()

#generate failing and missing csvs
for folder in failed_files:
    
    num_failed = len(failed_files[folder])
    num_missing = 0

    if folder in missing_files:
        num_missing = max(0, len(missing_files[folder]) - num_failed)

    folder_row = {'Folder': [folder],
           'Num. Failed': [num_failed],
           'Num. Missing': [num_missing]}
    
    folders_df = pd.concat([folders_df, pd.DataFrame(folder_row)], ignore_index=True, sort=False)

    for failed_file in failed_files[folder]:
        
        file_name = failed_file[0]
        reason = failed_file[1]

        file_row = {'File Name': [file_name],
                    'Folder': [folder],
                    'Reason': [reason]}
        
        failing_df = pd.concat([failing_df, pd.DataFrame(file_row)], ignore_index=True, sort=False)

    
failing_filenames = failing_df['File Name'].to_numpy()

# NOTE: Refactor using Pandas operations -- get rid of for loops

for folder in missing_files:
    
    for missing_file in missing_files[folder]:

        if missing_file in failing_filenames:
            continue
        
        file_row = {'File Name': [missing_file],
                    'Folder': [folder]}
        
        missing_df = pd.concat([missing_df, pd.DataFrame(file_row)], ignore_index=True, sort=False)


folders_df.to_csv(os.path.join(path, 'folders_fm_stats.csv'))
failing_df.to_csv(os.path.join(path, 'failing_files.csv'))
missing_df.to_csv(os.path.join(path, 'missing_files.csv'))