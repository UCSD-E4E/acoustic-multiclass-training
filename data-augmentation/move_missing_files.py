import shutil
import os
import pandas as pd

path = '/share/acoustic_species_id'
train_path = os.path.join(path, 'BirdCLEF2023_train_audio')
missing_path = os.path.join(path, 'missing_audio')

missing_df = pd.read_csv(os.path.join(path, 'missing_files.csv'))

for _, row in missing_df.iterrows():
    file_name = row['File Name'] + '.wav'
    folder = row['Folder']

    folder_path = os.path.join(train_path, folder)
    missing_folder_path = os.path.join(missing_path, folder)

    if not os.path.exists(missing_folder_path):
        os.makedirs(missing_folder_path)

    shutil.copyfile(os.path.join(folder_path, file_name), os.path.join(missing_folder_path, file_name))
