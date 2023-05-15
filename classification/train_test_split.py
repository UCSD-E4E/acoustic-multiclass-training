import argparse
import pandas as pd 
from sklearn.model_selection import train_test_split
import os
import glob
import shutil

# given a directory with chunked files, move files into a split of train and validation set
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--csv_path', default="BirdCLEF2023_TweetyNet_Chunks.csv")
parser.add_argument('-l', '--species_label', default="STRONG LABEL")
parser.add_argument('-t', '--train_size', default=0.8)
parser.add_argument('-r', '--random_state', default=0)

parser.add_argument('-f', '--folder_path', default="../../share/acoustic_species_id/BirdCLEF2023_split_chunks")
parser.add_argument('-td', '--train_destination', default="../../share/acoustic_species_id/BirdCLEF2023_split_chunks/train")
parser.add_argument('-vd', '--val_destination', default="../../share/acoustic_species_id/BirdCLEF2023_split_chunks/validation")

CONFIG = parser.parse_args()

df = pd.read_csv(CONFIG.csv_path)
species = df[CONFIG.species_label].unique()
train = []#pd.DataFrame()
test = []#pd.DataFrame()

# stratified split by class
for s in species:
    print(s)
    temp_df = df[df[CONFIG.species_label] == s]
    filenames = temp_df["IN FILE"].unique()
    if len(filenames) <= 1:
        x_train = filenames
        train.append(x_train)
    else:
        x_train, x_test = train_test_split(filenames,train_size=CONFIG.train_size, random_state=CONFIG.random_state)
        train.append(x_train)
        test.append(x_test)
        
        # make validation folders
        if not os.path.exists(CONFIG.val_destination):
            os.makedirs(CONFIG.val_destination)
        if not os.path.exists(os.path.join(CONFIG.val_destination, s)):
            os.makedirs(os.path.join(CONFIG.val_destination, s))

        files = x_test#["IN FILE"].unique()
        files = [i.split('.')[0] for i in files]
        entries = os.listdir(os.path.join(CONFIG.folder_path, s))
        for entry in entries:
            if entry.split('_')[0] in files:
                shutil.move(os.path.join(CONFIG.folder_path, s, entry), os.path.join(CONFIG.val_destination, s))
    
    if not os.path.exists(CONFIG.train_destination):
        os.makedirs(CONFIG.train_destination)
    shutil.move(os.path.join(CONFIG.folder_path, s), os.path.join(CONFIG.train_destination, s))

print("train output: " + str(len(train)))
print("test output: " + str(len(test)))