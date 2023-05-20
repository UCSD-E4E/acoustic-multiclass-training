from os import path
from pydub import AudioSegment
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

parser.add_argument('-f', '--folder_path', default="../acoustic-species-classification/BirdCLEF2023_split_chunks/training")
parser.add_argument('-td', '--train_destination', default="../acoustic-species-classification/BirdCLEF2023_split_chunks/training") #training
parser.add_argument('-vd', '--val_destination', default="../acoustic-species-classification/BirdCLEF2023_split_chunks/validation") #validation

CONFIG = parser.parse_args()

#df = pd.read_csv(CONFIG.csv_path)
#species = df[CONFIG.species_label].unique()
train = []#pd.DataFrame()
test = []#pd.DataFrame()

# stratified split by class

#print(os.listdir(CONFIG.folder_path))

species = [name for name in os.listdir(CONFIG.folder_path) if os.path.isdir(CONFIG.folder_path + "/" + name)]
for s in species:
    print(s)
    filenames  = [name for name in os.listdir(CONFIG.folder_path + "/" + s)]
    for file in filenames:
        print(file)
        try:
            sound = AudioSegment.from_file(CONFIG.folder_path + "/" + s + "/" + file)
            sound.export(CONFIG.folder_path + "/" + s + "/" + file.split(".")[0] + ".wav", format="wav")

            if ".wav" not in file:
                os.remove(CONFIG.folder_path + "/" + s + "/" + file)

        except Exception as e:
            print(e)
            pass