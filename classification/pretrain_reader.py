import os

import pandas as pd
import torchaudio
from config import get_config


class PretrainReader:
    def __init__(self):
        """ Init PretrainReader by loading the datasets.csv file
        confirms that required directories exist
        """
        
        self.config = get_config()

        # Load datasets.csv
        if not os.path.exists(self.config.dataframe):
            raise RuntimeError(f"datasets.csv file at {self.config.dataframe} not found at absolute path")
        self.datasets = pd.read_csv(self.config.dataframe, index_col=0)
    
        self.data = []
        
        # Confirm that required directories and files exist
        for index, row in self.datasets.iterrows():
            self.verify_dataset(index)
            # Read csv to self.data
            annotation_csv = os.path.join(self.config.data_path, row["folder"], row["annotations"])
            self.data.append(pd.read_csv(annotation_csv, index_col=0))
        
        self.cur_index = 0
        self.cur_dataset = 0
    

    def verify_dataset(self,index: int):
        row = self.datasets.iloc[index]
        # Verify that directory strucutre is correct
        folder = os.path.join(self.config.data_path, row["folder"])
        if not os.path.exists(folder):
            raise RuntimeError(f"Dataset located at {folder} does not exist")
        data_folder = os.path.join(self.config.data_path, row["folder"], row["data_folder"])
        if not os.path.exists(data_folder):
            raise RuntimeError(f"Dataset data folder located at {data_folder} does not exist")
        annotation_csv = os.path.join(self.config.data_path, row["folder"], row["annotations"])
        if not os.path.exists(annotation_csv):
            raise RuntimeError(f"Dataset annotation file located at {annotation_csv} does not exist")
        
        # Check for missing files
        missing = 0
        for index, row in self.datasets.iterrows():
            folder = os.path.join(self.config.data_path, row["folder"])
            data_folder = os.path.join(self.config.data_path, row["folder"], row["data_folder"])
            data_dir = os.listdir(data_folder)
            for file in self.data[index][self.config.file_name_col].unique():
                if row[self.config.file_name_col] not in data_dir:
                    missing += 1
                    if self.config.verbose: print("Cannot find",row[self.config.file_name_col],"in",data_folder)
        if missing > 0:
            print("Missing ",missing,"files in",row["folder"])

    def get_classes(self):
        """ Returns a list of all the classes in the pretrain_data
        """
        classes = []
        for index, row in self.datasets.iterrows():
            classes.append(row[self.config.manual_id_col])
        return classes.unique()
        
    def __iter__(self):
        """ Returns self
        """
        return self
    
    def __next__(self):
        """ Get next file in the dataset
        """
        self.cur_index += 1
        # Wrap around to next dataset
        if self.cur_index >= len(self.data[self.cur_dataset]):
            self.cur_index = 0
            self.cur_dataset += 1
            # Wrap around to first dataset and stop iteration
            if self.cur_dataset >= len(self.data):
                self.cur_dataset = 0
                raise StopIteration
        
        # Get audio
        annotation = self.data[self.cur_dataset].iloc[self.cur_index]
        file_path = os.path.join(self.config.data_path, 
                                 self.datasets.iloc[self.cur_dataset]["folder"], 
                                 self.datasets.iloc[self.cur_dataset]["data_folder"], 
                                 annotation[self.config.file_name_col])
        audio, sample_rate = torchaudio.load(file_path)

        return audio, annotation[self.config.offset_col], annotation[self.config.duration_col], annotation[self.config.manual_id_col]