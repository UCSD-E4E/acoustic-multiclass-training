""" Contains methods for loading the dataset and creates dataloaders for training and validation

    PyHaDataset is a generic loader with a given root directory.
    It loads the audio files and converts them to mel spectrograms.
    get_datasets returns the train and validation datasets as BirdCLEFDataset objects.

    If this module is run directly, it tests that the dataloader works

"""
import collections
import logging
import os
from typing import List, Tuple, Optional
import ast

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchaudio import transforms as audtr
from torchvision.transforms import RandomApply
from tqdm import tqdm
import wandb

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
from pyha_analyzer import config
from pyha_analyzer import utils
from pyha_analyzer.augmentations import (BackgroundNoise, LowpassFilter, Mixup, RandomEQ,
                                         HighpassFilter, SyntheticNoise)
from pyha_analyzer.chunking_methods import sliding_chunks

cfg = config.cfg

tqdm.pandas()
logger = logging.getLogger("acoustic_multiclass_training")

# sound separation code from https://github.com/google-research/chirp
SeparatorState = collections.namedtuple(
    'SeparatorState', ['session', 'audio_placeholder', 'output_tensor']
)

def load_separation_model(model_path, checkpoint_path):
    """Loads a separation model graph for inference."""
    metagraph_path_ns = os.path.join(model_path, 'inference.meta')
    graph_ns = tf.Graph()

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

    sess_ns = tf.Session(config=tf.ConfigProto(
        device_count={'GPU': 1},
        #gpu_options=gpu_options
    ), graph=graph_ns)
    with graph_ns.as_default():
        new_saver = tf.train.import_meta_graph(metagraph_path_ns)
        new_saver.restore(sess_ns, checkpoint_path)
        input_placeholder_ns = graph_ns.get_tensor_by_name(
            'input_audio/receiver_audio:0'
        )
    output_tensor_ns = graph_ns.get_tensor_by_name('denoised_waveforms:0')
    return SeparatorState(sess_ns, input_placeholder_ns, output_tensor_ns)

# pylint: disable=too-many-instance-attributes
class PyhaDFDataset(Dataset):
    """
        Dataset designed to work with pyha output
        Save unchunked data
    """

    # df, train, and species decided outside of config, so those cannot be added in there
    # pylint: disable-next=too-many-arguments
    def __init__(self,
                 df: pd.DataFrame,
                 train: bool,
                 species: List[str],
                 onehot:bool = False,
                 ) -> None:
        self.samples = df[~(df[cfg.file_name_col].isnull())]
        if onehot:
            if self.samples.iloc[0][species].shape[0] != len(species):
                logger.error(species)
                logger.error("make sure class list is fully onehot encoded")
                raise RuntimeError("One hot values differ from species list")

        self.num_samples = cfg.sample_rate * cfg.chunk_length_s
        self.train = train
        self.device = cfg.prepros_device
        self.onehot = onehot

        # List data directory and confirm it exists
        if not os.path.exists(cfg.data_path):
            raise FileNotFoundError("Data path does not exist")
        self.data_dir = set()
        for root, _, files in os.walk(cfg.data_path):
            self.data_dir |= {os.path.join(root,file) for file in files}

        #Log bad files
        self.bad_files = []

        #Preprocessing start
        self.samples[cfg.manual_id_col] = self.samples[cfg.manual_id_col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("{") else x
        )
        self.classes = species
        self.class_to_idx = dict(zip(species, np.arange(len(species))))

        self.num_classes = len(species)
        self.separator_state = load_separation_model(cfg.sep_model_path, cfg.sep_checkpoint_path)
        self.serialize_data()
        
        del self.separator_state
        self.separator_state = None

        #Data augmentations
        self.mixup = Mixup(self.samples, self.class_to_idx, cfg)
        audio_augs = {
                SyntheticNoise  : cfg.noise_p,
                RandomEQ        : cfg.rand_eq_p,
                LowpassFilter   : cfg.lowpass_p,
                HighpassFilter  : cfg.highpass_p,
                BackgroundNoise : cfg.bg_noise_p
            }.items()
        # List around aug(cfg) is necessary
        # because RandomApply expects an iterable
        self.audio_augmentations = torch.nn.Sequential(
                *[RandomApply([aug(cfg)], p=p) for aug, p in audio_augs]
            )

        self.image_augmentations = torch.nn.Sequential(
                RandomApply([audtr.FrequencyMasking(cfg.freq_mask_param)], p=cfg.freq_mask_p),
                RandomApply([audtr.TimeMasking(cfg.time_mask_param)],      p=cfg.time_mask_p))
    
    def verify_audio(self) -> None:
        """
        Checks to make sure files exist that are referenced in input df
        """
        missing_files = pd.Series(self.samples[cfg.file_name_col].unique()) \
            .progress_apply(
                lambda file: "good" if os.path.join(
                    cfg.data_path,file
                ) in self.data_dir else file
        )
        missing_files = missing_files[missing_files != "good"].unique()
        if missing_files.shape[0] > 0:
            logger.info("ignoring %d missing files", missing_files.shape[0])
            logger.debug("Missing files are: %s", str(missing_files))
        self.samples = self.samples[
            ~self.samples[cfg.file_name_col].isin(missing_files)
        ]

    def process_audio_file(self, file_name: str) -> pd.Series:
        """
        Save waveform of audio file as a tensor and save that tensor to .pt
        """
        exts = "." + file_name.split(".")[-1]
        new_name = file_name.replace(exts, ".pt")
        if not cfg.force_pt_regen and os.path.join(cfg.data_path, new_name) in self.data_dir:
            #ASSUME WE HAVE ALREADY PREPROCESSED THIS CORRECTLY
            return pd.Series({
                "FILE NAME": file_name,
                "files": new_name
            }).T


        try:
            # old error: "load" is not a known member of module "torchaudio"
            # Load is a known member of torchaudio:
            # https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html#loading-audio-data
            audio, sample_rate = torchaudio.load(       #pyright: ignore [reportGeneralTypeIssues ]
                os.path.join(cfg.data_path, file_name)
            )

            if len(audio.shape) > 1:
                audio = utils.to_mono(audio)

            # Resample
            if sample_rate != cfg.sample_rate:
                resample = audtr.Resample(sample_rate, cfg.sample_rate)
                audio = resample(audio)
            
            sep = self.separator_state.session.run(
                self.separator_state.output_tensor,
                feed_dict={
                    self.separator_state.audio_placeholder: audio[
                        np.newaxis, np.newaxis, :
                    ]
                },
            )
            sep = torch.Tensor(sep[0])
            #print(type(sep),sep.shape)
            audio = torch.stack([audio, sep[0], sep[1]])#, sep[2], sep[3]])
            #del sep
            #print(audio.shape)
            #exit()
            torch.save(audio, os.path.join(cfg.data_path,new_name))
            self.data_dir.add(new_name)
        # IO is messy, I want any file that could be problematic
        # removed from training so it isn't stopped after hours of time
        # Hence broad exception
        # pylint: disable-next=W0718
        except Exception as exc:
            logger.debug("%s is bad %s", file_name, exc)
            return pd.Series({
                "FILE NAME": file_name,
                "files": "bad"
            }).T


        return pd.Series({
                "FILE NAME": file_name,
                "files": new_name
            }).T


    def serialize_data(self) -> None:
        """
        For each file, check to see if the file is already a presaved tensor
        If the files is not a presaved tensor and is an audio file, convert to tensor to make
        Future training faster
        """
        self.verify_audio()
        files = pd.DataFrame(self.samples[cfg.file_name_col].unique(),
            columns=["files"]
        )


        files = files["files"].progress_apply(self.process_audio_file)
        logger.debug("%s", str(files.shape))

        num_files = files.shape[0]
        if num_files == 0:
            raise FileNotFoundError("There were no valid filepaths found, check csv")

        files = files[files["files"] != "bad"]
        self.samples = self.samples.merge(files, how="left",
                       left_on=cfg.file_name_col,
                       right_on="FILE NAME").dropna()

        logger.debug("Serialized form, fixed size: %s", str(self.samples.shape))

        if "files" in self.samples.columns:
            self.samples[cfg.file_name_col] = self.samples["files"].copy()
        if "files_y" in self.samples.columns:
            self.samples[cfg.file_name_col] = self.samples["files_y"].copy()

        self.samples["original_file_path"] = self.samples[cfg.file_name_col]

    def __len__(self):
        return self.samples.shape[0]

    def to_image(self, audio):
        """
        Convert audio clip to 3-channel spectrogram image
        """
        convert_to_mel = audtr.MelSpectrogram(
                sample_rate=cfg.sample_rate,
                n_mels=cfg.n_mels,
                n_fft=cfg.n_fft)
        convert_to_mel = convert_to_mel.to(self.device)

        # idk how to make this less expensive sorry
        # sep = separator_state.session.run(
        #     separator_state.output_tensor,
        #     feed_dict={
        #         separator_state.audio_placeholder: audio[
        #             np.newaxis, np.newaxis, :
        #         ]
        #     },
        # )
        # sep = sep[0]
        # chan0 = convert_to_mel(torch.from_numpy(sep[0,:]).clone())
        # chan1 = convert_to_mel(torch.from_numpy(sep[1,:]).clone())
        # image = torch.stack([convert_to_mel(audio), chan0, chan1])
        # del sep
        image = convert_to_mel(audio)
        print(audio.shape)
        print(image.shape)
        exit()
        # Mel spectrogram
        # Pylint complains this is not callable, but it is a torch.nn.Module
        # pylint: disable-next=not-callable
        # mel = convert_to_mel(audio)
        # print(mel.shape)
        # # Convert to Image
        # image = torch.stack([mel, mel, mel])
        
        # Convert to decibels
        # Log scale the power
        decibel_convert = audtr.AmplitudeToDB(stype="power")
        image = decibel_convert(image)
        
        # Normalize Image
        # Inspired by
        # https://medium.com/@hasithsura/audio-classification-d37a82d6715
        mean = image.mean()
        std = image.std()
        image = (image - mean) / (std + 1e-6)
        
        # Sigmoid to get 0 to 1 scaling (0.5 becomes mean)
        image = torch.sigmoid(image)
        return image


    def __getitem__(self, index): #-> Any:
        """ Takes an index and returns tuple of spectrogram image with corresponding label
        """
        assert isinstance(index, int)
        #print("get_annotation")
        audio, target = utils.get_annotation(
                df = self.samples,
                index = index,
                class_to_idx = self.class_to_idx)
        #print(audio)
        #print("SUCCESSFULLY GOT AUDUIO")
        audio, target = self.mixup(audio, target)
        if self.train:
            audio = self.audio_augmentations(audio)

        #print("GET IMAGE")
        image = self.to_image(audio)

        if self.train:
            image = self.image_augmentations(image)

        if image.isnan().any():
            logger.error("ERROR IN ANNOTATION #%s", index)
            self.bad_files.append(index)
            image = torch.zeros(image.shape)
            target = torch.zeros(target.shape)

        #If dataframe has saved onehot encodings, return those
        #Assume columns names are species names
        if  self.onehot:
            target = self.samples.loc[index, self.classes].values.astype(np.int32)
            target = torch.Tensor(target)

        target = torch.minimum(target, torch.ones(target.shape[0]))
        return image, target

    def get_num_classes(self) -> int:
        """ Returns number of classes
        """
        return self.num_classes

    def get_sample_weights(self) -> pd.Series:
        """ Returns the weights as computed by the first place winner of BirdCLEF 2023
        See https://www.kaggle.com/competitions/birdclef-2023/discussion/412808 
        Congrats on your win!
        """
        manual_id = cfg.manual_id_col
        all_primary_labels = self.samples[manual_id]
        sample_weights = (
            all_primary_labels.value_counts() / 
            all_primary_labels.value_counts().sum()
        )  ** (-0.5)
        weight_list = self.samples[manual_id].apply(lambda x: sample_weights.loc[x])
        return weight_list


def get_datasets() -> Tuple[PyhaDFDataset, PyhaDFDataset, Optional[PyhaDFDataset]]:
    """ Returns train and validation datasets
    does random sampling for train/valid split
    adds transforms to dataset
    """
    train_p = cfg.train_test_split
    path = cfg.dataframe_csv
    # Load the dataset
    if cfg.is_unchunked:
        data = pd.read_csv(path, usecols = [
            cfg.file_name_col,
            cfg.manual_id_col,
            cfg.offset_col,
            cfg.duration_col,
            "CLIP LENGTH"
        ], dtype={
            cfg.file_name_col: str,
            cfg.manual_id_col: str,
            cfg.offset_col: float,
            cfg.duration_col: float,
            "CLIP LENGTH": float})
        logger.info("Chunking with sliding windows")
        data = sliding_chunks.dynamic_yan_chunking(
            data,
            chunk_length_s=cfg.chunk_length_s,
            min_length_s=cfg.min_length_s,
            overlap=cfg.overlap,
            chunk_margin_s=cfg.chunk_margin_s,
            only_slide=False)
        logger.info("Chunking completed")
    else:
        data = pd.read_csv(path, usecols = [
            cfg.file_name_col,
            cfg.manual_id_col,
            cfg.offset_col,
            cfg.duration_col,

        ], dtype={
            cfg.file_name_col: str,
            cfg.manual_id_col: str,
            cfg.offset_col: float,
            cfg.duration_col: float,
           })

    # Get classes list
    data[cfg.manual_id_col] = data[cfg.manual_id_col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("{") else x
    )
    if cfg.class_list is not None:
        classes = cfg.class_list
    else:
        classes = set()
        for species in data[cfg.manual_id_col].dropna():
            if isinstance(species, dict):
                classes.update(species.keys())
            else:
                if not species is None:
                    classes.add(species)
        classes = list(classes)
        classes.sort()
        # pylint: disable-next=attribute-defined-outside-init
        cfg.config_dict["class_list"] = classes
        wandb.config.update({"class_list": classes}, allow_val_change=True)

    #for each species, get a random sample of files for train/valid split
    train_files = data.groupby(cfg.manual_id_col, as_index=False).apply(
        lambda x: pd.Series(x[cfg.file_name_col].unique()).sample(frac=train_p)
    )
    train = data[data[cfg.file_name_col].isin(train_files)]

    valid = data[~data.index.isin(train.index)]
    train_ds = PyhaDFDataset(train, train=True, species=classes)

    valid_ds = PyhaDFDataset(valid, train=False, species=classes)



    #Handle inference datasets
    if cfg.infer_csv is None:
        infer_ds = None
    else:
        infer = pd.read_csv(cfg.infer_csv)
        infer_ds = PyhaDFDataset(infer, train=False, species=classes, onehot=True)



    return train_ds, valid_ds, infer_ds

def set_torch_file_sharing(_) -> None:
    """
    Sets torch.multiprocessing to use file sharing
    """
    torch.multiprocessing.set_sharing_strategy("file_system")


def make_dataloaders(train_dataset, val_dataset, infer_dataset
        )-> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
        Loads datasets and dataloaders for train and validation
    """


    # Create our dataloaders
    # if sampler function is "specified, shuffle must not be specified."
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    
    if cfg.does_weighted_sampling:
        if train_dataset.samples[cfg.manual_id_col].any(lambda x: isinstance(x,dict)):
            raise NotImplementedError("Weighted sampling not implemented for overlapping targets")
        # Code used from:
        # https://www.kaggle.com/competitions/birdclef-2023/discussion/412808
        # Get Sample Weights
        weights_list = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights_list, len(weights_list))
        train_dataloader = DataLoader(
            train_dataset,
            cfg.train_batch_size,
            sampler=sampler,
            num_workers=cfg.jobs,
            worker_init_fn=set_torch_file_sharing
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.jobs,
            worker_init_fn=set_torch_file_sharing
        )

    val_dataloader = DataLoader(
        val_dataset,
        cfg.validation_batch_size,
        shuffle=False,
        num_workers=cfg.jobs,
    )
    if infer_dataset is None:
        infer_dataloader = None
    else:
        infer_dataloader = DataLoader(
                infer_dataset,
                cfg.validation_batch_size,
                shuffle=False,
                num_workers=cfg.jobs,
                worker_init_fn=set_torch_file_sharing
            )
    return train_dataloader, val_dataloader, infer_dataloader

def main() -> None:
    """
    testing function.
    """
    # run = wandb.init(
    #         entity=cfg.wandb_entity,
    #         project=cfg.wandb_project,
    #         config=cfg.config_dict,
    #         mode="online" if cfg.logging else "disabled")
    # run.name = "inference testing"
    # torch.multiprocessing.set_start_method('spawn')
    # utils.set_seed(cfg.seed)
    # _, _, infer_dataloader = get_datasets()
    # for _, (_, _) in enumerate(infer_dataloader):
    #     break
if __name__ == '__main__':
    main()
