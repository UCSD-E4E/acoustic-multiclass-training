# Acoustic Species Classification - Bird Team 1

![overview](../images/header.png)

Welcome to the Github repository for Acoustic Species Classification - Bird Team 1, as part of the UC San Diego CSE 145/237D course project. Below is an overview of the project itself, details on navigating the repository and reproducing results, and links to written reports on the project throughout the quarter.

## Contents
- [Project Overview](#project-overview)
- [Replication](#replication)
- [Documentation](#documentation)

## Project Overview

Passive acoustic monitoring (PAM) plays a crucial role in conservation efforts and the preservation of biodiversity. By capturing and analyzing the sounds produced by avian species in their natural habitats, this non-invasive method provides valuable insights into population dynamics, species richness, and habitat quality, as birds may act as key indicators of larger environmental effects. However, as PAM systems may collect terabytes of noisy audio data, with other species and human encroachment polluting the recordings, extracting useful information from such audio recordings (i.e. where are birds present in the recording, what species of birds are they, etc.) remains an open problem. 

Here, we present our joint work with the UCSD [Engineers for Exploration](https://e4e.ucsd.edu/), [PyHa](https://github.com/UCSD-E4E/PyHa), and the [BirdCLEF2023 Kaggle Competition](https://www.kaggle.com/competitions/birdclef-2023), where we have designed a full pipeline for processing noisy audio recordings to train an acoustic bird species classifier.

![outline](../images/main_diag.png)

Our main pipeline (shown above) can be described as follows:
1. For a given set of weakly-labeled noisy audio recordings (i.e. the entire recording may have a label for a bird species, but no information about where in the recording the call is), we use [PyHa](https://github.com/UCSD-E4E/PyHa) to extract 5s segment mel-spectrograms of the original audio, where each 5s segment is estimated to  include the bird call matching the given label.
2. We use this strongly labeled mel-spectra data to train a bird species classifier (as well as an optional bird detector), which at inference time is given an unlabeled 5s audio clip and predicts which species are present in the audio.

A detailed description of the project and producing it is shown below.

## Replication

### Installation

We recommend that you use miniconda for all package management. Once miniconda is downloaded (see [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for information on miniconda installation), run the following command to setup the conda environment for the repo:

```bash
conda env create -f environment.yml
```

In order to recreate our results, [PyHa](https://github.com/UCSD-E4E/PyHa) needs to be installed and set up. Furthermore, our results are based off of the [BirdCLEF2023 dataset](https://www.kaggle.com/competitions/birdclef-2023). You may also find it useful to use a  [no-call dataset](https://www.kaggle.com/code/sprestrelski/birdclef23-uniform-no-call-sound-chunks) compiled from previous competitions.

### Data Setup
The data processing pipeline assume a folder directory structure as follows
```
train_audio
├── abethr1
|   ├── XC128013.wav
|   ├── XC363502.wav 
|   └── XC363504.wav
├── barswa
|   ├── XC113914.wav  
|   ├── XC208241.wav  
|   └── XC324914.wav
└── carwoo1
    ├── XC126500.wav  
    └── XC294063.wav
```
We ran into issues running PyHa over `.ogg` files, so there is an included function in `gen_tweety_labels.py` to convert `.ogg` to `.wav` files and can be swapped out for your original filetype. This is an issue for the data processing pipeline. However, the training pipeline is able to predict on most filetypes.

### Data Processing

The first file in our data processing pipeline is `gen_tweety_labels.py`. After downloading and setting up PyHa, copy this script into the PyHa directory and cd into it. If PyHa was correctly set up, this script will run TweetyNet on the entire BirdCLEF2023 dataset in order to produce binary labels in a file called `BirdCLEF2023_TweetyNet_Labels.csv`. For example, if the BirdCLEF2023 directory called `train_audio` is located at `/share/acoustic_species_id`, the script can be run with the following command:

```bash
python gen_tweety_labels.py /share/acoustic_species_id
```

After generating the TweetyNet labels, we next run `attach_strong_labels.py`, we attach the strong labels as given from the `train_metadata.csv` included in the BirdCLEF2023 directory. This will produce a file called `BirdCLEF2023_Strong_Labels.csv`. Remember to include the path in the script as follows:

```bash
python attach_strong_labels.py /share/acoustic_species_id
```

Next, we need to chunk the data since some files are longer than the 5 second duration used at inference time. The script `gen_chunks_from_labels.py` chunks the clips in `BirdCLEF2023_Strong_Labels.csv` and outputs all of the chunks in a new directory called `BirdCLEF2023_train_audio_chunks`. To generate these chunks, we run the following command:

```bash
python gen_chunks_from_labels.py /share/acoustic_species_id
```

Next, we need to split the data into training and validation sets. These splits can either be done manually by putting clips in train/validation folders, or doing a random shuffle split. However, multiple audio chunks from a single file should be kept together in their respective folders to avoid data leakage. To do so automatically, we run the `distribute_chunks.py` script, which first distributes all of the audio files into 4:1 training/validation splits, and then distributes all of the chunks according to the file split. These chunks are stored in a new directory called `BirdCLEF2023_split_chunks`. We do so using the following command:

```bash
python distribute_chunks.py /share/acoustic_species_id
```


### Classification
The main file is `train.py`, which has the main training loop and uses functions from `dataset.py` and `model.py`. This has a number of hyperparameters related to training, logging, and data augmentation that can be passed in as arguments. For example, to run with a mixup probability of 0.6, with all other arguments kept to the defaults, you would run:

```py
python train.py –mix_p=0.6
```

See the top of `train.py` for more information on the hyperparameters.

In order to make sure the data loads correctly, lines 428-429 in `dataset.py` must be modified to point to the correct folders which house the train and validation splits for the dataset. For example:

```py
train_data = BirdCLEFDataset(root="./BirdCLEF2023_split_chunks/training", CONFIG=CONFIG)
val_data = BirdCLEFDataset(root="./BirdCLEF2023_split_chunks/validation", CONFIG=CONFIG)
```

To select a model, add a `model_name` parameter in `CONFIG` when instantiating `BirdCLEFModel`, or edit the `model.py` file directly. `model.py` loads in models using the [Pytorch Image Models library (timm)](https://timm.fast.ai/) and supports a variety of models including EfficientNets, ResNets, and DenseNets. To directly load local models, you would add another parameter for the checkpoint path:
```py
self.model = timm.create_model('tf_efficientnet_b1', checkpoint_path='./models/tf_efficientnet_b1_aa-ea7a6ee0.pth')
```

#### Logging
This project is set up with [WandB](https://wandb.ai), a dashboard to keep track of hyperparameters and system metrics. You’ll need to make an account and login locally to use it. WandB is extremely helpful for comparing models live and visualizing the training process.

![](../images/SampleWandBOutputs.PNG)
 
#### Training Binary Classification Models
To train a binary classification rather than multi-class model, merge all “bird” chunks into a single folder and “no bird”/“no call”/”noise” into another. In `CONFIG`, set `num_classes` to 2. The directory structure should look as follows:
```
train_audio
├── bird
|   ├── XC120632_2.wav
|   ├── XC120632_3.wav 
|   └── XC603432_1.wav
└── no_bird
    ├── nocall_122187_0.wav
    └── noise_30sec_1084_4.wav
```
You can produce your own no-call dataset from [this notebook](https://www.kaggle.com/code/sprestrelski/birdclef23-uniform-no-call-sound-chunks), which pulls data from previous BirdCLEF competitions and DCASE 2018

### Inference 
The `inference.ipynb` notebook can be directly uploaded to and run on Kaggle. In the import section, the notebook takes in a local path to a pretrained checkpoint (can be replaced with a `timm` fetched model) with the model architecture and to the final model. Replicate any changes you made to the BirdCLEFModel class, or directly import from `train.py` if running on a local machine.

Under the inference section, modify the `pd.read_csv` line to your training metadata file. This is used to get a list of labels to predict. Also, change the `filepaths` variable to where your test data is stored. The given notebook removes two classes from the predictions, as there was no training data actually used (chunks were not able to generate), but these can be removed. The final output is `submission.csv`, which outputs probabilistic predictions for each class for every 5 second chunk of the training data.

## Documentation
- [Project Specification](https://drive.google.com/file/d/1fRkkE5k19Y1tkMqt09pPONkNvDmWV2ux/view?usp=share_link)
- [Milestone Report](https://drive.google.com/file/d/1o02SRfyTS3GIVdu1qpCXBZATXyBO_V9d/view?usp=share_link)
- [Final Oral Presentation](https://docs.google.com/presentation/d/15ZEWugpDqcjfeiNdGHxnlNMUc3dn_99cYGKaOZddJ8Q/edit?usp=sharing)
- [Technical Report](https://drive.google.com/file/d/1SXEis3fDLvjq8cCrmwD2GHVhplUxQ8rj/view?usp=sharing)
  
## Contributors
- [Samantha Prestrelski](https://github.com/sprestrelski)
- [Lorenzo Mendes](https://github.com/lmendes14)
- [Arthi Haripriyan](https://github.com/aharipriyan)
- [Zachary Novack](https://github.com/ZacharyNovack)
