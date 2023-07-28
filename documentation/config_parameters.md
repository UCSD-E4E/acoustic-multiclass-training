# Config parameters
These parameters are stored in the `config.yml` file. Default values can be found at `documentation/default_config.yml`.

## Data paths
`dataframe_csv`: Path to the CSV file containing training data labels\
`data_path`: Path to the folder containing all the audio clips to be trained on\
`bg_noise_path`: Path to a folder containing audio clips for background noise in the BackgroundNoise data augmentation. Optional if the `bg_noise_p` is set to zero.

## Dataframe columns
`offset_col`: Name of the column with the within-clip offset of each chunk in seconds\
`duration_col`: Name of the column with the length of the chunk in seconds\
`file_name_col`: Name of the column with the name of the file the chunk came from\
`manual_id_col`: Name of the column with the manual id of the chunk\
`class_list`: Ordered list of class names to use for get_test_map.py. If left blank, will use automatically generated

## Audio parameters
`chunk_length`: Chunk length in seconds\
`sample_rate`: Target sample rate for loading clips

## System parameters
`prepros_device`: Device that preprocessing occurs on. Should never be changed from `"cpu"` unless CPU processing is limited.\
`device`: Determines device that training is performed on. If `"auto"`, will default to `"cuda"` if available, or `"cpu"` if not.\
`jobs`: Number of multiprocessing jobs for data loader\
`mixed_precision`: Use mixed precision in model training

## Training and model parameters
`train_batch_size`: Batch size for training\
`train_test_split`: What proportion of the dataset to use for training\
`do_weighted_sampling`: Whether to more frequently sample from less common classes\
`epochs`: Number of epochs to train for\
`learning_rate`: Learning rate of model\
`loss_fnc`: Loss function to use. Can be any of the following:
- "CE": cross entropy loss
- "BCE": binary cross entropy loss
- "BCEWL": binary cross entropy loss with logits
- "FL": focal loss

`model`: Which model architecture to use. Can be any of the following:
 - "eca\_nfnet\_l0" (recommended)
 - "tf\_efficientnet\_b4" (recommended)
 - "convnext\_tiny"
 - "resnetv2\_101"
 - "seresnext101\_32x4d"
 - "rexnet\_200"
 - "mobilenetv3\_large\_100\_miil\_in21k"  


 In addition, any model compatible with `timm` can be used. An input size of 224x224 is required.\
`model_checkpoint`: Path from which to load starting model weights. Optional.

## Validation parameters
`validation_batch_size`: Batch size for validation\
`valid_freq`: How often to perform validation. Unit is batches.\
`valid_dataset_ratio`: Proportion of the validation dataset to use for validation within an epoch. In between epochs, the entire validation dataset will be used every time.\
`num_folds`: Number of folds to use in validation

## Logging
`logging`: Whether to log runs in Weights and Biases. If true, requires `wandb_entity`, `wandb_project`, `wandb_run_name` to be set.\
`logging_freq`: How often to log model metrics with Weights and Biases. Unit is batches.\
`wandb_entity`: Entity name to log in Weights and Biases\
`wandb_project`: Project name to log in Weights and Biases\
`wandb_run_name`: Run name to log in WandB. Set to "auto" in order to automatically generate a unique run name\
`debug`: Whether to print all logger warnings

## Early stopping
`early_stopping`: Whether to perform early stopping\
`patience`: How many validation rounds can see no improvement in mAP before the run is stopped\
`min_valid_map_delta`: Minimum change in validation mAP that is considered an improvement

## Sweep settings
`sweep_id`: Weights and Biases sweep id. Leave blank if you would like to start a new sweep. To run another agent for a preexisting sweep, set this parameter to the sweep id\
`agent_run_count`: Number of times to run an agent

## Data augmentation probabilities
`mixup_p`: probability of applying mixup augmentation\
`noise_p`: probability of applying synthetic noise augmentation\
`freq_mask_p`: probability of applying frequency masking augmentation\
`time_mask_p`: probability of applying time masking augmentation\
`rand_eq_p`: probability of applying random equalization augmentation\
`lowpass_p`: probability of applying lowpass filter augmentation\
`highpass_p`: probability of applying highpass filter augmentation\
`bg_noise_p`: probability of applying background noise augmentation

## Data augmentation parameters
If a parameter specifies a range, then it is always a length two list with the first element specifying the minimum and the second specifying the maximum. Values will be uniformly sampled from this range.\
`noise_type`: Type of noise to use in the SyntheticNoise augmentation. Can be any of the following:
- "white"
- "violet"
- "blue"
- "pink"
- "brown"

`noise_alpha`: Strength of noise in an augmented clip after applying SyntheticNoise augmentation. Must be between 0 and 1\
`freq_mask_param`: How many sequential pixels to mask in the FrequencyMasking augmentation\
`time_mask_param`: How many sequential pixels to mask in the TimeMasking augmentation\
`mixup_alpha_range`: Range for the alpha parameter of Mixup. The alpha parameter gives the strength of the other clip in the augmented clip after applying Mixup. Min/max values are between 0 and 1\
`bg_noise_alpha_range`: Range for the alpha parameter of BackgroundNoise. The alpha parameter gives the strength of the other clip in the augmented clip after applying BackgroundNoise. Min/max values are between 0 and 1\
`rand_eq_f_range`: Range for the frequency parameter of RandomEQ. Min/max must be greater than zero\
`rand_eq_q_range`: Range for the Q parameter of RandomEQ. Must be greater than zero\
`rand_eq_g_range`: Range for the gain parameter of RandomEQ\
`rand_eq_iters`: Number of times to apply RandomEQ to any given clip. All RandomEQ parameters are resampled at each application. Must be nonnegative\
`lowpass_cutoff`: Cutoff frequency for the LowpassFilter augmentation\
`lowpass_q_val`: Q value for the LowpassFilter augmentation\
`highpass_cutoff`: Cutoff frequency for the HighpassFilter augmentation\
`highpass_q_val`: Q value for the HighpassFilter augmentation

## Spectrogram conversion settings
`n_hops`: Hop size in samples for fft. Do not change\
`n_mels`: Number of mel filterbanks to use in converting audio to spectrogram\
`n_fft`: Size of FFT in mel spectrogram conversion. Creates n\_fft // 2 + 1 bins
