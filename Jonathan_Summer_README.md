# AST Documentation
## General Nodes:
- This is an adaptation from the original training pipeline. The original pipeline is cut in half and the rest of the huggingface portion is added (like a mermaid)
## `dataset.py`
- The original pipeline takes care of the data preparation. While mostly the same, some of the changes to `dataset.py` are: 
1. in `get_datasets()`, a snippet that changes the filenames of the mathias dataset are added in line 361, this is only for running the evaluation, if in training using xeno canto, this snippet should be commented. 
2. in `__getitem__()`, the parts for data augmentation and audio mixup are commented out (huggingface pipeline's AST does not support multilabel). Additionally, this function returns the original audio rather than the image (`ASTFeatureExtractor` takes care of the spectrogram)
## `train.py`
- All of the helper functions in this file have been unused and deleted. Everything resides in the main function. 
- The section from the original pipeline is from line 63 to line 83, calling `get_dataset` and getting the pytorch dataset. 
- Lines from 86 to 112 is the conversion, where `Dataset.from_generator` calls `__getitem__` from `dataset.py` and does the conversion
- From then on, line 140 to 152 is the spectrogram generation using AST's provided feature extractor, this is done as a part of the preprocessing step.
- Lines 194 to 218 is the metric computation, where `compute_metrics` is called during the eval step, and is modified so that it also saves the model's predictions. 
## training parameters
- Rather than using `config.yml`, most of the training parameters are set in line 171
- From experimentation, there does not seem to be a significatn affect on learning rate or batch size on the model's performance. Since the largest batch size possible through experimentation seems to be 10 (in comparison to original AST's 12) However, the pipeline also uses gradient accumulation step of 2, so the effective batch size is twice the actual batch size.
- The biggest limitation of the pipeline is training speed, it can currently hit about 1.15 it/s, which is a significant increase from the raw adoptation of the AST on the huggingface pipeline at ~8 s/it. This is done through a series of experimenets. Among them, the important ones are mixed precision (`fp16=True`), and torch optimization (`torch_compile=True`). 
## `eval.py`
- this is basically the same as `train.py` except in line 268, where `train()` was changed to `evaluate()`. This file is mainly for running inference on the soundscape.
- To run evaluation, change the relevant filepaths in config.yml, and do `python entry_eval.py` rather than `python entry.py`. Additionally, lines 361 to 364 in `dataset.py` should be uncommented. 