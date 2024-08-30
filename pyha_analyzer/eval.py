"""
    Contains the training and validation function and logging to Weights and Biases
    Methods:
        train: trains the model
        valid: calculates validation loss and accuracy
        set_seed: sets the random seed
"""
import datetime
import logging
import os
from typing import Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch._dynamo.config
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm
import wandb
import pandas as pd
import ast

from pyha_analyzer import config
from pyha_analyzer.dataset import get_datasets, make_dataloaders, PyhaDFDataset
from pyha_analyzer.utils import set_seed

from datasets import Dataset, DatasetDict, ClassLabel, Features, Value, Audio, Sequence

from huggingface_hub import notebook_login

notebook_login()

tqdm.pandas()
time_now  = datetime.datetime.now().strftime('%Y%m%d-%H%M')
cfg = config.cfg
logger = logging.getLogger("acoustic_multiclass_training")

def set_name(run):
    """
    Set wandb run name
    """
    if cfg.wandb_run_name == "auto":
        # This variable is always defined
        cfg.wandb_run_name = cfg.model # type: ignore
    run.name = f"{cfg.wandb_run_name}-{time_now}"
    return run

def logging_setup() -> None:
    """ Setup logging on the main process
    Display config information
    """
    file_handler = logging.FileHandler("recent.log", mode='w')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.debug("Debug logging enabled")
    logger.debug("Config: %s", cfg.config_dict)
    logger.debug("Git hash: %s", cfg.git_hash)

def main(in_sweep=True) -> None:
    """ Main function
    """
    logger.info("Device is: %s, Preprocessing Device is %s", cfg.device, cfg.prepros_device)
    set_seed(cfg.seed)

    if in_sweep:
        run = wandb.init()
        for key, val in dict(wandb.config).items():
            setattr(cfg, key, val)
        wandb.config.update(cfg.config_dict)
    else:
        run = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            config=cfg.config_dict,
            mode="online" if cfg.logging else "disabled")
        set_name(run)

    # Load in dataset
    logger.info("Loading Dataset...")

    path = cfg.dataframe_csv
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
    
    data[cfg.manual_id_col] = data[cfg.manual_id_col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("{") else x
    )
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
    
    data[cfg.file_name_col] = data[cfg.file_name_col].str.replace(r'^T:\\Peru\\', '/home/hejonathan/', regex=True)
    data[cfg.file_name_col] = data[cfg.file_name_col].str.replace(r'\\', '/', regex=True)
    
    print('data:', data)
    val_dataset = PyhaDFDataset(data, train=False, species=classes, cfg=cfg)
    print(val_dataset)
    
    def pytorch_dataset_to_hf_dataset(pytorch_dataset):
        def generator():
            for i in range(len(pytorch_dataset)):
                audio, target, file_name = pytorch_dataset[i]
                audio = audio.numpy().astype(np.float32)
                yield {
                    'audio': {'array': audio, 'path': file_name, 'sampling_rate': 16000},
                    'file': file_name,
                    'label': int(target)  # Ensure target is an integer
                }

        features = Features({
            'audio': Audio(sampling_rate=16000),
            'file': Value('string'),
            'label': ClassLabel(names=classes)  # Use the ClassLabel feature here
        })

        hf_dataset = Dataset.from_generator(generator, features=features).with_format('torch')
        return hf_dataset
    

    #hf_train_ds = pytorch_dataset_to_hf_dataset(train_dataset).cast_column('label', ClassLabel(names=classes))
    hf_valid_ds = pytorch_dataset_to_hf_dataset(val_dataset).cast_column('label', ClassLabel(names=classes))

    dataset = DatasetDict({
        'validation': hf_valid_ds,
    })

    print(dataset)

    print(classes)

    dataset["validation"].features["label"] = classes

    labels = dataset["validation"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label


    ## Training
    #model_checkpoint = "MIT/ast-finetuned-audioset-10-10-0.4593"
    model_checkpoint = "ast-finetuned-audioset-10-10-0.4593-bs8-lr5e-06/checkpoint-24000"
    from transformers import AutoFeatureExtractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    print(feature_extractor)
    max_duration = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    def preprocess_function(examples):
        audio_arrays = [np.array(x["array"]) for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            truncation=True,
        )
        return inputs
    
    encoded_dataset = dataset.map(preprocess_function, remove_columns=["audio", "file"], batched=True)
    
    from transformers import ASTForAudioClassification, TrainingArguments, Trainer

    num_labels = len(id2label)
    model = ASTForAudioClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )
    model.to(device)



    model_name = model_checkpoint.split("/")[-1]

    bs = 24
    lr = 1e-5

    args = TrainingArguments(
        f"{model_name}-bs{bs}-lr{lr}",
        eval_strategy = "steps",
        eval_steps = 8000,
        save_strategy = "steps",
        save_steps = 8000,
        learning_rate=lr,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=bs,
        num_train_epochs=3,
        warmup_ratio=0.125,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="precision",
        push_to_hub=False,
        gradient_checkpointing=False,
        fp16 = True,
        torch_compile=True,
        save_safetensors=False
    )

    from evaluate import load
    from scipy.special import softmax
    # Load the metric for evaluation
    metric_precision = load("precision", trust_remote_code=True)
    metric_recall = load("recall", trust_remote_code=True)
    metric_f1 = load("f1", trust_remote_code=True)
    metric_roc_auc = load("roc_auc", "multiclass", trust_remote_code=True)
    metric_accuracy = load("accuracy", trust_remote_code=True)
    
    all_logits = []
    all_labels = []
    # Define the compute_metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        prob = softmax(logits, axis=-1)

        precision = metric_precision.compute(predictions=predictions, references=labels, average='macro')
        recall = metric_recall.compute(predictions=predictions, references=labels, average='macro')
        f1 = metric_f1.compute(predictions=predictions, references=labels, average='macro')
        accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
        
        all_logits.append(logits)
        all_labels.append(labels)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
        }


    trainer = Trainer(
        model,
        args,
        #train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    torch._dynamo.config.suppress_errors = True
    trainer.evaluate()

    import pickle
    pickle_file = f'{model_checkpoint}/logits_labels_soundscape.pkl'

    data_to_pickle = {
        'logits': all_logits,
        'labels': all_labels,
        'id2label': id2label
    }

    # Save the lists to a pickle file
    with open(pickle_file, 'wb') as file:
        pickle.dump(data_to_pickle, file)
    


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    main(in_sweep=False)
