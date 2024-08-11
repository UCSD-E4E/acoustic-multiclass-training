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
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm
import wandb

from pyha_analyzer import config
from pyha_analyzer.dataset import get_datasets, make_dataloaders, PyhaDFDataset
from pyha_analyzer.utils import set_seed
from pyha_analyzer.models.early_stopper import EarlyStopper
from pyha_analyzer.models.timm_model import TimmModel

from datasets import Dataset, DatasetDict, ClassLabel, Features, Value, Audio, Sequence

from huggingface_hub import notebook_login

notebook_login()

tqdm.pandas()
time_now  = datetime.datetime.now().strftime('%Y%m%d-%H%M')
cfg = config.cfg
logger = logging.getLogger("acoustic_multiclass_training")

def run_batch(model: TimmModel,
                mels: torch.Tensor,
                labels: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Runs the model on a single batch 
        Args:
            model: the model to pass the batch through
            mels: single batch of input data
            labels: single batch of expecte output
        Returns (tuple of):
            loss: the loss of the batch
            outputs: the output of the model
    """
    mels = mels.to(cfg.device)
    labels = labels.to(cfg.device)
    if cfg.device == "cpu": 
        dtype = torch.bfloat16
    else: 
        dtype = torch.float16
    if cfg.device == "cuda":
        with autocast(device_type=cfg.device, dtype=dtype, enabled=cfg.mixed_precision):
            outputs = model(mels)
            loss = model.loss_fn(outputs, labels) # type: ignore
    else:
        outputs = model(mels)
        loss = model.loss_fn(outputs, labels) #type: ignore
        
    outputs = outputs.to(dtype=torch.float32)
    loss = loss.to(dtype=torch.float32)
    return loss, outputs

def map_metric(outputs: torch.Tensor,
               labels: torch.Tensor,
               class_dist: torch.Tensor) -> Tuple[float, float]:
    """ Mean average precision metric for a batch of outputs and labels 
        Returns tuple of (class-wise mAP, sample-wise mAP) """
    metric = MultilabelAveragePrecision(num_labels=len(class_dist), average="none")
    out_for_score = outputs.detach().cpu()
    labels_for_score = labels.detach().cpu().long()
    map_by_class = metric(out_for_score, labels_for_score)
    cmap = map_by_class.nanmean()
    smap = (map_by_class * class_dist/class_dist.sum()).nansum()
    # https://forums.fast.ai/t/nan-values-when-using-precision-in-multi-classification/59767/2
    # Could be possible when model is untrained so we only have FNs
    if np.isnan(cmap):
        return 0, 0
    return cmap.item(), smap.item()

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
def train(model: TimmModel,
        data_loader: DataLoader,
        valid_loader: DataLoader,
        infer_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        best_valid_cmap: float,
       ) -> float:
    """ Trains the model
    Returns new best valid map
    """
    logger.debug('size of data loader: %d', len(data_loader))
    model.train()

    log_n = 0
    log_loss = 0

    #scaler = torch.cuda.amp.GradScaler()
    start_time = datetime.datetime.now()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    log_pred = []
    log_labels = []

    for i, (mels, labels) in enumerate(data_loader):

        optimizer.zero_grad()

        loss, outputs = run_batch(model,  mels, labels)

        if cfg.mixed_precision and cfg.device != "cpu":
            # Pyright complains about scaler.scale(loss) returning iterable of unknown types
            # Problem in the pytorch typing, documentation says it returns iterables of Tensors
            #  keep if needed - noqa: reportGeneralTypeIssues
            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()
        else:
            if cfg.mixed_precision:
                logger.warning("cuda required, mixed precision not applied")
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        log_pred.append(torch.clone(F.sigmoid(outputs).cpu()).detach())
        log_labels.append(torch.clone(labels.cpu()).detach())
        log_loss += loss.item()
        log_n += 1

        if (i != 0 and i % (cfg.logging_freq) == 0) or i == len(data_loader) - 1:
            dataset: PyhaDFDataset = data_loader.dataset # type: ignore
            cmap, smap = map_metric(torch.cat(log_pred),torch.cat(log_labels),dataset.class_dist)
            duration = (datetime.datetime.now() - start_time).total_seconds()
            start_time = datetime.datetime.now()
            annotations = ((i % cfg.logging_freq) or cfg.logging_freq) * cfg.train_batch_size
            #Log to Weights and Biases
            wandb.log({
                "train/loss": log_loss / log_n,
                "train/mAP": cmap,
                "train/smAP": smap,
                "i": i,
                "epoch": epoch,
                "clips/sec": annotations / duration,
                "epoch_progress": epoch + float(i)/len(data_loader),
            })
            logger.info("i: %s   epoch: %s   clips/s: %s   Loss: %s   cmAP: %s   smAP: %s",
                str(i).zfill(5),
                str(round(epoch+float(i)/len(data_loader),3)).ljust(5, '0'),
                str(round(annotations / duration,3)).ljust(7), 
                str(round(log_loss / log_n,3)).ljust(5), 
                str(round(cmap,3)).ljust(5),
                str(round(smap,3)).ljust(5)
            )
            log_loss = 0
            log_n = 0
            log_pred = []
            log_labels = []

        if (i != 0 and i % (cfg.valid_freq) == 0):
            # Free memory so gpu is freed before validation run
            del mels
            del outputs
            del labels

            valid_start_time = datetime.datetime.now()
            _, best_valid_cmap = valid(model,
                                      valid_loader,
                                      infer_loader,
                                      epoch + i / len(data_loader),
                                      best_valid_cmap)
            model.train()
            # Ignore the time it takes to validate in annotations/sec
            start_time += datetime.datetime.now() - valid_start_time

    return best_valid_cmap

def valid(model: Any,
          data_loader: DataLoader,
          infer_loader: Optional[DataLoader],
          epoch_progress: float,
          best_valid_cmap: float = 1.0,
          ) -> Tuple[float, float]:
    """ Run a validation loop
    Arguments:
        model: the model to validate
        data_loader: the validation data loader
        epoch_progress: the progress of the epoch
            - Note: If this is an integer, it will run the full
                    validation set, otherwise runs cfg.valid_dataset_ratio
    Returns:
        Loss
    """
    model.eval()

    running_loss = 0
    log_pred, log_label = [], []
    dataset_ratio: float = cfg.valid_dataset_ratio
    if float(epoch_progress).is_integer():
        dataset_ratio = 1.0

    num_valid_samples = int(len(data_loader)*dataset_ratio)

    # tqdm is a progress bar
    dl_iter = tqdm(data_loader, position=5, total=num_valid_samples)

    with torch.no_grad():
        for index, (mels, labels) in enumerate(dl_iter):
            if index > num_valid_samples:
                # Stop early if not doing full validation
                break

            loss, outputs = run_batch(model, mels, labels)
                
            running_loss += loss.item()
            
            log_pred.append(torch.clone(outputs.cpu()).detach())
            log_label.append(torch.clone(labels.cpu()).detach())


    # softmax predictions
    log_pred = F.sigmoid(torch.cat(log_pred)).to(cfg.device)

    dataset: PyhaDFDataset = data_loader.dataset # type: ignore
    cmap, smap = map_metric(log_pred, torch.cat(log_label), dataset.class_dist)

    # Log to Weights and Biases
    wandb.log({
        "valid/loss": running_loss/num_valid_samples,
        "valid/map": cmap,
        "valid/smAP": smap,
        "epoch_progress": epoch_progress,
    })

    logger.info("Validation Loss:\t%f\nValidation cmAP:\t%f\nValidation smAP:\t%f", 
                running_loss/len(data_loader),
                cmap, smap)

    if cmap > best_valid_cmap:
        logger.info("Model saved in: %s", save_model(model))
        logger.info("Validation cmAP Improved - %f ---> %f", best_valid_cmap, cmap)
        best_valid_cmap = cmap


    inference_valid(model, infer_loader, epoch_progress, cmap)
    return cmap, best_valid_cmap


def inference_valid(model: Any,
          data_loader: Optional[DataLoader],
          epoch_progress: float,
          valid_cmap: float):

    """ Test Domain Shift To Soundscapes

    """
    if data_loader is None:
        return

    model.eval()

    log_pred, log_label = [], []

    num_valid_samples = int(len(data_loader))

    # tqdm is a progress bar
    dl_iter = tqdm(data_loader, position=5, total=num_valid_samples)

    with torch.no_grad():
        for _, (mels, labels) in enumerate(dl_iter):
            _, outputs = run_batch(model, mels, labels)
            log_pred.append(torch.clone(outputs.cpu()).detach())
            log_label.append(torch.clone(labels.cpu()).detach())

    # sigmoid predictions
    log_pred = F.sigmoid(torch.cat(log_pred)).to(cfg.device)

    dataset: PyhaDFDataset = data_loader.dataset # type: ignore
    cmap, smap = map_metric(log_pred, torch.cat(log_label), dataset.class_dist)
    # Log to Weights and Biases
    domain_shift = np.abs(valid_cmap - cmap)
    wandb.log({
        "valid/domain_shift_diff": domain_shift,
        "valid/inferance_map": cmap,
        "valid/inference_smap": smap,
        "epoch_progress": epoch_progress,
    })

    logger.info("Infer cmAP: %f", cmap)
    logger.info("Infer smAP: %f", smap)
    logger.info("Domain Shift Difference: %f", domain_shift)

def save_model(model: TimmModel) -> str:
    """ Saves model in the models directory as a pt file, returns path """
    path = os.path.join("models", f"{cfg.model}-{time_now}.pt")
    if not os.path.exists("models"):
        os.mkdir("models")
    torch.save(model.state_dict(), path)
    return path

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
    train_dataset, val_dataset, infer_dataset, classes = get_datasets(cfg)
    
    
    print(train_dataset)
    def pytorch_dataset_to_hf_dataset(pytorch_dataset):
        def generator():
            for i in range(len(pytorch_dataset)):
                audio, target, file_name = pytorch_dataset[i]
                audio = audio.numpy().astype(np.float32)
                #print(f"Shape of audio: {audio.shape}")
                #print(f"Type of image_list: {type(image)}")
                #print(f"Length of image_list: {len(image)}")
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
    if infer_dataset is not None: hf_test_ds = pytorch_dataset_to_hf_dataset(infer_dataset)

    dataset = DatasetDict({
        #'train': hf_train_ds,
        'validation': hf_valid_ds,
        #'test': None
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
    metric_precision_multi = load("precision", "multilabel", trust_remote_code=True)
    metric_recall = load("recall", trust_remote_code=True)
    metric_recall_multi = load("recall", "multilabel", trust_remote_code=True)
    metric_f1 = load("f1", trust_remote_code=True)
    metric_f1_multi = load("f1", "multilabel", trust_remote_code=True)
    metric_roc_auc = load("roc_auc", "multiclass", trust_remote_code=True)
    metric_roc_auc_multi = load("roc_auc", "multilabel", trust_remote_code=True)
    metric_accuracy = load("accuracy", trust_remote_code=True)
    metric_accuracy_multi = load("accuracy", "multilabel", trust_remote_code=True)

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
        roc_auc = metric_roc_auc.compute(prediction_scores=prob, references=labels, average="macro", multi_class="ovr")

        # onehot = np.zeros((labels.size, labels.max()+1), dtype=int)
        # onehot[:,labels] = 1
        # predictions_onehot = np.zeros((predictions.size, predictions.max()+1), dtype=int)
        # predictions_onehot[:,predictions] = 1

        # precision_multi = metric_precision_multi.compute(predictions=predictions_onehot, references=onehot, average='macro')
        # recall_multi = metric_recall_multi.compute(predictions=predictions_onehot, references=labels, average='macro')
        # f1_multi = metric_f1_multi.compute(predictions=predictions_onehot, references=onehot, average='macro')
        # accuracy_multi = metric_accuracy_multi.compute(predictions=predictions_onehot, references=onehot)
        # roc_auc_multi = metric_roc_auc_multi.compute(prediction_scores=prob, references=onehot, average="macro", multi_class="ovr")

        all_logits.append(logits)
        all_labels.append(labels)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            # 'precision_multi': precision_multi,
            # 'recall_multi': recall_multi,
            # 'f1_multi': f1_multi,
            # 'accuracy_multi': accuracy_multi,
            # 'roc_auc_multi': roc_auc_multi,
        }


    trainer = Trainer(
        model,
        args,
        #train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )
    
    trainer.evaluate()

    import pickle
    pickle_file = f'{model_checkpoint}/logits_labels.pkl'

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
