import os
import shutil
import pandas as pd

from datasets import Dataset, load_dataset, Audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import numpy as np
from torchmetrics.classification import MultilabelAveragePrecision

base_path = '/share/acoustic_species_id/BirdCLEF2023_split_chunks_new/'
train_path = '/share/acoustic_species_id/BirdCLEF2023_split_chunks_new/training'
valid_path = '/share/acoustic_species_id/BirdCLEF2023_split_chunks_new/validation'
filetype = ".wav"

def gen_dicts():
    def create_df(subfolders):
        df = pd.DataFrame()
        for s in subfolders:
            species = s.split('/')[-1]
            s_list = [f.path for f in os.scandir(s) if f.path.endswith(filetype)]
            s_dict = [{"label" : species, "filename" : f} for f in s_list]
            temp = pd.DataFrame(s_dict)
            df = pd.concat([df, temp])
        
        return df.sort_values(by=["label", "filename"], ignore_index=True)

    train = create_df([f.path for f in os.scandir(train_path) if f.is_dir()])
    valid = create_df([f.path for f in os.scandir(valid_path) if f.is_dir()])
    return train, valid

    
def gen_datasets(train, valid):
    train_dataset = Dataset.from_pandas(train)#.cast_column("filename", Audio(sampling_rate=16_000))
    valid_dataset = Dataset.from_pandas(valid)#.cast_column("filename", Audio(sampling_rate=16_000))
    print(train_dataset[0])

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True
    )
    return inputs

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    metric = MultilabelAveragePrecision(num_labels=CONFIG.num_classes, average="weighted")
    valid_map = metric(predictions, eval_pred.label_ids)
    print(f"cmAP: {valid_map}")
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

if __name__ == '__main__':
    dataset = load_dataset("audiofolder", data_dir=base_path)
    print(dataset["train"][0])

    # label processing
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # preprocessing
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    max_duration = 5.0
    
    encoded_dataset = dataset.map(preprocess_function, remove_columns="audio", batched=True)

    # training
    num_labels = len(id2label)
    
    model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base", 
            num_labels=num_labels, 
            label2id=label2id, 
            id2label=id2label)
    
    training_args = TrainingArguments(
        output_dir="wav2vec_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()