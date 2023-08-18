"""
    Fine tuning on debiased pseudo labeling
    
    Based on the paper "Debiased Learning from Naturally Imbalanced Pseudo-Labels"
    Link: https://arxiv.org/abs/2201.01490
    
    The goal is to fine tune on pseudo labels while maintaining performance on training data
"""
# pylint: disable=duplicate-code
import datetime
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pyha_analyzer import config, dataset, utils
from pyha_analyzer import pseudolabel
from pyha_analyzer import train
from pyha_analyzer.models.timm_model import TimmModel

cfg = config.cfg
logger = logging.getLogger("acoustic_multiclass_training")


class DebiasTrainProcess(train.TrainProcess):
    """ Train process with debias_pl_epoch """
    def __init__(
            self, 
            model: TimmModel, 
            augs_ds: dataset.PyhaDFDataset,
            train_dl: DataLoader, 
            valid_dl: DataLoader, 
            infer_dl: Optional[DataLoader],
            unlabel_dl: DataLoader
        ):
        """ Init function """
        super().__init__(model,train_dl,valid_dl,infer_dl)
        self.unlabel_dl = unlabel_dl
        self.augs_ds = augs_ds
        self.qhat = torch.tensor(
            [1/model.num_classes for i in range(model.num_classes)]
        ).to(cfg.device)
    
    def update_qhat(self,probs):
        """ Updates class penalties """
        mean_probs = probs.detach().mean(dim=0)
        self.qhat = 0.9*self.qhat + 0.1*mean_probs

    # pylint: disable-next=too-many-statements, too-many-locals
    def debias_pl_epoch(self):
        """Train model for one epoch using debiased pseudolabeling"""
        self.model.train()
        log_n = log_loss = log_map = 0
        start_epoch = self.epoch
        start_time = datetime.datetime.now()
        train_iter = iter(self.train_dl)

        for i, (audio, labels) in enumerate(self.unlabel_dl):
            mels = []
            orig_mels = []
            for aud in audio:
                orig_mels.append(self.augs_ds.to_image(aud))
                aud = self.augs_ds.audio_augmentations(aud)
                mel = self.augs_ds.to_image(aud)
                mels.append(self.augs_ds.image_augmentations(mel))
            mels = torch.stack(mels)
            orig_mels = torch.stack(orig_mels)

            mels_l, labels_l, _ = next(train_iter)

            self.epoch = start_epoch + i/len(self.unlabel_dl)
            self.optimizer.zero_grad()
            with torch.no_grad():
                _, outputs_uw = self.model.run_batch(orig_mels, labels)
            _, outputs_us = self.model.run_batch(mels, labels)
            loss_l, outputs_l = self.model.run_batch(mels_l, labels_l)
            p_u = F.sigmoid(outputs_uw - 1.0 * torch.log(self.qhat))
            max_probs, pseudo_label = torch.max(p_u, dim=-1)
            mask = max_probs.ge(cfg.pseudo_threshold).float()

            outputs_us += 1.0 * torch.log(self.qhat)
            loss_u = (F.cross_entropy(outputs_us, pseudo_label) * mask).mean()
            loss = loss_l + loss_u * 10

            if cfg.mixed_precision and cfg.device != "cpu":
                self.scaler.scale(loss).backward()  # type: ignore
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # CPU step back
                if cfg.mixed_precision:
                    logger.warning("cuda required, mixed precision not applied")
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
    
            log_pred = F.sigmoid(outputs_l)
            log_map += train.map_metric(log_pred, labels_l, self.model.num_classes)
            log_loss += loss.item()
            log_n += 1
    
            #Log and reset metrics
            if (i != 0 and i % (cfg.logging_freq) == 0) or i == len(self.unlabel_dl) - 1:
                train.log_metrics(self.epoch, i, start_time, log_n, log_loss, log_map)
                start_time = datetime.datetime.now()
                log_n = log_loss = log_map = 0
    
            # Free memory so gpu is freed before validation run
            if (i != 0 and i % (cfg.valid_freq) == 0):
                del orig_mels
                del mels
                del mels_l

                valid_start_time = datetime.datetime.now()
                self.valid()
                self.model.train()
                # Ignore the time it takes to validate in annotations/sec
                start_time += datetime.datetime.now() - valid_start_time

class OrigAudioDataset(dataset.PyhaDFDataset):
    """ Dataset that returns original audio rather than augmented spectrogram """
    def __init__(self,
                 df: pd.DataFrame,
                 species: List[str],
                 onehot:bool = False,
                 ) -> None:
        super().__init__(df,True,species,onehot)
    
    def __getitem__(self,index):
        assert isinstance(index, int)
        audio, target = utils.get_annotation(
                df = self.samples, index = index, class_to_idx = self.class_to_idx)

        if  self.onehot:
            target = self.samples.loc[index, self.classes].values.astype(np.int32)
            target = torch.Tensor(target)

        return audio, target


def main():
    """ Main function """
    # Init
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    print(f"Device is: {cfg.device}, Preprocessing Device is {cfg.prepros_device}")
    utils.set_seed(cfg.seed)
    utils.logging_setup()
    utils.wandb_init(False)

    print("Creating model...")
    model = TimmModel(num_classes=len(cfg.class_list), model_name=cfg.model).to(cfg.device)
    model.create_loss_fn(None)
    if not model.try_load_checkpoint():
        raise RuntimeError("Model checkpoint required")

    logger.info("Generating raw dataframe...")
    if cfg.config_dict["class_list"] is None:
        raise ValueError("Pseudo-labeling requires class list")
    raw_df = pseudolabel.make_raw_df()

    logger.info("Loading dataset...")
    train_ds, valid_ds, infer_ds = dataset.get_datasets()
    train_dl, valid_dl, infer_dl = dataset.get_dataloader(train_ds, valid_ds, infer_ds)
    model.create_loss_fn(train_ds)
    unlabel_ds = OrigAudioDataset(raw_df, species=cfg.class_list)
    unlabel_dl = dataset.make_dataloader(unlabel_ds, cfg.train_batch_size, False, True)
    assert len(train_ds) > len(unlabel_ds), "More training samples than unlabeled samples"
    
    train_ds.samples = train_ds.samples.sample(n=len(unlabel_ds))
    assert len(train_ds) == len(unlabel_ds)
    train_dl = DataLoader(
        train_ds,
        cfg.train_batch_size,
        sampler=range(0,len(unlabel_ds)),
        num_workers=cfg.jobs,
        worker_init_fn=dataset.set_torch_file_sharing,
        persistent_workers=True
    )

    train_process = DebiasTrainProcess(model, unlabel_ds, train_dl, valid_dl, infer_dl, unlabel_dl)
    train_process.valid()
    for _ in range(cfg.epochs):
        train_process.debias_pl_epoch()
        train_process.valid()

if __name__ == "__main__":
    main()
