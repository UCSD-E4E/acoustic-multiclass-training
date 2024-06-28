
# %%
import pandas as pd
from pyha_analyzer.train import run_batch
from pyha_analyzer.dataset import get_datasets, make_dataloaders, PyhaDFDataset
from pyha_analyzer import config
from pyha_analyzer.models.timm_model import TimmModel
import logging

logger = logging.getLogger("acoustic_multiclass_training")

cfg = config.cfg

import pandas as pd
df = pd.read_csv("/home/shperry/acoustic-multiclass-training/peru-2019-pyha-anaylzer-inferance.csv")

# %%


classes = [
      "amabaw1",
      "amapyo1",
      "astgna1",
      "baffal1",
      "barant1",
      "bartin2",
      "batman1",
      "blacar1",
      "blbthr1",
      "blcbec1",
      "blctro1",
      "blfant1",
      "blfcot1",
      "blfjac1",
      "blfnun1",
      "blgdov1",
      "blhpar1",
      "bltant2",
      "blttro1",
      "bobfly1",
      "brratt1",
      "bsbeye1",
      "btfgle1",
      "bubgro2",
      "bubwre1",
      "bucmot4",
      "buffal1",
      "butsal1",
      "butwoo1",
      "chwfog1",
      "cinmou1",
      "cintin1",
      "citwoo1",
      "coffal1",
      "coltro1",
      "compot1",
      "cowpar1",
      "crfgle1",
      "ducatt1",
      "ducfly",
      "ducgre1",
      "duhpar",
      "dutant2",
      "elewoo1",
      "eulfly1",
      "fasant1",
      "fepowl",
      "forela1",
      "garkin1",
      "gilbar1",
      "gnbtro1",
      "gocspa1",
      "goeant1",
      "gogwoo1",
      "gramou1",
      "grasal3",
      "grcfly1",
      "greant1",
      "greibi1",
      "gretin1",
      "grfdov1",
      "gryant1",
      "gryant2",
      "gycfly1",
      "gycwor1",
      "hauthr1",
      "horscr1",
      "letbar1",
      "littin1",
      "litwoo2",
      "lobwoo1",
      "lowant1",
      "meapar",
      "muswre2",
      "olioro1",
      "oliwoo1",
      "partan1",
      "pavpig2",
      "pirfly1",
      "plbwoo1",
      "pltant1",
      "pluant1",
      "plupig2",
      "plwant1",
      "puteup1",
      "putfru1",
      "pygant1",
      "rcatan1",
      "rebmac2",
      "renwoo1",
      "rinant2",
      "rinkin1",
      "rinwoo1",
      "royfly1",
      "ruboro1",
      "rucant2",
      "rudpig",
      "rufant3",
      "ruftof1",
      "ruqdov",
      "scapig2",
      "scbwoo5",
      "scrpih1",
      "sobcac1",
      "specha3",
      "spigua1",
      "spwant2",
      "squcuc1",
      "stbwoo2",
      "strcuc1",
      "strwoo2",
      "strxen1",
      "stwqua1",
      "tabsco1",
      "thlwre1",
      "undtin1",
      "viotro3",
      "wespuf1",
      "whbtot1",
      "whcspa1",
      "whfant2",
      "whltyr1",
      "whnrob1",
      "whrsir1",
      "whttou1",
      "whtwoo2",
      "whwbec1",
      "wibpip1",
      "yectyr1",
      "yemfly1",
      "yercac1",
      "yetwoo2"
    ]


df["MANUAL ID"] = "yetwoo2"

# %%
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
df = df[["OFFSET", "DURATION", "SourceFile", "MANUAL ID"]]
infer_dataset = PyhaDFDataset(df, train=False, species=classes, cfg=cfg)
infer_dataloader = DataLoader(
                infer_dataset,
                cfg.validation_batch_size,
                shuffle=False,
                num_workers=cfg.jobs,
            )

# %%
print(len(infer_dataset))

# %%

import torch
model_for_run = TimmModel(num_classes=len(classes), 
                            model_name=cfg.model).to(cfg.device)

if cfg.model_checkpoint != "":
    model_for_run.load_state_dict(torch.load(cfg.model_checkpoint))

# %%
from tqdm import tqdm
import torch.nn.functional as F
model = model_for_run
model.eval()

log_pred, log_label = [], []

num_valid_samples = int(len(infer_dataloader))

# tqdm is a progress bar
dl_iter = tqdm(infer_dataloader, position=5, total=num_valid_samples)

def run_batch(model: TimmModel,
                mels: torch.Tensor,
                labels: torch.Tensor,
                ):
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
    outputs = model(mels)
    return [0], outputs

print("ran model")
with torch.no_grad():
    for _, (mels, labels) in enumerate(dl_iter):
        try:
            _, outputs = run_batch(model, mels, labels)
            log_pred.append(torch.clone(outputs.cpu()).detach())
            log_label.append(torch.clone(labels.cpu()).detach())
        except Exception as e:
            print(e)

print("save results")
# sigmoid predictions
log_pred = F.sigmoid(torch.cat(log_pred)).cpu()
print(log_pred.shape)
pd.DataFrame(log_pred, columns=classes).to_csv("result.csv")

#cmap, smap = map_metric(log_pred, torch.cat(log_label), dataset.class_dist)

# %%
cfg.data_path2

# %%


# %%



