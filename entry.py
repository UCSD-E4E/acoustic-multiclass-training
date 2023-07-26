import pyha_analyzer as pa
import torch


torch.multiprocessing.set_sharing_strategy('file_system')
torch.multiprocessing.set_start_method('spawn')
pa.sweeps.main()