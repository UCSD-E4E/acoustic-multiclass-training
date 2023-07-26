""" Entry point into PyHa Analyzer train function """
import torch
import pyha_analyzer as pa

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    DO_TRAIN = True
    if DO_TRAIN:
        pa.train.main()
    else:
        pa.sweeps.main()
