""" Stores default argument information for the argparser
    Methods:
        create_parser: returns an ArgumentParser with the default arguments
"""
import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-nf', '--num_fold', default=5, type=int)
    parser.add_argument('-nc', '--num_classes', default=264, type=int)
    parser.add_argument('-tbs', '--train_batch_size', default=32, type=int)
    parser.add_argument('-vbs', '--valid_batch_size', default=32, type=int)
    parser.add_argument('-sr', '--sample_rate', default=32_000, type=int)
    parser.add_argument('-hl', '--hop_length', default=512, type=int)
    parser.add_argument('-mt', '--max_time', default=5, type=int)
    parser.add_argument('-nm', '--n_mels', default=224, type=int)
    parser.add_argument('-nfft', '--n_fft', default=1024, type=int)
    parser.add_argument('-s', '--seed', default=0, type=int)
    parser.add_argument('-j', '--jobs', default=4, type=int)
    parser.add_argument('-l', '--logging', default='True', type=str)
    parser.add_argument('-lf', '--logging_freq', default=20, type=int)
    parser.add_argument('-vf', '--valid_freq', default=1000, type=int)
    parser.add_argument('-mch', '--model_checkpoint', default=None, type=str)
    parser.add_argument('-md', '--map_debug', action='store_true')
    parser.add_argument('-p', '--p', default=0, type=float, help='p for mixup')
    parser.add_argument('-i', '--imb', action='store_true', help='imbalance sampler')
    parser.add_argument('-pw', "--pos_weight", type=float, default=1, help='pos weight')
    parser.add_argument('-lr', "--lr", type=float, default=1e-3, help='learning rate')
    parser.add_argument('-mp', "--mix_p", type=float, default=0.4, help='mixup p')
    parser.add_argument('-cpa', "--cutmix_alpha", type=float, default=2.5, help='cutmix alpha')
    parser.add_argument('-mpa', "--mixup_alpha", type=float, default=0.6, help='mixup alpha')
    parser.add_argument('-tsp', "--time_shift_p", type=float, default=0, help='time shift p')
    parser.add_argument('-np', "--noise_p", type=float, default=0.35, help='noise p')
    parser.add_argument('-nsd', "--noise_std", type=float, default=0.005, help='noise std')
    parser.add_argument('-fmp', "--freq_mask_p", type=float, default=0.5, help='freq mask p')
    parser.add_argument('-fmpa', "--freq_mask_param", type=int, default=10, help='freq mask param')
    parser.add_argument("-tmp", "--time_mask_p", type=float, default=0.5, help='time mask p')
    parser.add_argument("-tmpa", "--time_mask_param", type=int, default=25, help='time mask param')
    parser.add_argument('-sm', '--smoothing', type=float, default=0.05, help='label smoothing')

    parser.add_argument('-st', '--offset_col', default='OFFSET', type=str)
    parser.add_argument('-et', '--duration_col', default='DURATION', type=str)
    parser.add_argument('-fp', '--file_path_col', default='IN FILE', type=str)
    parser.add_argument('-mi', '--manual_id_col', default='SCIENTIFIC', type=str)
    return parser
