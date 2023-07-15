
""" Stores default argument information for the argparser
    Methods:
        get_config: returns an ArgumentParser with the default arguments
"""
import argparse

def get_config():
    """ Returns a config variable with the command line arguments or defaults
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--chunk_length_s', default=5, type=int, help='duration')
    parser.add_argument('-f', '--filetype', default='.wav', type=str)
    parser.add_argument('-w', '--sliding_window', action='store_true')

    parser.add_argument('-a', '--audio_path', default='~/path/to/data/', type=str)
    parser.add_argument('-m', '--metadata', default='~/metadata.csv', type=str)
    parser.add_argument('-s', '--strong_labels', default='~/strong_labels.csv', type=str)
    parser.add_argument('-c', '--chunk_labels', default='~/chunks.csv', type=str)

    return parser.parse_args()
