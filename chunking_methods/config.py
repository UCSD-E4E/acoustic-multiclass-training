
""" Stores default argument information for the argparser
    Methods:
        get_config: returns an ArgumentParser with the default arguments
"""
import argparse

def get_config():
    """ Returns a config variable with the command line arguments or defaults
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-cd', '--chunk_duration', default=5, type=int)
    parser.add_argument('-ft', '--filetype', default=".wav", type=str)
    parser.add_argument('-sl', '--sliding_chunks', default=True, type=bool)

    parser.add_argument('-dp', '--data_path', default='~/path/to/data/', type=str)
    parser.add_argument('-mp', '--metadata', default='~/metadata.csv', type=str)
    parser.add_argument('-sp', '--strong_labels', default='~/strong_labels.csv', type=str)
    parser.add_argument('-cp', '--chunk_path', default='~/chunks.csv', type=str)

    return parser.parse_args()
