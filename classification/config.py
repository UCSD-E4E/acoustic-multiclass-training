
""" Stores default argument information for the argparser
    Methods:
        get_config: returns an ArgumentParser with the default arguments
"""
import sys
import argparse
import git
import yaml
import os
import shutil

class Config():
    def __new__(self):
        if not hasattr(self, 'instance'):

            self.test_ = "testing!"
            self.instance = super(Config, self).__new__(self)
        else:
            print("CONFIG ALREADY EXISTS")
            return self.instance

        #Set defaults
        with open('config.yml', 'r') as file:
            self.config_dict = yaml.safe_load(file)

        default_keys = set()
        print(self.config_dict)
        for (key, value) in self.config_dict.items():
            setattr(self, key, value)
            default_keys.add(key)

        #Set User Custom Values
        if (os.path.exists('config_personal.yml')):
            with open('config_personal.yml', 'r') as file:
                self.config_personal_dict = yaml.safe_load(file)

            for (key, value) in self.config_personal_dict.items():
                setattr(self, key, value)
                

            self.config_dict.update(self.config_personal_dict)
        else:
            shutil.copy("config.yml", "config_personal.yml")


        # Update personal dict

        attrs_to_append = []

        for key in default_keys:
            if key in self.config_personal_dict: continue
            
            appending_attrs = {
                key: self.config_dict[key]
            }

            #https://media.tenor.com/dxPl_UoR8J0AAAAC/fire-writing.gif
            attrs_to_append.append(appending_attrs) 

        if (len(attrs_to_append) == 0):
            return self.instance
        
        with open('config_personal.yml', 'a') as file:
            file.write("\n\n#NEW DEFAULTS\n")

            for new_attrs in attrs_to_append:
                yaml.dump(new_attrs, file)

        return self.instance


# Machine learning has a lot of arugments
# pylint: disable=too-many-statements
def get_config():
    """ Returns a config variable with the command line arguments or defaults
    """
    parser = argparse.ArgumentParser()

    # Dataset settings
    parser.add_argument('-df', '--dataframe', default='CHANGEME.csv', type=str)
    parser.add_argument('-dp', '--data_path', default='./all_10_species', type=str)


    parser.add_argument('-st', '--offset_col', default='OFFSET', type=str)
    parser.add_argument('-et', '--duration_col', default='DURATION', type=str)
    parser.add_argument('-fn', '--file_name_col', default='FILE NAME', type=str)
    parser.add_argument('-mi', '--manual_id_col', default='SCIENTIFIC', type=str)

    # Env settings
    parser.add_argument('-tbs', '--train_batch_size', default=4, type=int)
    parser.add_argument('-vbs', '--valid_batch_size', default=4, type=int)
    parser.add_argument('-wbs', '--wandb_session', default="acoustic-species-reu2023", 
                        type=str, help="wandb project name")

    # Functional settings
    parser.add_argument('-j', '--jobs', default=2, type=int)
    parser.add_argument('-s', '--seed', default=0, type=int)
    parser.add_argument('-l', '--logging', default='True', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-lf', '--logging_freq', default=20, type=int)
    parser.add_argument('-vf', '--valid_freq', default=2000, type=int)

    # Model Training settings
    # pylint: disable=pointless-string-statement
    """
        Suggested model types:
            eca_nfnet_l0 (90 MB)
            tf_efficientnet_b4 (70 MB)
            convnext_nano (60 MB)
            convnext_tiny (110 MB)
            resnetv2_50 (100 MB)
            resnetv2_101 (170 MB)
            seresnext50_32x4d (100 MB)
            seresnext101_32x4d (200 MB)
            rexnet_200 (70 MB)
            mobilenetv3_large_100_miil_in21k (70 MB)
    """
    parser.add_argument('-m', '--model', default='eca_nfnet_l0', type=str)
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-nf', '--num_fold', default=5, type=int)
    parser.add_argument('-tts', '--train_test_split', default=0.8, type=float)
    parser.add_argument('-sr', '--sample_rate', default=32_000, type=int)
    parser.add_argument('-hl', '--hop_length', default=512, type=int)
    parser.add_argument('-mt', '--max_time', default=5, type=int)
    parser.add_argument('-nm', '--n_mels', default=194, type=int)
    parser.add_argument('-nfft', '--n_fft', default=1400, type=int)
    parser.add_argument('-mch', '--model_checkpoint', default=None, type=str)
    parser.add_argument('-md', '--map_debug', action='store_true')
    parser.add_argument('-mxp', '--mixed_precision', action='store_true')

    # Early stopping
    parser.add_argument('-es', '--early_stopping', action='store_true')
    parser.add_argument('-pa', '--patience', type=int, default=3, help="epochs to wait before stopping")
    parser.add_argument('-del', '--min_delta', type=float, default=0.01, help='min improvement btwn epochs')
    # Transforms settings
    parser.add_argument('-p', '--p', default=0, type=float, help='probability for mixup')
    parser.add_argument('-i', '--imb', action='store_true', help='imbalance sampler')
    parser.add_argument('-pw', "--pos_weight", type=float, default=1, help='pos weight')
    parser.add_argument('-lr', "--lr", type=float, default=1e-3, help='learning rate')
    parser.add_argument('-mp', "--mix_p", type=float, default=0.4, help='mixup probability')
    parser.add_argument('-tsp', "--time_shift_p", type=float, default=0, help='time shift probability')
    parser.add_argument('-np', "--noise_p", type=float, default=0.35, help='noise probability')
    parser.add_argument('-nsd', "--noise_std", type=float, default=0.005, help='noise std')
    parser.add_argument('-fmp', "--freq_mask_p", type=float, default=0.5, help='freq mask probability')
    parser.add_argument('-fmpa', "--freq_mask_param", type=int, default=10, help='freq mask param')
    parser.add_argument("-tmp", "--time_mask_p", type=float, default=0.5, help='time mask probability')
    parser.add_argument("-tmpa", "--time_mask_param", type=int, default=25, help='time mask param')
    parser.add_argument('-sm', '--smoothing', type=float, default=0.05, help='label smoothing')

    CONFIG = parser.parse_args()
    
    # Convert string arguments to boolean
    CONFIG.logging = CONFIG.logging == 'True'
    
    #Add git hash to config so wand logging can track vrs used for reproduciblity
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        setattr(CONFIG, "git_hash", sha)
        print(sha)

    #I want to catch this spefific error, and it doesn't extend from base exception
    #¯\(ツ)/¯
    # pylint: disable=no-member
    except git.exc.InvalidGitRepositoryError:
        print("InvalidGitRepositoryError found, this means we cannot save git hash :(")
        print("You are likely calling a python file outside of this repo") 
        print("if from command line, cd into acoustic-mutliclass-training")
        print("then you can run the script again")
        sys.exit(1)

    return CONFIG


def testing():
    """
    Test functionality of generating and caching configs
    """
    config = Config()
    config2 = Config()
    print(config == config2)
    print(config.test_)
    print(config2.test_)

if __name__ == "__main__":
    testing()