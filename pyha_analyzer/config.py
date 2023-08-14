""" 
Stores default argument information for the argparser
"""
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

# "Repo" is not exported from module "git" Import from "git.repo" instead
# https://gitpython.readthedocs.io/en/stable/tutorial.html?highlight=repo#meet-the-repo-type
import git
import importlib_resources as pkg_resources
import yaml
from git import Repo  # pyright: ignore [reportPrivateImportUsage]
from torch.cuda import is_available

logger = logging.getLogger("acoustic_multiclass_training")

class Config():
    """
    Saves config from config files
    Allows for customising runs
    """
    def __init__(self):
        """Constructor that runs after creating the singleton class
        Post processing after reading config file
        """
        self.required_checks("dataframe_csv")
        self.required_checks("data_path")
        self.get_git_hash()
        self.cli_values()
        self.get_device()

    def __new__(cls):
        """
        Constructor for Config

        Takes data from config files and generates a singleton class
        that can store all system vars

        returns a refrence to the singleton class

        Intended to read config files
        """
        
        # Set up singleton class design template
        if not hasattr(cls, 'instance'):
            cls.instance = None
            cls.instance = super(Config, cls).__new__(cls)
        else:
            return cls.instance
        cls.package_root = pkg_resources.files("pyha_analyzer")
        #Set defaults config
        # Pyright is wrong, this function does exist
        with cls.package_root.joinpath("default_config.yml") \
                .open('r', encoding='utf-8') as file: #type: ignore
            cls.config_dict = yaml.safe_load(file)

        default_keys = set()
        for (key, value) in cls.config_dict.items():
            setattr(cls, key, value)
            default_keys.add(key)

        #Set User Custom Values
        config_path = cls.get_conf_path()
        # Pyright is wrong, this function does exist
        with config_path.open('r', encoding='utf-8') as file: #type: ignore
            cls.config_personal_dict = yaml.safe_load(file)

        for (key, value) in cls.config_personal_dict.items():
            setattr(cls, key, value)
            
        cls.config_dict.update(cls.config_personal_dict)
        
        attrs_to_append = []

        for key in default_keys:
            if key in cls.config_personal_dict: 
                continue
            
            value = cls.config_dict[key]
            appending_attrs = {
                key: value
            }

            #https://media.tenor.com/dxPl_UoR8J0AAAAC/fire-writing.gif
            attrs_to_append.append(appending_attrs) 


        if len(attrs_to_append) != 0:
            logger.warning("There are new updates in default config")
            logger.warning("please manually update these keys from the new config")
            logger.warning("%s", str(attrs_to_append))

        # Update personal dict with new keys
        return cls.instance
    
    @classmethod
    def get_conf_path(cls) -> Path:
        """ Get pathlib path to config file """
        # Check for local configs first
        local_paths = ["pyha_analyzer/config.yml","config.yml","../config.yml","../../config.yml"]
        for path in local_paths:
            if os.path.exists(path):
                return Path(path)

        # Check for package configs second
        config_path = cls.package_root.joinpath("config.yml")
        if config_path is not None:
            if config_path.is_file():
                return Path(str(config_path))
        config_path = cls.package_root.joinpath("../config.yml")
        if config_path is not None and config_path.is_file():
            return Path(str(config_path))

        # Copy default_config to config.yml and exit
        # Pyright is wrong, this function does exist
        with cls.package_root.joinpath("config.yml").open("w") as config_file: # type: ignore
            with cls.package_root.joinpath("default_config.yml") \
                    .open("r",encoding="utf-8") as default_config_file: #type: ignore
                shutil.copyfileobj(default_config_file, config_file)
        logger.error("No config file found, creating one for you")
        logger.error("Config file to edit: %s", str(cls.package_root / "config.yml"))
        logger.error("Add your data path and csv file to the config.yml")
        sys.exit(1)

    def cli_values(self):
        """ 
        Saves all command line arguments to config class
        Primarily intended for quick flags such as turning a property on or off
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-l', '--logging', action='store_false')
        parser.add_argument('-d', '--debug', action='store_true')
        
        arg_cfgs = parser.parse_args()
        
        # Add all command line args to config
        # Overwrite because user is defining them most recently
        # Unless they are default value
        arg_cfgs = vars(arg_cfgs)
        for key in arg_cfgs:
            if self.config_dict[key] == parser.get_default(key):
                setattr(self, key, arg_cfgs[key])
        
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        if self.debug:
            console_handler.setLevel(logging.DEBUG)
        else: 
            console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    def required_checks(self, parameter):
        """
        Parameter string

        Raises ValueError if parameter isn't decalred in config yaml or isn't defined
        """
        if parameter not in self.config_dict:
            raise ValueError(f'The required parameter "{parameter}" is not present in yaml')
        if  self.config_dict[parameter] is None:
            raise ValueError(f'The required parameter "{parameter}" is not defined in yaml')

    def __getattr__(self,attr):
        """
        Gets a config value in a dict-like way

        attr string for varible name as defined in a config.yml file
        """
        return self.config_dict[attr]

    def get_git_hash(self):
        """
        Gets the hash of the current git commit
        This requires recording the hash before model training
        """

        #Add git hash to config so wand logging can track vrs used for reproduciblity
        sha = "git hash not found"
        try:
            #
            repo = Repo(path=str(Config.package_root),search_parent_directories=True)
            sha = repo.head.object.hexsha
            self.config_dict["git_hash"] = sha

        #I want to catch this specific error, and it doesn't extend from base exception
        #becuase of this, this leads to a lot of linting issues ¯\(ツ)/¯
        # pylint: disable=no-member
        except git.exc.InvalidGitRepositoryError: # pyright: ignore [reportGeneralTypeIssues]
            logger.error("InvalidGitRepositoryError found, this means we cannot save git hash :(")
            logger.error("You are likely calling a python file outside of this repo") 
            logger.error("if from command line, cd into acoustic-mutliclass-training")
            logger.error("then you can run the script again")
        return sha
    
    def get_device(self):
        """ Gets the current device of the system if user defined device config param as 'auto'
        Returns nothing
        Raises value error if device doesn't exist in config file
        """

        self.required_checks("device")
        if self.config_dict["device"] is None:
            raise ValueError('The required parameter "device" is not defined in yaml')
        if self.config_dict["device"] == "auto":
            device = "cuda" if is_available() else "cpu"
            # For whatever reason you have to use cfg.device instead of self.device
            # even though that doesn't make any sense at all
            self.device = device
        self.config_dict["device"] = self.device

def testing():
    """
    Test functionality of generating and caching configs
    """

    config = Config()

    # I want to test singleton class creation
    # So I want to change one instance var and see that var in a new class
    # in practice, I never want to change a config setting outside of config
    # pylint: disable=attribute-defined-outside-init
    
    config2 = Config()
    assert config == config2
    logger.info("%s", str(config.dataframe_csv))
    logger.info("%s", str(config.logging))


#Expose variable to page scope
cfg = Config()


if __name__ == "__main__":
    testing()
