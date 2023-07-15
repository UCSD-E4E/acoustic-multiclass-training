
""" Stores default argument information for the argparser
    Methods:
        get_config: returns an ArgumentParser with the default arguments
"""
import sys
import os
import shutil

import git
import yaml


class Config():
    """
    Saves config from config files
    Allows for customising runs
    """
    def __init__(self):
        """Constructor that runs after creating the singleton class
        """
        self.get_git_hash()
    
    def __new__(cls):
        """
        Constructor for Config

        Takes data from config files and generates a singleton class
        that can store all system vars

        returns a refrence to the singleton class
        """
        
        # Set up singleton class design template
        print("hello new")
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)
        else:
            return cls.instance

        #Set defaults config
        with open('config.yml', 'r', encoding='utf-8') as file:
            cls.config_dict = yaml.safe_load(file)

        default_keys = set()
        print(cls.config_dict)
        for (key, value) in cls.config_dict.items():
            setattr(cls, key, value)
            default_keys.add(key)

        #Set User Custom Values
        if os.path.exists('config_personal.yml'):
            with open('config_personal.yml', 'r', encoding='utf-8') as file:
                cls.config_personal_dict = yaml.safe_load(file)

            for (key, value) in cls.config_personal_dict.items():
                setattr(cls, key, value)
                
            cls.config_dict.update(cls.config_personal_dict)
        else:
            shutil.copy("config.yml", "config_personal.yml")


        # Update personal dict with new keys
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
            print("There are new updates in default config")
            print("please manually update these keys from the new config")
            print(attrs_to_append)

        return cls.instance

    def generate_config_file(self, filename="test.yml"):
        """
        Sends all configs saved to class to new file
        overwrites file
        """
        with open(filename, 'w', encoding='utf-8') as file:
            yaml.dump(self.config_dict, file)
    
    def __getattribute__(self,attr):
        return self.config_dict[attr]

    def get_git_hash(self):
        """
        Gets the hash of the current git commit
        This requires recording the hash before model training
        """

        #Add git hash to config so wand logging can track vrs used for reproduciblity
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            setattr(cls.config_dict, "git_hash", sha)
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
        return sha

def get_config():
    """ Returns a config variable with the command line arguments or defaults
    Decrepated, returns Config to prevent largescale code breaks
    """
    return Config().config_dict

def testing():
    """
    Test functionality of generating and caching configs
    """

    config = Config()

    # I want to test singleton class creation
    # So I want to change one instance var and see that var in a new class
    # in practice, I never want to change a config setting outside of config
    # pylint: disable=attribute-defined-outside-init
    config.change = " hah"
    config2 = Config()
    print(config == config2, "Expect True")
    print(config.change, "Expect hah")
    print(config2.change, "Expect hah")

    Config().generate_config_file()


#Expose variable to page scope
cfg = Config()

if __name__ == "__main__":
    testing()
