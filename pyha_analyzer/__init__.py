""" Gets project directory for config """
from os import path
from . import sweeps  

pyha_project_directory = path.dirname(path.dirname(__file__))
