""" Contains the early stopper class
    TODO

"""

import config

# timm is a library of premade models


cfg = config.cfg

class EarlyStopper():
    """Stop when the model is no longer improving
    """
    def __init__(self, patience: int=3, min_delta: float=0):
        self.patience = patience # epochs to wait before stopping
        self.min_delta = min_delta # min change that counts as improvement
        self.counter = 0
        self.max_valid_map = 0

    def early_stop(self, valid_map: float):
        """ Returns True if the model should early stop
        """
        # reset counter if it improved by more than min_delta
        if valid_map > self.max_valid_map + self.min_delta:
            self.max_valid_map = valid_map
            self.counter = 0
        # increase counter if it has not improved
        elif valid_map < (self.max_valid_map - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
