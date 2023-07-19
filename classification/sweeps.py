import wandb
from train import sweep_main
import config
import multiprocessing as mp
#mp.set_start_method('spawn')
cfg = config.cfg


sweep_config = {
    'method'    : 'bayes',
    'name'      : 'sweep',
    'metric'    :  {'goal': 'maximize', 'name' : 'val/map'},
    'parameters': {
         'batch_size'   : {'values': [16, 32, 64, 128]},
          'epochs'      : {'values': [5, 10, 15]},
          'lr'          : {'max': 0.1, 'min': 0.00},
          'mixup_p'     : {'max': 1., 'min': 0.},
          'time_shift_p': {'max': 1., 'min': 0.},
          'noise_p'     : {'max': 1., 'min': 0.},
          'freq_mask_p' : {'max': 1., 'min': 0.},
          'time_mask_p' : {'max': 1., 'min': 0.},
          'rand_eq_p'   : {'max': 1., 'min': 0.}
          }
      }

if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(
            sweep_config,
            entity=cfg.wandb_entity,
            project=cfg.wandb_project)
    #controller = wandb.controller(sweep_id,
    #        entity=cfg.wandb_entity,
    #        project=cfg.wandb_project)
    wandb.agent(sweep_id, function = sweep_main, count=1)
