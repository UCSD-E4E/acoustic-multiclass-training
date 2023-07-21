"""
Sweeps file: 
    Run it to start a new sweep or start another agent for an existing
sweep. If the former, set the sweep_id option in config
"""
import config
import wandb
from train import main as train_main

cfg = config.cfg


sweep_config = {
    'method'            : 'bayes',
    'name'              : 'sweep',
    'early_terminate'   : {'type': 'hyperband', 'min_iter': 3},
    'metric'            : {'goal': 'maximize', 'name' : 'val/map'},
    'parameters'        : {
        # General
        'lr' : {
            'max': 0.01,
            'min': 5e-4,
            'distribution': 'log_uniform_values'
        },
        'model' : {'values':["eca_nfnet_l0",
                 "tf_efficientnet_b4",
                 "resnetv2_50",
                 "resnetv2_101",
                 "seresnext50_32x4d",
                 "seresnext101_32x4d",
                 "rexnet_200"]},
        #Data Aug Probabilities
        'mixup_p'           : {'max': .6, 'min': 0.},
        'time_shift_p'      : {'max': .6, 'min': 0.},
        'noise_p'           : {'max': .6, 'min': 0.},
        'freq_mask_p'       : {'max': .6, 'min': 0.},
        'time_mask_p'       : {'max': .6, 'min': 0.},
        'rand_eq_p'         : {'max': .6, 'min': 0.},
        'bg_noise_p'        : {'max': .6, 'min': 0.},
        'lowpass_p'         : {'max': .6, 'min': 0.},
        #Data Aug Params
        'noise_type': {'values':
            ['white', 'pink', 'brown', 'blue', 'violet']
        },
        'noise_alpha': {'max': .25, 'min': 0.},
        'freq_mask_param': {'max': 50, 'min': 5},
        'time_mask_param': {'max': 50, 'min': 5},
        'mixup_alpha_range': {'values':
            [[0.1,0.4],
             [0.0, 0.6],
             [0.2, 0.3],
             [0.0, 0.3],
             [0.3, 0.6]]
        },
        'rand_eq_f_range': {'values':
            [[100, 8000],
             [20, 8000],
             [100, 18000],
             [400, 2000]]
        },
        'rand_eq_g_range': {'values':
            [[-8, 8],
             [-4, 4],
             [-2, 8],
             [-8, 2]]
        },
        'rand_eq_q_range': {'values':
            [[1, 9],
             [0.3, 9],
             [0.3, 3]]
        },
        'rand_eq_iters': {'max': 5, 'min': 0},
        'lowpass_cutoff': {
            'max': 18000,
            'min': 4000,
            'distribution': 'log_uniform_values'
        },
        'lowpass_q_val': {'max': 3., 'min': 0.1},
        'bg_noise_alpha_range': {'values':
            [[0.0, 0.2],
             [0.0, 0.4],
             [0.2, 0.3],
             [0.0, 0.3],
             [0.0, 0.1]]
        }
    }
}

def main():
    """
    Main function
    """
    sweep_id = cfg.sweep_id
    wandb.login()
    if not sweep_id:
        print("Starting a new sweep")
        sweep_id = wandb.sweep(
            sweep_config,
            entity=cfg.wandb_entity,
            project=cfg.wandb_project + "-sweep")
    wandb.agent(sweep_id, function = train_main, count=1)

if __name__ == "__main__":
    main()
