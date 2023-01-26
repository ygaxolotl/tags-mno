import os
import yaml
import wandb
from pydantic.utils import deep_update
from pce_pinns.utils.utils import sync_nested_wandb_config

def update_dynamic_cfg_lorenz96(config_default):
    config_default= deep_update(config_default, {
        'data_loader': {
            'chunk_size': int(config_default['de']['tmax']/config_default['de']['dt']), #X.shape[0],
            'batch_size': int(config_default['de']['tmax']/config_default['de']['dt']), # Has to be > 2 
        },
    })
    return config_default


def load_config(args, experiment_name='default', 
    diffeq_name='lorenz96',
    update_dynamic_cfg=None,
    sweep=False,
    ):
    """
    Loads config file
    Args:
        update_dynamic_cfg fn: config_default->config_default: function to update dynamic config parameters
        sweep bool: If true, loads wandb sweep config.
    Returns:
    """
    if args.no_wandb:
        os.environ['WANDB_SILENT'] = 'true'
        os.environ['WANDB_MODE'] = 'dryrun'

    # Load data config
    config_default = yaml.load(open(f'models/{experiment_name}/{diffeq_name}/config.yaml'), 
        Loader=yaml.FullLoader)
    # Load model config
    config_default = deep_update(config_default, yaml.load(
        open(f'models/{experiment_name}/{diffeq_name}/{args.model_type}/config.yaml'), 
        Loader=yaml.FullLoader)) # Model

    # Update config with user args
    config_default = deep_update(config_default, {
        'de': {
            'n_samples': args.n_samples,
            'seed': args.seed,
        },
        'model': {
            'seed': args.seed,
        },
    })
    # Update config parameters that contain computations
    if update_dynamic_cfg is not None:
        config_default = update_dynamic_cfg(config_default)

    # Initialize wandb and hyperparameter sweep
    if sweep:
        pass
        # sweep_config = yaml.load(open(f'models/default/{args.model_type}/sweep.yaml'), Loader=yaml.FullLoader)
    else:
        wandb.init(config=config_default, project='mno-'+diffeq_name+'-sweep', entity='blutjens')

    if args.no_wandb:
        config = config_default
    else:
        config = config_default
    #    config = wandb.config
    #    if sweep:
    #        config, config_default = sync_nested_wandb_config(
    #            sweep_config, config_default, verbose=True)

    return config