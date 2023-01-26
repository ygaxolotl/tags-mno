import pickle
import numpy as np
from pathlib import Path

def pickle_dump(file, folder='data/', name='temp.obj'):
    """
    Dumps file 
    """
    folder = Path(folder)
    if not folder.exists():
        folder.mkdir(parents=True)
    with open(Path(folder,name), 'wb') as filehandler:
        pickle.dump(file, filehandler)
    return 1

def store_interim_data(store_path, u_args=None, u_target=None,
    rand_insts=None, u_prior=None, grid=None):
    """
    Args:
        u_args: see load_interim_data
        u_target: see load_interim_data
        rand_insts: see load_interim_data
        u_prior: see load_interim_data
        grid: see load_interim_data
        store_path str: Path to store data
    """
    print('Saving interim data. Load with --load_interim_data_path'\
        ' \"'+store_path+'\"')

    store_path = Path(store_path)
    folder = store_path.parent
    filename = store_path.stem
    data = {
        'u_args': u_args,
        'u_target': u_target,
        'rand_insts': rand_insts,
        'u_prior': u_prior,
        'grid': grid
    }
    pickle_dump(data, folder=folder, 
        name=filename+'.pickle')
    return 1

def load_interim_data(load_path, verbose=False):
    """
    Args:
        load_path str: Path to load data
    Returns:
        u_args {
            'y_param': np.array((n_samples_after_warmup, n_tgrid, K)): Parametrization of middle-res onto low-res effects
            'rand_param': np.array((n_samples_after_warmup, 1)) : Random parameter, currently undefined
            'rand_ic': (np.array((n_samples_after_warmup, K))
                np.array((n_samples_after_warmup, J, K))
                np.array((n_samples_after_warmup, I, J, K))
                ): Random initial condition
        } 
        u_target (np.array((n_samples_after_warmup, n_tgrid, K))
            np.array((n_samples_after_warmup, n_tgrid, J, K))
            np.array((n_samples_after_warmup, n_tgrid, I, J, K))
        )
        rand_insts np.array((n_samples_after_warmup, 1))
        u_prior see u_target
        grid see main_qgturb.py
    """
    if verbose: print(f'Loading interim data from: ', load_path+'.pickle')

    with open(r''+load_path+'.pickle', 'rb') as filehandler:
        data = pickle.load(filehandler)

    if 'u_args' in data.keys():
        u_args = data['u_args']
    else:
        u_args = None

    if 'u_target' in data.keys():
        u_target = data['u_target']
    else:
        u_target = None

    if 'rand_insts' in data.keys():
        rand_insts = data['rand_insts']
    else:
        rand_insts = None

    if 'u_prior' in data.keys():
        u_prior = data['u_prior']
    else:
        u_prior = None

    if 'grid' in data.keys():
        grid = data['grid']
    else:
        grid = None

    if verbose: print('Loaded interim data')
    return u_args, u_target, rand_insts, u_prior, grid

def get_fn_at_x(xgrid, fn, at_x):
    """
    Returns function at x

    Args:
        xgrid np.array(n_grid): Grid where function was evaluated
        fn np.array((n_samples,n_grid)): Multiple function evaluations at xgrid 
        at_x float: Value closest to the desired function evaluation   

    Returns:
        fn_at_x np.array(n_samples): Function evaluations that where closest to x 
    """
    x_id = np.abs(xgrid - at_x).argmin().astype(int)
    fn_at_x = fn[:,x_id]
    return fn_at_x

def wandb_to_dict(wandb_cfg):
    """
    Converts 1-layer nested wandb config to dict() config
    Args:
        wandb_cfg wandb.config(dict())
    Returns
        cfg dict(dict())
    """
    cfg = dict()
    for key in wandb_cfg.keys():
        if key == '_wandb': # Skip placeholder key
            continue
        if type(wandb_cfg[key]) is dict:
            cfg[key] = wandb_cfg[key].copy()
        else:
            cfg[key] = wandb_cfg[key]
    return cfg

def sync_nested_wandb_config(sweep_config, config_defaults, verbose=True):
    """
    Syncs default config dictionary and wandb sweep config. This is a workaround 
    because wandb doesn't natively support nested dictionaries. 
    Args:
        sweep_config run.config: Non-nested sweep config from wandb
        config_defaults dict(dict()) : Nested default config
    Updates:
        wandb.config(dict()): Nested wandb config that contains dictionaries and 
            matches returned config_defaults
    Returns: 
        wandb.config(dict()): Nested wandb config that contains dictionaries and 
            matches returned config_defaults
        config_defaults: Nested default config, changed by sweep parameters 
    """
    # Pull default config into wandb.config
    sweep_config.setdefaults(config_defaults)

    # Transfer sweep config into nested parameters
    for sweep_param_key in sweep_config.keys():
        if sweep_param_key == '_wandb': # Skip placeholder key
            continue
        for parent_key in config_defaults:
            if type(config_defaults[parent_key])==dict:
                if sweep_param_key in config_defaults[parent_key]:
                    sweep_config[parent_key][sweep_param_key] = sweep_config[sweep_param_key]
    # Update config_defaults with new parameters from sweep
    for key in config_defaults.keys():
        config_defaults[key] = sweep_config[key]

    return sweep_config, config_defaults
