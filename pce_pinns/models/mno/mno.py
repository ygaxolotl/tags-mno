import os
import json
import psutil
from pathlib import Path
import numpy as np
from pprint import pprint

import torch
import wandb

from pce_pinns.models.fno import fno2d_gym 
from pce_pinns.models.fno.fno_train import train_fno
from pce_pinns.models.mno.eval import measure_runtime_mno, eval_predictions
from pce_pinns.models.mno.config import get_trained_model_cfg
from pce_pinns.models.mno.dataloader import save_fno_data
from pce_pinns.models.mno.dataloader import init_val_dataloader
from pce_pinns.models.mno.dataloader import get_paths

import pce_pinns.utils.shaping as shaping
import pce_pinns.utils.plotting as plotting
import pce_pinns.utils.parallel as parallel
from pce_pinns.utils.utils import pickle_dump

def get_specifier(config):
    """
    Returns:
        specifier string: Terse specifier of config
    """
    specifier = (f"{config['model']['model_dims']}D"
        f"t{config['data_loader']['n_hist']}{config['data_loader']['n_fut']}"
        f"_d{config['model']['depth']}c{config['model']['n_channels']}"
        f"m{''.join(map(str,config['model']['n_modes']))}"
        )
    return specifier
# Todo: move to utils
import gc
def auto_garbage_collect(pct=80.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return

def test(grid,
    y_args=None,
    diffeq=None,
    eval_model_digest=None,
    config=None,
    run_parallel=False,
    measure_runtime=False,
    max_n_samples = 10e6,
    ):
    """
    Loads model, runs coupled inference, dumps predictions, and evaluates
    
    Args:
        grid np.array(): See fcnn.neural_net.interpolate_param_nn()
        y_args np.array(): Additional inputs. See fcnn.neural_net.interpolate_param_nn()
        diffeq DiffEquation(object): See fcnn.neural_net.interpolate_param_nn()
        eval_model_digest string: Hex digest code to trained model. 
        config dict(): Test config file. See fcnn.neural_net.interpolate_param_nn()
        run_parallel bool: If true, evaluates model via parallel processing
        measure_runtime bool: If true, creates log-log runtime plot
        max_n_samples int: Maximum number of test samples
    Returns:
        sol_true np.array(): Ground-truth solution. See interpolate_param_mno()
        sol_pred np.array(): Predicted solution. See interpolate_param_mno()
    """
    dir_out = Path(config['dir_out'])
    cfg = get_trained_model_cfg(eval_model_digest, dir_out)

    model_load = fno2d_gym.make_model(cfg["model"])
    model_load.load_state_dict(torch.load(str(dir_out / "{}_modstat.pt".format(eval_model_digest)))) 

    n_tgrid = int(grid.shape[1])
    val_loader = init_val_dataloader(n_tgrid, full_cfg=config, saved_cfg=cfg)

    # sol_true, sol_pred = eval_helmholtz(model_load, val_loader, grid):
    # Evaluate Lorenz96
    y_param_mode_ori = diffeq.y_param_mode
    diffeq.y_param_mode = 'full_subgrid_forcing'

    print('torch dtype: ', torch.get_default_dtype())
    # Initialize arrays for logging.
    n_samples_val = int(y_args.shape[0] * config['data_loader']['val_size'])
    max_n_samples = min(max_n_samples, n_samples_val)
    n_x1grid = y_args.shape[-2] # This is Lorenz96.K
    n_xdim = 1 # Lorenz96 is univariate for small- and large-scale state
    x_pred = torch.zeros((n_samples_val, n_tgrid, n_x1grid, n_xdim))
    x_true = torch.zeros((n_samples_val, n_tgrid, n_x1grid, n_xdim))
    y_param_pred = torch.zeros((n_samples_val, n_tgrid, n_x1grid, n_xdim))
    y_param_true = torch.zeros((n_samples_val, n_tgrid, n_x1grid, n_xdim))

    # Init parallel processing    
    predict_fn, predict_tasks = parallel.init_preprocessing(fn=predict, parallel=run_parallel, dir_spill=config['dir_spill'])

    # Run walk-forward inference
    model_load.eval()
    with torch.no_grad():
        print('Creating predictions.')
        for i, (x_true[i,...], y_param_true[i,...]) in enumerate(val_loader):
            if (i-1)%100==0: 
                print(f'sampleID={i}')
                parallel.print_memory()
                auto_garbage_collect(pct=80.0)
            predict_args = {
                'x_init': x_true[i,0,...],
                'model': model_load, 
                'diffeq': diffeq, 
                'n_tgrid': n_tgrid
            }
            # Create or queue prediction sample
            if not run_parallel:
                x_pred[i,...], y_param_pred[i,...] = predict_fn(**predict_args)
            else:
                predict_tasks.append(predict_fn(**predict_args))
                # Both models are deterministic, so no random seeds need to be updated
            
            # Break early during testing
            if i==max_n_samples:
                break

        # Parse parallel tasks
        if run_parallel:
            predict_tasks = parallel.get_parallel_fn(predict_tasks)
            for i in range(min(len(predict_tasks), max_n_samples)):
                x_pred[i,...], y_param_pred[i,...] = predict_tasks[i]

    diffeq.y_param_mode = y_param_mode_ori
    # Detach logged data
    sol_pred = x_pred.cpu().detach()
    sol_true = x_true.cpu().detach()
    y_param_pred = y_param_pred.cpu().detach()
    y_param_true = y_param_true.cpu().detach()

    # Dump predictions
    predictions = {
        'config': config,
        'grid': grid,
        'sol_pred': sol_pred.numpy(),
        'sol_true': sol_true.numpy(),
        'y_param_pred': y_param_pred.numpy(),
        'y_param_true': y_param_true.numpy(),
    }
    print('Saving predictions at', Path(config['dir_predictions'],eval_model_digest))
    pickle_dump(predictions, folder=config['dir_predictions'], name=eval_model_digest+'.pickle')
    
    # Create log-log runtime plot  
    if measure_runtime:
        measure_runtime_mno(eval_model_digest=eval_model_digest, 
            dir_out=dir_out, model_load=None, y_args_sample=None, 
            val_loader=None, coupled=True)

    return sol_true, sol_pred

def step(x, model, diffeq):
    """
    Predicts next state with MNO, i.e., calculates the effect of small-scale 
    dynamics with FNO and then steps the large-scale dynamics with the 
    differential equation.

    Args:
        x torch.Tensor((1, n_xgrid1, ..., n_xgridN, n_xdim)): Large-
            scale state at current time, t
        model torch.nn.Module: Small-scale model
        diffeq pce_pinns.solver.diffeq.DiffEq: Large-scale model
    Returns:
        x_next torch.Tensor((1, n_xgrid1, ..., n_xgridN, n_xdim): Predicted 
            large-scale state at next time, t+1
        y_param_pred torch.Tensor((1, n_xgrid1, ..., n_xgridN, n_ydim)): Predicted
            parametrization at current time, t
    """
    # Predict small-scale state
    y_param_pred = model(x)

    # Reduce input to fit diffeq input and predict large-scale state
    x_next = diffeq.step_large_scale(x[0,...,0], y_param_pred[0,...,0], verbose=True)

    return x_next[None,...,None], y_param_pred

def predict(x_init, model, diffeq, n_tgrid=200):
    """
    Integrates MNO forward in time
    
    Args:
        x_init torch.Tensor(n_xgrid1, ..., n_xgridN, n_xdim): Large-
            scale state at current time, t
        model torch.nn.Module: Small-scale model
        diffeq pce_pinns.solver.diffeq.DiffEq: Initialized large-scale model
        n_tgrid int: Number of predicted timesteps
    Returns:
        x_pred torch.Tensor(n_tgrid, n_xgrid1, ..., n_xgridN, n_xdim): Predicted large-scale state
        y_param_pred torch.Tensor(n_tgrid, n_xgrid1, ..., n_xgridN, n_xdim): Predicted
            parametrizations
    """
    # Initialize prediction arrays
    if hasattr(model[0], 'dtype'):
        torch.set_default_dtype(model[0].dtype)
    else:
        torch.set_default_dtype(torch.float32)
    x_pred = torch.zeros((n_tgrid, ) + x_init.shape)
    y_param_pred = torch.zeros((n_tgrid, ) + x_init.shape)

    # Init diffeq for parallel processing
    diffeq.minus = np.copy(diffeq.minus)
    diffeq.minus2 = np.copy(diffeq.minus2)
    diffeq.plus = np.copy(diffeq.plus)

    # Set large-scale initial state
    x_pred[0,...] = x_init[...]

    # Predict coupled low-res. variable
    for t in range(1,n_tgrid):
        x_pred[t:t+1,...], y_param_pred[t-1:t,...] = step(x=x_pred[t-1:t,...], model=model, diffeq=diffeq)

    return x_pred, y_param_pred

def interpolate_param_mno(grid, y, y_args=None,
        n_epochs=20, batch_size=None, n_layers=2, n_units=128,
        lr=0.001,
        target_name='u', 
        alpha_indices=None, n_test_samples=1, 
        diffeq=None, loss_name=None, normalize=False,
        u_params=None, grid_in_dims=(-1,),
        val_size=0.0, test_size=0.2,
        plot=False, rand_insts=None, debug=False,
        config=None, eval_model_digest=None,
        run_parallel=False, no_test=False,
        load_processed_data_path=None):
    """
    Args:
        grid: See fcnn.neural_net.interpolate_param_nn
        eval_model_digest string: See fno.test(). If not None, training is skipped
        load_processed_data_path string: Path to load processed MNO-ready data
    Returns:
        sol_true np.array(n_samples_val, n_xgrid1, ..., n_xgridN, n_xdim): Ground-truth solution
        sol_pred np.array(n_samples_val, n_xgrid1, ..., n_xgridN, n_xdim): Predicted solution
    """
    device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dir_out = Path(config['dir_out'])
    dir_plot = Path('doc/figures/'+diffeq.name+'/fno') # Path('doc/figures/helmholtz/fno')
    if not os.path.exists(dir_plot): 
        os.makedirs(dir_plot)

    ##########################
    ## Get Processed data----
    ##########################
    if load_processed_data_path:
        paths, _ = get_paths(full_path=load_processed_data_path)
        # Retrieving batch_size from folder name, assuimng it's path/to/raw/data/n<n_samples>_t<batch_size>
        # todo: properly handle batch, history, and chunk=snippet size.
        if load_processed_data_path[-1] == '/':
            load_processed_data_path = load_processed_data_path[:-1]
        config['data_loader']['batch_size'] = int(load_processed_data_path.split('/')[-1].split('t')[-1])
    else:
        # Converts interim to processed data
        auto_garbage_collect()
        ## Reshape data from general to FNO format and save
        x, y, dim_out, rand_insts, batch_size = shaping.shape_2D_to_2D_nn_input(grid=grid, 
            y=y, y_args=y_args, rand_insts=rand_insts, target_name=target_name, 
            batch_size=config['data_loader']['batch_size'], alpha_indices=alpha_indices,
            grid_in_dims=grid_in_dims)
        config['data_loader']['batch_size'] = batch_size

        # Save MNO-ready // processed data s.t., FNO can load it dynamically.
        if config['data_loader']['dir_store_ml_data'] is not None:
            paths = save_fno_data(x, y, y_args.shape[0], 
                config['data_loader']['val_size'], config['data_loader']['dir_store_ml_data'],
                config['data_loader']['test_size'], config['data_loader']['split_style'])

        # Calculate mean and standard deviation of dataset
        data_stats = {
            "means_x" : x.mean(axis=tuple(range(x.ndim-1))),
            "means_y" : y.mean(axis=tuple(range(y.ndim-1))),
            "stdevs_x" : x.std(axis=tuple(range(x.ndim-1))),
            "stdevs_y" : y.std(axis=tuple(range(y.ndim-1)))
        }
        print('Copy these data statistics into config file, if not done already: ')
        pprint(data_stats)

        # Delete interim data from memory
        del x
        del y
        del y_args
        del rand_insts

    ##########################
    ## Train model---
    ##########################
    if eval_model_digest is None:
        # Normalize data
        if normalize:
            raise NotImplementedError('normalization not implemented')
        else:
            scalerx = None
            scalery = None

        # Save FNO config file --> TODO: fuse with main_
        specifier = get_specifier(config)
        paths['f_cfg'] = dir_out / "{}.json".format(specifier)
        print('Saving FNO config at : ', paths['f_cfg'])
        if not os.path.exists(dir_out): 
            os.makedirs(dir_out)
        with open(paths['f_cfg'], "w") as f:f.write(json.dumps(config))

        ##########################
        ## train---
        ##########################
        fno_args = " ".join([
            f"--config {paths['f_cfg']}",
            f"--trainx {paths['f_xtrain']}",
            f"--trainy {paths['f_ytrain']}",
            f"--testx {paths['f_xtest']}",
            f"--testy {paths['f_ytest']}",
            f"--lossmsk {paths['f_lossmsk']}",
            f"--outdir {str(dir_out)}",
            f"--device {device}",
            f"--epochs {str(config['n_epochs'])}",
            "--verbose",
            "--overwrite",
        ])
        print("$ python train_fno.py", fno_args)
        eval_model_digest = train_fno(fno_args.split())

    ##########################
    ## Report training stats---
    ##########################
    print('Loading model config from: ', Path(dir_out, eval_model_digest))
    cfg = get_trained_model_cfg(eval_model_digest, dir_out)
    try:
        ls = cfg["loss"]["validation"]
        minls = min(ls)
        aminls = ls.index(minls)
        print("{0:.2g} @ep{1} Min Loss Val".format(minls, aminls)) 
        wandb.log({"minloss_val": minls})

        ls = cfg["loss"]["training"]
        minls = min(ls)
        aminls = ls.index(minls)
        print("{0:.2g} @ep{1} Min Loss Train".format(minls, aminls)) 
        wandb.log({"minloss_train": minls})
    
        if plot: plotting.plot_fno_training_curve(cfg['loss'], dir_plot)
    except:
        print('Loss has not been logged.')

    ###################
    ## Predict and evaluate---
    ###################
    auto_garbage_collect()
    if not no_test:
        sol_true, sol_pred = test(grid=grid, y_args=y_args, diffeq=diffeq,
            eval_model_digest=eval_model_digest,
            config=config, run_parallel=run_parallel)
    else:
        sol_true = np.zeros((1,1,1,1))
        sol_pred = np.zeros((1,1,1,1)) 
    return sol_true, sol_pred
