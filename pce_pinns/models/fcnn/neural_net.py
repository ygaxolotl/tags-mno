"""
Neural network
Author: 
"""
import os
import json
import time
import math
import numpy as np
from tqdm import tqdm
from pathlib import Path
from hashlib import md5

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from sklearn.preprocessing import MinMaxScaler

import pce_pinns.rom.pce as pce
import pce_pinns.models.fcnn.fcnn as fcnn
import pce_pinns.models.losses as losses
from pce_pinns.models.dataloader import init_dataloader, create_and_store_splits
import pce_pinns.utils.shaping as shaping
import pce_pinns.utils.plotting as plotting
import pce_pinns.utils.parallel as parallel
from pce_pinns.utils.utils import pickle_dump

def train(model, train_loader, optimizer, criterion, n_epochs, device, plot=False, 
    custom_rand_inst=False, batch_size=None, eval_args={}):
    """
    Args:
        custom_rand_inst bool: If true, the random instances of the NN-based model and training data are the same   
        batch_size int: Batch size; only used to set rand_inst
    """
    timey = False
    times = []
    if timey: times.append({'start': time.time()})
    # Init logger
    val_interval = int(math.ceil(float(n_epochs) / 15.)) # in n_epochs
    print('Validation interval: ', val_interval)
    log_interval = 1
    wandb.define_metric('train_loss', define_metric='batch', summary='min')
    if timey: times.append({'wandb': time.time()})

    print("Begin training.")
    for e in tqdm(range(1, n_epochs+1)):
        
        # TRAINING
        train_epoch_loss = 0
        pce_epoch_loss = 0
        pinn_epoch_loss = 0

        model.train()
        if timey: times.append({'.train()': time.time()})

        for batch_idx, (X_train_batch, y_train_batch) in enumerate(train_loader):
            #print('i-th batch:', i, y_train_batch[0])
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            if timey: times.append({'.to()': time.time()})
            # Track input gradients to compute PINN loss 
            #if type(criterion).__name__ is 'PcePinnLoss':
            #    X_train_batch.requires_grad = True
            # Forward pass: Compute predicted y by passing x to the model    
            y_train_pred = model(X_train_batch)
            if timey: times.append({'.model()': time.time()})

            loss_dict = losses.calculate_losses(criterion, y_train_pred, y_train_batch, 
                y_inputs=X_train_batch, batch_idx=batch_idx, custom_rand_inst=custom_rand_inst)
            loss = loss_dict['loss']
            if timey: times.append({'.calculate_losses()': time.time()})

            # Zero gradients, perform a backward pass, and update the weights. 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if timey: times.append({'optimizer': time.time()})  

            # Log
            train_epoch_loss += loss.item()
            pce_epoch_loss += loss_dict['pce_loss']
            pinn_epoch_loss += loss_dict['pinn_loss']

            if batch_idx % log_interval == 0:
                wandb.log({"train_loss": loss.item(),
                    "train_pce_loss": loss_dict['pce_loss'],
                    "train_pinn_loss": loss_dict['pinn_loss'],
                    })#, step=e*len(train_loader)+batch_idx)
            if timey: times.append({'logging': time.time()})
        if timey: times.append({'train_batch': time.time()})

        # Validation
        if val_interval != 0:
            if e % val_interval == 0:
                eval_args['model'] = model
                #eval_args['plot'] = False
                _,_ = eval(**eval_args)
                if timey: times.append({'.eval()': time.time()})
                model.train()
                if timey: times.append({'.train()': time.time()})

        if type(criterion).__name__ == 'PcePinnLoss':
            print(f'Epoch {e+0:03} | Train Loss: {train_epoch_loss/len(train_loader):.6f} ' + 
                 f'PCE Loss: {pce_epoch_loss/len(train_loader):.6f} ' + 
                 f'PINN Loss: {pinn_epoch_loss/len(train_loader):.6f} ')
        else:
            print(f'Epoch {e+0:03} | Train Loss: {train_epoch_loss/len(train_loader):.6f}')
        
        if times:
            t_times = []
            for key in ['.train()', '.wandb()', '.to()', '.model()', '.calculate_losses()', 'optimizer', 'logging', 'train_batch', 'eval()']:
                t_times.append({key: np.mean([ts[list(ts)[0]]-times[i-1][list(times[i-1])[0]] for i, ts in enumerate(times) if list(ts.keys())[0]==key])})
            print(t_times)
    return model

def predict(model, x_test, device, target_name='pce_coefs', 
    alpha_indices=None, n_test_samples=1, plot=False,
    # Todo: check if these input arguments are necessary
    xgrid=None, tgrid=None, n_xgrid=None, n_tgrid=None, 
    y_test=None, debug=False):
    """
    TODO: figure out what's going on in this function.
    Args:
        x_test np.array((n_grid,1)): 

    Returns:
        y_test np.array((n_grid, n_alpha_indices))
        k_samples np.array((n_grid, n_samples))
    """
    n_test_samples = 50
    if debug:
        n_test_samples = 10 
    y_samples = torch.zeros((n_test_samples, n_tgrid*n_xgrid))

    print('sampling learned solution')
    for n in range(n_test_samples):
        y_samples[n,:] = pce.sample_pce_torch(pce_coefs=y_test, alpha_indices=alpha_indices)[:,0]

    y_samples = y_samples.cpu().numpy().reshape(n_test_samples, n_tgrid, n_xgrid)
    if True:# plot:
        plotting.plot_2d_sol_avg(xgrid, tgrid, y_samples)
        # plotting.plot_nn_k_samples(x_test[:,1], y_samples)

    import sys; sys.exit()
    import pdb;pdb.set_trace()

    k_samples = None
    with torch.no_grad():
        model.eval()
        x_test_batch = torch.from_numpy(x_test).to(device)# .type(torch.float32)
        y_test = model.predict(x_test_batch)
    if plot: 
        # Plot pce_coefs
        plotting.plot_nn_pred_1D(x=x_test_batch.cpu().numpy(), y=y_test.cpu().numpy())
    if target_name == 'k' or target_name=='k_true':
        # Sample param k, given learned deterministic pce_coefs
        k_samples = torch.zeros((n_test_samples, x_test.shape[0]))
        for n in range(n_test_samples):
            k_samples[n,:] = pce.sample_pce_torch(y_test, alpha_indices)[:,0]
        k_samples = k_samples.cpu().numpy()
        if plot:
            plotting.plot_nn_k_samples(x_test[:,0], k_samples)
    return y_test.cpu().numpy(), k_samples

def test():
    """
    todo
    """
    # Create test grid:
    x_test[batch_idx] = shaping.get_1sample_0D_nn_input(grid, y_args, dim_in=dim_in, 
        target_name="u", grid_in_dims=grid_in_dims, i=0)
    if normalize:
        x_test[batch_idx] = scalerx.transform(x_test[batch_idx])
    x_test_batch = torch.from_numpy(x_test[batch_idx]).to(device) # .type(torch.float32)

    # with torch.no_grad():
    # eval.predict()
    
    return 

def eval(model, val_loader, criterion, grid, y_args, device, dim_in, dim_out, grid_in_dims, 
        normalize=False, scalerx=None, scalery=None, n_val_samples=1, mode='batch',
        plot=False, custom_rand_inst=False, diffeq=None, dir_predictions=None,
        config=None, run_parallel=False):
    """
    Eval model that predicts solution, u   
    # TODO: This functoin has only been tested for model that predicts the solution, u, and needs more testing for PCE coefficients.

    Args:
        model torch.model()
        val_loader
        criterion 
        grid np.array((n_samples, n_grid)) or 
            (np.array(n_xgrid), np.array(n_tgrid)) or 
            np.array((n_samples, n_tgrid, n_x1grid, n_x2grid,..., dim_grid)): grid for 1D, 2D, or ND input, respectively
        y_args np.array((n_samples, n_grid, dim_y_args)) or 
            np.array(n_samples, n_tgrid, n_x1grid, n_x2grid,..., dim_y_args): Additional inputs, e.g., forcing terms 
        device torch.device()
        dim_in int: Number of input dimensions of the NN.
        dim_out int: Number of output dimensions of the NN.
        grid_in_dims tuple(int): Indices of grid dimensions that are used as input, e.g., 0, 1 for tgrid and x1grid. -1 for all grids
        normalize bool: If true, normalize data.
        scalerx sklearn.preprocessing.MinMaxScaler
        scalery sklearn.preprocessing.MinMaxScaler
        n_val_samples int: Number of val samples to generate
        mode str: Prediction mode, e.g., for 'batch' or 'sequential'. For 'sequential', model uses the predicted instead of ground-truth y as input (if y is an input)
        plot bool: If true, plot pred vs. target of first sample 
        diffeq DiffEquation(object): Differential equation object. At least necessary for PINN loss and 'coupled' eval().
        dir_predictions str: Directory to store predictions
    Returns:
        x_val np.array((n_val_samples, n_tgrid, n_x1grid, ..., dim_grid)): Test grid
        y_val_pred np.array((n_val_samples, n_tgrid, n_x1grid, ..., dim_y)): Predicted val solution
    """
    # Init metrics logger
    log_interval = 1
    pred_runtime = 0.
    val_epoch_loss = 0
    pce_epoch_loss = 0
    pinn_epoch_loss = 0
    val_epoch_loss_x = 0

    # Reshaping grid to univeral shape (n_samples, n_tgrid, n_x1grid, n_x2grid,..., dim_grid)
    grid_shp = grid.shape 
    if mode == 'sequential_coupled_local': # param-no-mem
        n_xngrid = y_args.shape[2:-1]
        grid_shp = (grid_shp[:2] + n_xngrid + (grid_shp[-1],))
        # Only save 1 grid sample to save memory, assuming constant grid across samples 
        grid = grid[:1,...] 
        for n_xgrid in n_xngrid:
            grid = np.repeat(grid[...,None,:], repeats=n_xgrid, axis=-2)
    
    # Set dimensions
    n_val_samples = len(val_loader)
    dim_y = val_loader.dataset.y_data.shape[-1]
    if dim_out != dim_y:
        y_out = torch.zeros((n_val_samples,) + grid.shape[1:-1] + (dim_out,))
    if len(grid_in_dims) == 0:
        dim_grid_in = 0
    elif grid_in_dims[0] == (-1):
        dim_grid_in = grid.shape[-1]
    else:
        dim_grid_in = len(grid_in_dims)
    dim_y_args_wo_yprev = dim_in-dim_y
    dim_x = dim_in-dim_y-dim_grid_in
    if 'coupled' in mode:
        # if args.est_lorenz == 'no-scale-sep-param-no-mem'
        y_param_mode_ori = diffeq.y_param_mode
        diffeq.y_param_mode = 'full_subgrid_forcing' # TODO: set mode depending on config.est_lorenz
        dim_x = dim_in - dim_grid_in # TODO: this assumes y is not an input! Needs to be changed for 'no-mem' not in config.est_lorenz
        dim_y_args_wo_yprev = dim_in 

    # Init data log 
    x_val = np.zeros((n_val_samples,) + grid.shape[1:-1] + (dim_in,)) # log
    y_val_pred = torch.zeros((n_val_samples,) + grid.shape[1:-1] + (dim_y,))
    y_val_true = np.zeros((n_val_samples,) + grid.shape[1:-1] + (dim_y,))
    x_val_pred = torch.zeros((n_val_samples,) + grid.shape[1:-1] + (dim_x,))
    x_val_true = np.zeros((n_val_samples,) + grid.shape[1:-1] + (dim_x,))

    for batch_idx, (x_val_target, y_val_target) in tqdm(enumerate(val_loader)):
        x_val[batch_idx] = x_val_target.cpu().numpy().reshape(grid.shape[1:-1] + (dim_in,))
        y_val_true[batch_idx] = y_val_target.cpu().numpy().reshape(grid.shape[1:-1] + (dim_y,))
        x_val_true[batch_idx] = x_val_target[:,dim_grid_in:dim_y_args_wo_yprev].cpu().numpy().reshape(grid.shape[1:-1] + (dim_x,))
        x_val_target, y_val_target = x_val_target.to(device), y_val_target.to(device)
        
        with torch.no_grad():
            model.eval()

            if mode == 'batch':
                # Predict as batch over time, e.g., Y(t) = NN(t)
                if dim_out == dim_y:
                    y_val_pred[batch_idx] = model.predict(x_val_target).reshape(grid.shape[1:-1] + (dim_y,))
                else:
                    # Predict PCE coefficients
                    y_out_pred = model.predict(x_val_target)
                    y_val_pred[batch_idx] = pce.sample_pce_torch(y_out_pred, 
                        criterion.alpha_indices, rand_inst=criterion.rand_inst
                        ).reshape(grid.shape[1:-1] + (dim_y,)) # Predict y
                    y_out[batch_idx] = y_out_pred.reshape(grid.shape[1:-1] + (dim_out,))
            elif mode == 'sequential_coupled_local':                
                # Initial state
                xprev = x_val_target.reshape(grid.shape[1:-1] + (dim_x,))[0]
                # Log init state
                x_val_pred[batch_idx, 0] = xprev
                y_val_pred[batch_idx, 0] = y_val_target.reshape(grid.shape[1:-1] + (dim_x,))[0]
                # np.all(x_val_target == y_args[0,...])
                y_t = torch.zeros((grid.shape[2],1))
                for t in range(1,grid.shape[1]): # Start predicting with model after lag
                    start = time.time()
                    for k in range(grid.shape[2]): # Iterate over spatial nodes
                        y_t[k] = model.predict(xprev[k])
                    xprev = diffeq.step_large_scale(x=xprev, f_y=y_t)
                    # Log
                    pred_runtime += time.time() - start
                    x_val_pred[batch_idx, t] = xprev
                    y_val_pred[batch_idx, t] = y_t
            elif 'sequential' in mode:
                # Predict sequentially over time, e.g., Y(t+1) = NN(Y(t))
                
                if dim_out != dim_y:
                    raise NotImplementedError("NN predicting PCE coefs in time-series is not implemented.")
                # Initial state
                yprev = x_val_target[0:1,-dim_y:]
                xprev = x_val_target[0:1,dim_grid_in:dim_y_args_wo_yprev]
                # Log ground-truth during lag
                y_val_pred[batch_idx,0] = yprev.reshape((1,) + grid.shape[2:-1] + (dim_y,)) # 1, n_x1grid, n_x2grid, ..., dim_y    
                x_val_pred[batch_idx,0] = xprev.reshape((1,) + grid.shape[2:-1] + (dim_x,))

                for t in range(1,grid.shape[1]): # Start predicting with model after lag
                    start = time.time()
                    if 'coupled' in mode:
                        # Predict sequentially over time while coupling NN output and physics model
                        # E.g., Y(t) = NN(X(t-1), Y(t-1)); X(t) = Physics(X(t), Y(t))
                        # if args.est_lorenz == "no-scale-sep-param-no-mem" or "no-scale-sep"
                        yargs_wo_yprev = torch.cat((x_val_target[t:t+1,:dim_grid_in], xprev),axis=1)
                    else:
                        yargs_wo_yprev = x_val_target[t:t+1,:dim_y_args_wo_yprev]
                    #TODO: write this: if args.est_lorenz == "no-scale-sep":
                    if dim_in != dim_y_args_wo_yprev:
                        x_t = torch.cat((yargs_wo_yprev, yprev), axis=1)
                    else: # if args.est_lorenz == "no-scale-sep-param-no-mem"
                        x_t = yargs_wo_yprev
                    y_t = model.predict(x_t)

                    if 'coupled' in mode: # Step coupled physics model
                        xprev = diffeq.step_large_scale(x=xprev[0], f_y=y_t[0]).reshape(1,-1)
                        x_val_pred[batch_idx, t] = xprev.reshape((1,) + grid.shape[2:-1] +(dim_x,))
                    yprev = y_t

                    # Log 
                    pred_runtime += time.time() - start
                    y_t = y_t.reshape((1,) + grid.shape[2:-1] + (dim_y,)) # 1, n_x1grid, n_x2grid, ..., dim_y    
                    y_val_pred[batch_idx,t] = y_t
        
            # Calculate validation loss
            if dim_out == dim_y:
                y_out_pred = y_val_pred[batch_idx]
            if mode == 'sequential_coupled_local':
                y_out_pred = y_val_pred[batch_idx].reshape(-1,1)
                x_out_pred = x_val_pred[batch_idx].reshape(-1,1)
            else:
                x_out_pred = x_val_pred[batch_idx]
            loss_dict = losses.calculate_losses(criterion, y_out_pred, y_val_target, 
                y_inputs=x_val[batch_idx], batch_idx=batch_idx, custom_rand_inst=custom_rand_inst)
            if 'coupled' in mode:
                loss_dict_x = losses.calculate_losses(criterion, x_out_pred, 
                    x_val_target[:,dim_grid_in:dim_y_args_wo_yprev],batch_idx=batch_idx)
                val_epoch_loss_x += loss_dict_x['loss'].item()

            # Log 
            val_epoch_loss += loss_dict['loss'].item()
            pce_epoch_loss += loss_dict['pce_loss']
            pinn_epoch_loss += loss_dict['pinn_loss']


    y_val_pred = y_val_pred.cpu().numpy()
    if np.any(np.isnan(y_val_pred)):
        print('WARNING: predicted y_val contains nan values. Converting to 0')
        y_val_pred = np.nan_to_num(y_val_pred, copy=False, nan=0.)
    if 'coupled' in mode:
        x_val_pred = x_val_pred.cpu().numpy()
        if np.any(np.isnan(x_val_pred)):
            x_val_pred = np.nan_to_num(x_val_pred, copy=False, nan=0.)
        diffeq.y_param_mode = y_param_mode_ori

    # De-normalize data
    if normalize:
        if dim_out != dim_y and mode == "sequential":
            raise NotImplementedError("NN predicting PCE coefs in time-series is not implemented.")
        x_val = x_val.reshape((n_val_samples * np.prod(grid.shape[1:-1]),) + (x_val.shape[-1],))
        x_val = scalerx.inverse_transform(x_val)
        y_val_pred = y_val_pred.reshape((n_val_samples * np.prod(grid.shape[1:-1]),)+(dim_y,))
        y_val_pred = scalery.inverse_transform(y_val_pred)

    x_val = x_val.reshape((n_val_samples,) + grid.shape[1:-1] + (x_val.shape[-1],))
    y_val_pred = y_val_pred.reshape((n_val_samples,) + grid.shape[1:-1] + (dim_y,))

    # Dump predictions
    if dir_predictions is not None:
        predictions = {
            # 'config': config, # todo: including config throws *** KeyError: '__getstate__'
            'grid': grid,
            'sol_pred': x_val_pred,
            'sol_true': x_val_true,
            'y_param_pred': y_val_pred,
            'y_param_true': y_val_true,
        }
        eval_model_digest,_ = get_specifier(config._items)
        print('Saving predictions at', Path(config['dir_predictions'],eval_model_digest))
        pickle_dump(predictions, folder=config['dir_predictions'], name=eval_model_digest+'.pickle')

    # Logging
    try:
        wandb.log({"runtime/tstep NN": pred_runtime/float(grid.shape[1]-1)/float(n_val_samples)})
        wandb.log({"val_loss": val_epoch_loss/float(len(val_loader)),
            "val_pce_loss": pce_epoch_loss/float(len(val_loader)),
            "val_pinn_loss": pinn_epoch_loss/float(len(val_loader)),
            }) # todo add step
        if 'coupled' in mode:
            wandb.log({"val_loss_x": val_epoch_loss_x/float(len(val_loader))})
    except:
        print('activate wandb for logging')

    # Plot predictions
    if plot:
        print(f'          | Val   Loss: {val_epoch_loss/float(len(val_loader)):.6f} | '+
            f'PCE Loss: {pce_epoch_loss/float(len(val_loader)):.6f} | ' +
            f'PINN Loss: {pinn_epoch_loss/float(len(val_loader)):.6f}')
        if 'coupled' in mode:
            print(f'          | Val-X Loss: {val_epoch_loss_x/float(len(val_loader)):.6f}')
        if 'sequential' in mode:
            plotting.plot_nn_lorenz96_solx(grid[0,...,0], y_val_true[0], y_val_pred[0], title='soly')
            plotting.plot_nn_lorenz96_err(grid[0,...,0], y_val_true, y_val_pred, title='erry')
            if 'coupled' in mode:
                plotting.plot_nn_lorenz96_solx(grid[0,...,0], x_val_true[0], x_val_pred[0], title='solx')
                plotting.plot_nn_lorenz96_err(grid[0,...,0], x_val_true, x_val_pred, title='errx')
        elif dim_in == 1:
            plotting.plot_nn_pred_1D(x=x_val, y=y_val_pred)
        elif dim_in == 2:
            plotting.plot_nn_pred_2D(x=x_val, y_pred=y_val_pred, y_target=y_val_true)
            plotting.plot_nn_pred_2D_extra_std(x=x_val, y_pred=y_val_pred, y_target=y_val_true)
            if dim_out != dim_y:
                plotting.plot_nn_pred_pce_coefs_2D(x=x_val, y=y_out)
        elif dim_in == 5 and dim_out == 16:
            # plotting.plot_nn_lorenz96_solx(grid[0,...,0], y_val_true[0], y_val_pred[0])
            plotting.plot_nn_lorenz96_sol_xnull(grid[0,...,0], y_val_true[0], y_val_pred[0])

    return x_val, y_val_pred

def save_fno_2D_data_lorenz96_MAKE_BEAUTIFUL(x, y, config):
    """        
    Args:
        x np.array((n, dim_in)): 1D input array, see shaping.flatten_to_0D_data for info
        y np.array((n, dim_y)): 1D output array, see shaping.flatten_to_0D_data for info
                            e.g., 
                            n = n_samples * n_tgrid
                            dim_in = k + k*j
                            dim_y = k
    Saves:
        f_xtrain np.array((n*train_size, j, k, 2): t and y
        f_xtest np.array((n*val_size, j, k, 2)):
        f_ytrain np.array((n*train_size, j, k, 1))): y
        f_ytest np.array((n*val_size, j, k, 1)): 
        f_lossmsk np.array((j,k,1)): Loss mask; zero for masked values
        f_data_cfg dict(): Config 
    """
    from pathlib import Path
    import pdb;pdb.set_trace()

    k = config['de']['K']
    j = k
    X_jk = np.repeat(x[:,np.newaxis, 1:k+1], axis=-2, repeats=j) 
    Y_jk_in = x[:,k+1:].reshape(-1,j,k) 
    fno_in = np.concatenate((X_jk[...,np.newaxis], Y_jk_in[...,np.newaxis]), axis=-1)
    Y_jk_target = y.reshape(-1,j,k)[...,np.newaxis]
    loss_mask = np.ones(Y_jk_target[0].shape, dtype=bool)
    
    # Save data
    import pdb;pdb.set_trace()
    d_proc = Path('data/processed/temp/fcnn/lorenz96_k{:d}'.format(config['de']['K']) + 
        '_n{:d}'.format(config['de']['n_samples']) + 
        '_t{:d}'.format(int(x.shape[0]/config['de']['n_samples'])))
    if not os.path.exists(d_proc): 
        os.makedirs(d_proc)
    f_xtrain = d_proc / "xtrain.npy" 
    f_xtest = d_proc / "xtest.npy" 
    f_ytrain = d_proc / "ytrain.npy" 
    f_ytest = d_proc / "ytest.npy" 
    f_lossmsk = d_proc / "notlandbool.npy" 
    f_lossmsk = d_proc / "data_cfg.npy" 
    
    n_train = int((1-config['data_loader']['val_size'])*fno_in.shape[0])
    n_val = int(config['data_loader']['val_size']*fno_in.shape[0])
    np.save(f_xtrain, fno_in[:n_train])
    np.save(f_xtest, fno_in[-n_val:])
    np.save(f_ytrain, Y_jk_target[:n_train])
    np.save(f_ytest, Y_jk_target[-n_val:])
    np.save(f_lossmsk, loss_mask)
    np.save(f_data_cfg, config)
    return 1

def get_specifier(config):
    """
    Returns model specifier and digest
    """
    specifier = ('fcnn_l{:02d}'.format(config['model']['n_layers'])
        + 'u{:04d}e{:03d}'.format(config['model']['n_units'],config['n_epochs'])
        + 'lr'+('{:.5f}'.format(config['optimizer']['lr']).split('.')[1]) # only decimal digits
        + 'n{:06d}'.format(config['de']['n_samples'])
        + 'k{}'.format(config['de']['K']))
    config_json = json.dumps(config)
    digest = md5(config_json.encode("utf-8")).hexdigest()[:10]
    return specifier, digest

def get_trained_model_cfg(digest, dir_out):
    """
    Returns trained model config given digest
    """
    with (dir_out / "{}_cfg.json".format(digest)).open() as jf:
        cfg = json.load( jf )
    return cfg

def dump(model, config, path, overwrite=False):
    """
    Stores model and config
    Args:
        model torch.model
        config wandb.config or dict()
        path string or Path: Root Path to store model and config
        overwrite bool: If true, overwrites existing files
    Returns
        specifier string: Interpretable string that specifies model config and is used for the filename
        digest string: Uninterpretable hexcode of the config dictionary
    """
    # Store model 
    path = Path(path)
    if not os.path.exists(path): 
        os.makedirs(path)
    if not path.is_dir():
        raise ValueError(f"{path} is not a valid directory.")

    config_json = json.dumps(config)
    specifier, digest = get_specifier(config)

    config_file = path / (specifier + "_cfg.json")
    model_file = path / (specifier + "_modstat.pt")

    if not overwrite and (config_file.is_file() or model_file.is_file()):
        raise ValueError(f"{config_file} or {model_file} already exists.")

    with open(config_file, "w") as f:
        f.write(config_json)

    torch.save(model.state_dict(), model_file)

    print('Model stored at: ', model_file)
    print('Config stored at: ', config_file)

    return specifier, digest

def torch_set_default_dtype(dtype):
    """
    Args:
        dtype str: According to numpy convention
    Returns:
        torch.dtype : default torch dtype
    """
    if dtype=='float16':
        torch.set_default_dtype(torch.float16)
    if dtype=='float32':
        torch.set_default_dtype(torch.float32)
    if dtype=='float64':
        torch.set_default_dtype(torch.float64)
    return torch.zeros((1,)).dtype

def interpolate_param_nn(grid, y, y_args=None,
        n_epochs=20, batch_size=None, n_layers=2, n_units=128,
        lr=0.001,
        target_name='pce_coefs', 
        alpha_indices=None, n_test_samples=1, 
        diffeq=None, loss_name=None, normalize=False,
        u_params=None, grid_in_dims=(-1,),
        test_mode='batch',
        val_size=0.0, test_size=0.2,
        plot=False, rand_insts=None, debug=False,
        config=None, eval_model_digest=None,
        overwrite=True,
        run_parallel=False):
    """
    Train interpolating NN that predicts params, PCE coefs, or param eigenvecs 

    Args:
        grid np.array((n_samples, n_grid)) or 
            (np.array(n_xgrid), np.array(n_tgrid)) or 
            np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, ..., dim_grid)): grid for 1D, 2D, or ND input, respectively
        y np.array((n_samples, n_grid, dim_y)) or
            np.array((n_samples, n_tgrid, n_xgrid, dim_y)): Target solution, e.g., temperature
        y_args np.array((n_samples, n_grid, dim_y_args)) or 
            np.array(n_samples, n_tgrid, n_x1grid, n_x2grid,..., dim_y_args): Additional inputs, e.g., forcing terms 
        target_name (string): Indication which target
        alpha_indices np.array((n_alpha_indices, ndim)): Set of alpha indices. If not None, used as prediction target and in PceLoss. See rom.pce.multi_indices()
        diffeq DiffEquation(object): Differential equation object. At least necessary for PINN loss and 'coupled' eval().
        loss_name string: Sets loss, e.g., mseloss, pcepinnloss, pceloss 
        u_params dict(np.array((n_samples, n_grid, dim_out))): Parameters of each differential equation sample; only used for PINN loss
        rand_insts np.array((n_samples, dim_rand)): Random instances, used to generated training dataset
        grid_in_dims tuple(int): Indices of grid dimensions that are used as input, e.g., 0, 1 for tgrid and x1grid. -1 for all grids
        config dict(dict()): Config. Could be wandb.config or dictionary.
        eval_model_digest string: Filename of stored model for evaluation 
        overwrite bool: If true, overwrites model and config files after training
    Returns:
        y_test np.array(n_grid,dim_out) or
            np.array((n_test_samples, n_tgrid, n_x1grid, ..., dim_out)): Direct output of neural network on test grid
        u_test np.array(?): Solution, based on NN output, on test grid

    """
    """
    TODO START FROM HERE-
    --- Make sure that train and val loss are calculated equal, because it's kinda weird that they're so far off?
    --- Add regularizer?
    --- EVALUATE x and y accuracy with NN + superparametrization vs. superparametrization (w null param); 
    --- EVALUATE runtime as fn of dimensionality: : NN + superparametrization (0.0001699s) vs. high-res. xyz  
    --- EVALUATE distribution over forcing? parameters: create stochastic dataset. Use train, test split to get good dataset --- 
    """
    # init cpu/gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using ', device)
    torch_dtype = torch_set_default_dtype(y.dtype.name)

    # init logger
    config = wandb.config
    dir_out = Path(config.dir_out)

    # Reshape into neural net data shape:
    x, y, dim_out, rand_insts, batch_size = shaping.flatten_to_0D_data(grid=grid, y=y, y_args=y_args, 
        rand_insts=rand_insts, target_name=target_name, batch_size=batch_size, alpha_indices=alpha_indices,
        grid_in_dims=grid_in_dims, mode=test_mode)

    # Get dimensions
    dim_in = x.shape[-1]   

    # todo: delete this conversion into FNO2D inputs. 
    # if config['data_loader']['save_fno_data']:
    #     save_fno_2D_data_lorenz96_MAKE_BEAUTIFUL(x, y, config)

    # Normalize data: TODO: move fn into model as normalization layer?
    if normalize:
        norm_range_y = (0,1) # -2,5
        norm_range_x = (0,1) # -5,5
        if test_mode=='sequential':
            assert np.all(norm_range_y == norm_range_x), "In sequential"\
                "test model x and y normalization range has to be equal."
        # Normalizes feature-wise 
        scalery = MinMaxScaler(feature_range=norm_range_y)
        y = scalery.fit_transform(y) 
        scalerx = MinMaxScaler(feature_range=norm_range_x)
        x = scalerx.fit_transform(x)
    else:
        scalerx = None
        scalery = None

    n_epochs = config['n_epochs']
    if debug: n_epochs = 2

    paths = create_and_store_splits(x, y, batch_size, 
        dir_store_ml_data=config['data_loader']['dir_store_ml_data'],
        test_size=config['data_loader']['test_size'], 
        val_size=config['data_loader']['val_size'])

    # Delete memory after having processed the variables 
    print('Deleting x and y from memory. Untested. Delete this print once tested.')
    del x
    del y

    train_loader, val_loader, test_loader = init_dataloader(paths,
        batch_size, test_size=config['data_loader']['test_size'], val_size=config['data_loader']['val_size'])

    # Select loss function
    if target_name == 'pce_coefs' or target_name == 'mu_k' or target_name == 'k_eigvecs':
        criterion = nn.MSELoss()
    elif target_name == 'k' or target_name == 'k_true' or \
        target_name == 'u' or target_name == 'u_true':
        if loss_name == 'pcepinnloss':
            criterion = losses.PcePinnLoss(diffeq=diffeq, u_params=u_params, 
                alpha_indices=alpha_indices, verbose=False, rand_insts=rand_insts, 
                a_pinn=0.1, debug=debug)
        elif loss_name == 'mseloss':
            criterion = nn.MSELoss()
        elif loss_name == 'pceloss':
            criterion = losses.PceLoss(alpha_indices=alpha_indices, verbose=False, 
            rand_insts=rand_insts)#, test_return_pce_coefs=False)
        else:
            raise NotImplementedError("Loss function left undefined in neural_net.interpolate_param_nn()")
    
    # Define evaluation function
    eval_args = {'model': None, 
        'val_loader': val_loader, 
        'criterion': criterion,
        'grid': grid, 'y_args': y_args,
        'device': device, 
        'dim_in': dim_in, 'dim_out': dim_out, 
        'grid_in_dims': grid_in_dims,  
        'normalize': normalize, 'scalerx': scalerx, 'scalery': scalery, 
        'n_val_samples': n_test_samples, 'mode': test_mode,
        'plot': plot, 'custom_rand_inst': (rand_insts is not None),
        'diffeq': diffeq,
        'dir_predictions': None,
        'config':config,
        'run_parallel':run_parallel
    }
    ##
    # Train
    ##
    if eval_model_digest is None:
        model = fcnn.FCNN(dim_in, dim_out=dim_out, 
            n_layers=config['model']['n_layers'], 
            n_units=config['model']['n_units'])
        model.to(device)
        print(model)
        eval_args['model'] = model

        optimizer = optim.Adam(model.parameters(), lr=config['optimizer']['lr'])

        model = train(model, train_loader, optimizer, criterion, n_epochs, device, 
            plot=plot, custom_rand_inst=(rand_insts is not None), batch_size=batch_size,
            eval_args=eval_args)

        print('Done training. Saving model.')
        eval_model_digest, _ = dump(model, config._items, path=dir_out, overwrite=overwrite)

    ###
    # Predict & Evaluate
    ###
    config = get_trained_model_cfg(eval_model_digest, dir_out)
    model_load = fcnn.FCNN(dim_in, dim_out=dim_out, 
        n_layers=config['model']['n_layers'], 
        n_units=config['model']['n_units'])
    model_load.to(device)
    model_load.load_state_dict(torch.load(dir_out / '{:s}'.format(eval_model_digest+'_modstat.pt')))

    # Query learned solution on x_test grid
    # eval_args['plot'] = True
    eval_args['model'] = model_load
    eval_args['dir_predictions'] = config['dir_predictions']
    x_test, y_test = eval(**eval_args)

    # Measure runtime
    msr_runtime = False
    m_samples = 1000
    if msr_runtime:
        if len(grid_in_dims) == 0:
            dim_grid_in = 0
        elif grid_in_dims[0] == (-1):
            dim_grid_in = grid.shape[-1]
        else:
            dim_grid_in = len(grid_in_dims)

        for batch_idx, (x_val_target, _) in enumerate(val_loader):
            x_val_target = x_val_target.to(device)
            t = 0
            # Create proper input shape
            if test_mode == 'batch':
                x_in = x_val_target
            elif 'sequential' in test_mode:
                dim_y = val_loader.dataset.y_data.shape[-1]
                if 'coupled' in test_mode:
                    xprev = x_val_target[0:1,dim_grid_in:dim_in-dim_y]
                    yargs_wo_yprev = torch.cat((x_val_target[t:t+1,:dim_grid_in], xprev),axis=1)
                else:
                    yargs_wo_yprev = x_val_target[t:t+1,:dim_y_args_wo_yprev]
                yprev = x_val_target[0:1,-dim_y:]
                x_in = torch.cat((yargs_wo_yprev, yprev), axis=1)
            
            start = time.time()
            for m in range(m_samples):
                y_t = model_load.predict(x_in)
            runtime = (time.time() - start)/float(m_samples)
            print(f'Average time: {runtime:.8f}s')
            break
    return y_test, x_test
