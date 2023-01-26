import os
import time
from pathlib import Path
import numpy as np
import pickle
import torch

from pce_pinns.models.mno.config import get_trained_model_cfg
from pce_pinns.models.losses import calc_infty_loss, calc_mse_loss

import pce_pinns.utils.shaping as shaping
import pce_pinns.utils.plotting as plotting
from pce_pinns.utils.utils import pickle_dump

def calculate_rmse_t(sol_true, sol_pred):
    """
    Calculates RMSE over time according to:
    RMSE(t) = 1/K sum_{k=0}^K sqrt{ 1/N sum_{i=0}^N (hat X_{k,i}(t) - X_{k,i}(t))^2 }

    Args:
        sol_true: np.array((n_samples, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim))
        sol_pred: np.array((n_samples, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim))
    Returns:
        rmse_t np.array(n_tgrid, n_xdim)
    """
    # se = np.square(sol_pred - sol_true) # Square error
    # mse = np.mean(se, axis=0) # Average over ensemble
    # rmse = np.sqrt(mse)
    # rmse_t = np.mean(rmse, axis=xgrid_axes) # Average over space
    xgrid_axes = tuple([i-1 for i in range(2,2+len(sol_true.shape)-3)]) # Axis IDs of xgrids
    rmse_t = np.mean(np.sqrt(np.mean(np.square(sol_pred-sol_true),axis=0)),axis=xgrid_axes)

    return rmse_t

def calculate_rmse(sol_true, sol_pred):
    """
    Calculates RMSE according to:
    RMSE = 1/T sum_{t=0}^T 1/K sum_{k=0}^K sqrt{ 1/N sum_{i=0}^N (hat X_{k,i}(t) - X_{k,i}(t))^2 }

    Args:
        sol_true: np.array((n_samples, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim))
        sol_pred: np.array((n_samples, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim))
    Returns:
        rmse np.array(n_xdim)
    """
    rmse = np.mean(calculate_rmse_t(sol_true, sol_pred),axis=0)
    return rmse

def calculate_mse(sol_true, sol_pred):
    """
    Calculates MSE according to:
    MSE = 1/T sum_{t=0}^T 1/K sum_{k=0}^K { 1/N sum_{i=0}^N (hat X_{k,i}(t) - X_{k,i}(t))^2 }

    Args:
        sol_true: np.array((n_samples, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim))
        sol_pred: np.array((n_samples, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim))
    Returns:
        mse np.array(n_xdim)
    """
    xgrid_axes = tuple([i-1 for i in range(2,2+len(sol_true.shape)-3)]) # Axis IDs of xgrids
    mse_t = np.mean(np.mean(np.square(sol_pred-sol_true),axis=0),axis=xgrid_axes)
    mse = np.mean(mse_t,axis=0)
    return mse

def load_predictions(folder, name):
    """
    Returns: {
            'config': test config
            'grid': see ND input at fcnn.neural_net.interpolate_param_nn
            'sol_pred': see mno.interpolate_param_mno
            'sol_true': see mno.interpolate_param_mno
            'y_param_pred': np.array((n_samples_val, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim))
            'y_param_true': np.array((n_samples_val, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim))
        }

    """
    with open(Path(folder,name), "rb") as input_file:
        predictions = pickle.load(input_file)
    return predictions

def eval_predictions(eval_model_digest, 
        dir_predictions='models/temp/lorenz96/mno/predictions/',
        dir_results='models/temp/lorenz96/mno/results/'):
    """
    Args:
        dir_predictions str: Path that contains predictions in dictionary form
        dir_results str: Path to store evaluation results
    Returns:
        results {
            'grid' np.array(n_samples, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim)
            'rmse_t' np.array(n_tgrid, n_xdim)
            'sol_true' np.array(n_samples, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim)
            'sol_pred' np.array(n_samples, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim)
        }
    """
    results = {}
    # Load predictions dictionary
    preds = load_predictions(folder=dir_predictions, name=eval_model_digest+'.pickle')
    results['grid'] = preds['grid']
    # Calculate results
    results['rmse_t'] = calculate_rmse_t(preds['sol_true'], preds['sol_pred'])

    # Calculate metrics over full range
    max_t = np.min((preds['sol_true'].shape[1], 200)) # Calculatei
    print(f'Calculating (R)MSE for the first {max_t} time steps')
    results['rmse'] = calculate_rmse(preds['sol_true'][:,:max_t,...], 
            preds['sol_pred'][:,:max_t,...])
    results['mse'] = calculate_mse(preds['sol_true'][:,:max_t,...],
            preds['sol_pred'][:,:max_t,...])
        
    # Dump results
    print('Saving MNO results at', Path(dir_results,eval_model_digest))
    pickle_dump(results, folder=dir_results, name=eval_model_digest+'.pickle')

    # Don't store predictions as they are stored in predictions.
    results['sol_true'] = preds['sol_true']
    results['sol_pred'] = preds['sol_pred']

    # Plot
    plt_id = 0 # Example to plot
    # Plot RMSE over time
    # plotting.plot_nn_lorenz96_rmse_over_t(grid[0,...,0], sol_true[:max_n_samples,...,0], sol_pred[:max_n_samples,...,0], title='rmse')
    # Plot large-scale
    # plotting.plot_nn_lorenz96_solx(grid[plt_id,:,0], sol_true[plt_id,...,0], sol_pred[plt_id,...,0], title='solx')
    # Plot parametrizations
    # plotting.plot_nn_lorenz96_solx(grid[plt_id,:,0], y_param_true[plt_id,...,0], y_param_pred[plt_id,...,0], title='soly')
    # plotting.plot_nn_lorenz96_long_term(grid[plt_id,:,0], sol_true[plt_id,...,0], sol_pred[plt_id,...,0])
    # plotting.plot_nn_lorenz96_long_term_err(grid[plt_id,:,0], sol_true[plt_id,...,0], sol_pred[plt_id,...,0])

    return results

def msr_runtime(fn, args, M=100, verbose=False):
    """
    Measures average runtime of function, fn
    """
    t = time.time()
    for m in range(M):
        _ = fn(args)
    avg_t = (time.time()-t) / float(M)
    if verbose:
        print('Avg runtime: ', avg_t)
    return avg_t

def measure_runtime_mno(eval_model_digest=None, dir_out=None, model_load=None, 
    y_args_sample=None, val_loader=None, coupled=False, dummy_init=True):
    """
    Creates log-log runtime vs. grid-size plot of FNO model. The runtimes
    are averaged across m_samples runs. Only the runtime of model() is timed.

    Args:
        eval_model_digest str: Digest of stored model, used to load model weights.
        dir_out Path: Path to stored model config file.
        model_load torch.nn.Module: FNO model. 1D or 2D (optional)
        y_args_sample torch.Tensor((n_batch,n_x1grid,{n_x2grid},n_dim)): Uses test sample, if supplied.
        val_loader torch.Dataloader: Uses val_loader samples instead of test sample, if supplied (Optional)
        coupled bool: If true, evaluates runtime with coupled differential equation
        dummy_init bool: If true, initalizes differential equation with zeros or ones instead of random values.
    Returns:
        runtimes list(): List of runtimes         
        Ks list(): List of grid sizes that have been sampled.
    """
    # Load model
    if eval_model_digest is not None and dir_out is not None:
        cfg = get_trained_model_cfg(eval_model_digest, dir_out)
        model_load = fno2d_gym.make_model(cfg["model"])
        model_load.load_state_dict(torch.load(str(dir_out / "{}_modstat.pt".format(eval_model_digest)))) 
    elif model_load is not None:
        pass
    else:
        print('Pass model to runtime measurement via stored or loaded model.')
    
    # Ensure Python is running on single thread.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Initialize number of repeated samples
    Ks = [8388608, 4194304, 2097152, 1048576, 524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4]
    #Ks = [1024, 512, 256, 128, 64, 32, 16, 8, 4] # 8192, 4096, 
    m_samples = (2000*np.ones(len(Ks),dtype=int)).astype(int) # Repeats time measurement m-times.
    #m_samples[0] = 1
    #m_samples[1] = 1
    #m_samples[2] = 1
    #m_samples[3] = 1
    #m_samples[4] = 10
    #m_samples[5] = 100
    #m_samples[6] = 100
    m_samples[Ks.index(8388608)] = 1 
    m_samples[Ks.index(4194304)] = 1
    m_samples[Ks.index(2097152)] = 5
    m_samples[Ks.index(1048576)] = 10
    m_samples[Ks.index(524288)] = 10 
    m_samples[Ks.index(262144)] = 20
    m_samples[Ks.index(131072)] = 20
    m_samples[Ks.index(65536)] = 10 
    m_samples[Ks.index(32768)] = 20
    m_samples[Ks.index(16384)] = 20
    m_samples[Ks.index(8192)] = 20
    m_samples[Ks.index(4096)] = 200
    m_samples[Ks.index(2048)] = 1000
    m_samples[Ks.index(1024)] = 2000
    m_samples[Ks.index(512)] = 2000
    m_samples[Ks.index(256)] = 2000 

    # Create sample input
    if y_args_sample is not None:
        y_args = y_args_sample
    elif val_loader is not None:
        for i, (y_args_val, y) in enumerate(val_loader):
            y_args = y_args_val[0:1,:,0,None] # 0:1 is time-step
            if i==0:
                break
    else:
        y_args = torch.rand((1,Ks[-1],1)) #, dtype=torch.float32

    # Iterate over grid sizes and measure runtimes.
    model_load.eval()
    with torch.no_grad():
        runtimes = []
        for n_i, K in enumerate(Ks):
            print(n_i, K)
            # TODO: for implement model.rebuild_dft_matrix
            model_load[0].f[1].sconv.is_init = False
            model_load[0].f[2].sconv.is_init = False
            if K > 4:
                factors = int(np.log2(K) - np.log2(4))
                # Create fake input by upscaling y_args   
                y_up = y_args.clone()
                for _ in range(factors):
                    if len(y_args.shape) == 4: # FNO2D config['model']['model_dims'] == 2: 
                        y_up = torch.cat((y_up,y_up), dim=(1))[:,:,:,:]
                        y_up = torch.cat((y_up,y_up), dim=(2))[:,:,:,:]
                    elif len(y_args.shape) == 3: # FNO1D
                        y_up = torch.cat((y_up,y_up), dim=(1))[:,:,:]
                    else:
                        raise NotImplementedError("Shape of input argument not recognized: ", y_args.shape)
                y_args_tst = y_up
            else:
                y_args_tst = y_args

            # Initialize coupling with differential equations model.
            if not coupled: 
                run_model = model_load
            else:
                from pce_pinns.solver.lorenz96 import init_sample_diffeq_instance
                
                diffeq, _ = init_sample_diffeq_instance(grid_size=K, dummy_init=dummy_init)

                def run_model(y_args_tst):
                    y_pred = model_load(y_args_tst)
                    _ = diffeq.step_large_scale(x=y_args_tst[0,...,0], f_y=y_pred[0,...,0], verbose=True)

            _ = run_model(y_args_tst) # Warmup to build dft matrix
            runtimes.append(msr_runtime(run_model, y_args_tst, M=int(m_samples[n_i])))
            print(f'N={K:5d}: Runtime {runtimes[-1]:.6f}s')

    print(runtimes)
    plotting.plot_lorenz96_runtimes(Ns=Ks, ts=runtimes)

    # Reset environment variables
    os.environ["OMP_NUM_THREADS"] = ""
    os.environ["NUMEXPR_NUM_THREADS"] = ""
    os.environ["MKL_NUM_THREADS"] = ""

    return runtimes, Ks

def eval_helmholtz(model_load, val_loader, grid):
    # Helmholtz specific. Rewrite for 1D Lorenz96.

    measure_runtime=False
    infty_losses = []  
    mse_losses = []
    runtimes = []
    N_maxs = [256, 128, 64, 32, 16, 8, 4]
    # N_maxs = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4]
    m_runtime_avgs = 100*np.ones(len(N_maxs)) # Repeats time measurement m-times.
    m_runtime_avgs[0] = 1# 1
    m_runtime_avgs[1] = 5# 1
    m_runtime_avgs[2] = 10# 10
    m_runtime_avgs[3] = 20# 
    
    example = 0
    for i, (x, y) in enumerate(val_loader):
        for n_i, N_max in enumerate(N_maxs):
            xgrid = grid[example,0,0,:,1]
            ygrid = np.flip(grid[example,0,:,0,2])

            if N_max <= 256:
                xgrid = shaping.downsample1d_log2(torch.from_numpy(xgrid[np.newaxis,:].copy()), tgt_size=N_max)
                ygrid = shaping.downsample1d_log2(torch.from_numpy(ygrid[np.newaxis,:].copy()), tgt_size=N_max)
                x = shaping.downsample2d_log2(x, tgt_size=N_max)
                y = shaping.downsample2d_log2(y, tgt_size=N_max)

            # Measure runtime
            if measure_runtime and i==0:
                print(n_i, N_max)
                if N_max > 256:
                    factors = int(np.log2(N_max) - np.log2(256))
                    x_up_fake = x.clone()
                    for _ in range(factors):
                        x_up_fake = torch.cat((x_up_fake,x_up_fake), dim=(1))[:,:-1,:,:]
                        x_up_fake = torch.cat((x_up_fake,x_up_fake), dim=(2))[:,:,:-1,:]
                    x_tst = x_up_fake
                else:
                    x_tst = x
                runtimes.append(msr_runtime(model_load, x_tst, M=int(m_runtime_avgs[n_i])))
                print(f'N={N_max:5d}: Runtime {runtimes[-1]:.6f}s')

            else:
                # Predict
                y_pred = model_load(x)
                data = torch.cat((y_pred.cpu().detach()[example], y.cpu().detach()[example]), -1)
                y_pred_np = data[...,0].numpy()
                y_true_np = data[...,1].numpy()

                # Evaluate infinity error
                infty_loss = calc_infty_loss(y_pred, y, grid_dims=(1,2))
                infty_loss = torch.mean(infty_loss,axis=0).detach().cpu().numpy()[0] # Average across batch
                infty_losses.append(infty_loss)
                mse_loss = calc_mse_loss(y_pred, y, grid_dims=(1,2))
                mse_loss = torch.mean(mse_loss, axis=0).detach().cpu().numpy()[0]
                mse_losses.append(mse_loss)
                print(f'N={N_max:5d}: Err_Infty {infty_loss: .6f}; MSE {mse_loss: .6f}')

                # Plot difference
                xgrid = grid[example,0,0,:(N_max+1),1]
                ygrid = np.flip(grid[example,0,:(N_max+1),0,2])
                try:
                    plotting.plot_fno_helmholtz_diff(xgrid, ygrid, y_pred_np, y_true_np, fname=f'sol_diff_Nmax{N_max}')
                except:
                    print(f'Failed diff plot for N:{N_max}')
                # Plot solution 
                try:
                    plotting.plot_fno_helmholtz(xgrid, ygrid, y_pred_np, y_true_np, fname=f'sol_pred_Nmax{N_max}')
                except:
                    print(f'Failed sol plot for N:{N_max}')

                # Plot solution
                #plotting.plot2Dfeats_fno(data, 
                #    fname=str(dir_plot / "{}ypred.png".format(specifier)), 
                #    featnames=["pred", f"true"])
        break       

    # Plot runtime over domain size
    if measure_runtime:
        plotting.plot_fno_runtimes_vs_N(np.asarray(N_maxs), np.asarray(runtimes))

    # Plot accuracy over domain size
    plotting.plot_fno_accuracy_vs_N(np.asarray(N_maxs), np.asarray(mse_losses), np.asarray(infty_losses))
    # Plot runtime over accuracy
    plotting.plot_fno_runtime_vs_accuracy(np.asarray(N_maxs), np.asarray(mse_losses))

    # Plot runtime over accuracy
    return y_true_np[example], y_pred_np[example]

