import time
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from pce_pinns.utils.utils import pickle_dump

class Poly(object):
    def __init__(self, dtype='float64'):
        self.A = None # weights
        self.dtype = dtype
    # Calculate weights on training set
    def train(self, features, target):
        """
        Args:
            features
        """
        self.dtype = features.dtype.name
        features = np.concatenate((features, np.ones(features.shape, 
            dtype=self.dtype)), axis=-1)
        self.A = np.dot(np.matmul(la.inv(np.matmul(features.T, features)), features.T), target)
        return 1

    def predict(self, y_args_val, 
        diffeq, grid, n_samples, val_size, 
        n_tgrid, K):
        """
        Predicts all parametrizations and integrates coupled model forward in time
        """
        # Predict parametrizations
        y_args_val = np.concatenate((y_args_val, 
            np.ones(y_args_val.shape, dtype=self.dtype)), axis=-1)
        y_param_pred = np.matmul(y_args_val, self.A)
        # create_scatter_plot(y_args_val, y_param_true, y_param_pred):
        
        # Unmerge space-channel, K, then time-channel
        y_args_val = y_args_val[:,:1].reshape(int(n_samples*val_size*n_tgrid),K)
        y_args_val = y_args_val.reshape(int(n_samples*val_size), n_tgrid,K)
        y_param_pred = y_param_pred.reshape(int(n_samples*n_tgrid*val_size),K)
        y_param_pred = y_param_pred.reshape(int(n_samples*val_size),n_tgrid,K)
        grid = grid[::K]
        grid = grid.reshape(n_samples, n_tgrid)
        
        # Predict coupled low-res. variable
        sol_pred = np.zeros(y_args_val.shape, dtype=self.dtype)
        for i in tqdm(range(int(n_samples*val_size))):
            sol_pred[i,:,:] = diffeq.test_full_subgrid_forcing(
                x_target=y_args_val[i,:,:], y_param=y_param_pred[i,:,:], 
                tgrid=grid[i,:], plot=False)

        return sol_pred, y_param_pred

    def get_specifier(self, config):
        """
        Returns:
            specifier string: Terse specifier of config
        """
        specifier = (f'poly_n{config["de"]["n_samples"]}')
        return specifier

def create_scatter_plot(y_args_val, y_param_true, y_param_pred):
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.scatter(y_args_val[:,:1], y_param_true, color='blue', alpha=0.5, label='Ground-Truth, $X_k$')
    axs.plot(y_args_val[:,:1], y_param_pred, color='orange', label='Predicted, $X_k$')
    axs.set_xlabel(r'Low-res., $X_k$')
    axs.set_ylabel(r'Parametrization, $f(X_k)$')
    axs.legend()
    fig.savefig('doc/figures/lorenz96/x_vs_y_scatter.png')
    plt.close(fig)

def interpolate_param_poly(grid, y, y_args=None,
    diffeq=None,
    plot=False, 
    config=None):
    """
    Args:
        see interpolate_param_nn
    Returns:
        sol_true np.array(n_samples_val, n_xgrid1, ..., n_xgridN, n_xdim): Ground-truth solution
        sol_pred np.array(n_samples_val, n_xgrid1, ..., n_xgridN, n_xdim): Predicted solution
        y_param_true np.array(n_samples_val, n_xgrid1, ..., n_xgridN, n_xdim): Ground-truth y parametrization
        y_param_pred np.array(n_samples_val, n_xgrid1, ..., n_xgridN, n_xdim): Predicted y parametrization
    """
    # Y = ax + b
    # y = A[x+0]
    n_samples = y.shape[0]
    n_tgrid = grid.shape[1]
    K = y_args.shape[-1]
    val_size = config['data_loader']['val_size']
    
    # Merge time- then K-channel into different samples
    y_args = y_args.reshape(n_samples*n_tgrid,K)
    y_args = y_args.reshape(n_samples*n_tgrid*K,1)
    y = y.reshape(n_samples*n_tgrid,K)
    y = y.reshape(n_samples*n_tgrid*K,1)
    grid_poly = np.copy(grid).reshape(n_samples*n_tgrid,1)
    grid_poly = np.repeat(grid_poly, repeats=K, axis=0)

    # Train test split
    if val_size > 0:
        sol_train, sol_true, y_param, y_param_true = train_test_split(y_args, y, 
            test_size=val_size, shuffle=False, random_state=config['de']['seed']+1)

    poly = Poly()
    poly.train(features=sol_train, target=y_param)
    
    # Predict
    sol_pred, y_param_pred = poly.predict(y_args_val=sol_true, diffeq=diffeq,
        grid=grid_poly, n_samples=n_samples, val_size=val_size, 
        n_tgrid=n_tgrid, K=K)
    # Dump model
    eval_model_digest = poly.get_specifier(config)
    print('Specifier: ', eval_model_digest)

    # Unmerge space and time channel
    sol_true = sol_true[:,:1].reshape(int(n_samples*val_size*n_tgrid),K)
    sol_true = sol_true.reshape(int(n_samples*val_size), n_tgrid,K)
    y_param_true = y_param_true.reshape(int(n_samples*n_tgrid*val_size),K)
    y_param_true = y_param_true.reshape(int(n_samples*val_size),n_tgrid,K)

    # Add xdim
    sol_pred = sol_pred[...,None]
    sol_true = sol_true[...,None]
    y_param_pred = y_param_pred[...,None]
    y_param_true = y_param_true[...,None]

    if config['dir_predictions'] is not None:
        # Dump predictions
        predictions = {
            'config': config,
            'grid': grid[:1, ...],
            'sol_pred': sol_pred,
            'sol_true': sol_true,
            'y_param_pred': y_param_pred,
            'y_param_true': y_param_true,
        }
        eval_model_digest = poly.get_specifier(config)
        print('Saving predictions at', Path(
            config['dir_predictions'],eval_model_digest))
        pickle_dump(predictions, 
            folder=config['dir_predictions'], 
            name=eval_model_digest+'.pickle')

    # Plot errx, erry, solx, soly
    # Measure MAE and MSE
    # !!TODO!!: do I have to shift by dt=1?
    from pce_pinns.models.mno import eval as eval_mno
    max_t = np.min((sol_true.shape[1], 200))
    rmse = eval_mno.calculate_rmse(sol_true[:,:max_t,...], 
        sol_pred[:,:max_t,...])
    print('RMSE: ', rmse)

    # Measures time to predict Y_params, averaged across n_samples and time.
    msr_runtime = False
    if msr_runtime:
        m_samples = 1000
        times = []
        for _ in range(m_samples):
            start = time.time()
            _ = np.matmul(sol_train, self.A)
            times.append(time.time()-start)
        runtime = np.mean(np.asarray(times)) / float(sol_train.shape[0])
        print('Runtime: {:.20f}s'.format(runtime))
    return y_param_pred, sol_pred
