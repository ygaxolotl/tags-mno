import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from pce_pinns.utils.utils import pickle_dump

class Climatology(object):
    def __init__(self):
        self.y_shape = None

    # Calculate climatology on training set
    def train(self, y_train):
        self.y_shape = y_train.shape
        self.clim_sol = np.mean(y_train, axis=(0,1))

    # Predict
    def predict(self, n_samples):
        y_pred = np.repeat(self.clim_sol[None, ...], repeats=self.y_shape[1], axis=0) # share over time. 
        y_pred = np.repeat(y_pred[None, ...], repeats=n_samples, axis=0) # share over val samples

        return y_pred

    def get_specifier(self, config):
        return f'climatology_n{config["de"]["n_samples"]}'

def interpolate_param_clim(grid, y, y_args=None,
    plot=True, 
    config=None):
    """
    Calculates climatology
    # Y_pred = mean(Y_train)
    \hat X_k = 1/T \sum_{t=0}^T 1/N \sum_{i=0}^N X_{k,i}(t)
    Args:
        y np.array(n_samples, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim)
        y_args None
        grid np.array(n_samples, n_tgrid, n_x1grid, ..., n_xngrid)
    Returns:
        sol_true np.array(n_samples_val, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim)
        sol_pred np.array(n_samples_val, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim)
    """
    val_size = config['data_loader']['val_size']

    # Prepare dataset
    if val_size > 0:
        y_train, y_val = train_test_split(y, test_size=val_size, shuffle=False, random_state=config['de']['seed']+1)

    # Init, train, and predict model
    climatology = Climatology()
    print('Training climatology')
    climatology.train(y_train)
    print('Predicting climatology')
    y_pred = climatology.predict(n_samples=y_val.shape[0])

    # Dump predictions
    if config['dir_predictions'] is not None:
        predictions = {
            'config': config,
            'grid': grid,
            'sol_pred': y_pred,
            'sol_true': y_val,
            'y_param_pred': None,
            'y_param_true': None,
        }
        eval_model_digest = climatology.get_specifier(config)
        print('Saving predictions at', Path(config['dir_predictions'],eval_model_digest))
        pickle_dump(predictions, folder=config['dir_predictions'], name=eval_model_digest+'.pickle')

    # Evaluate
    if plot:
        from pce_pinns.models.mno import eval as eval_mno 
        from pce_pinns.utils.plotting import plot_lorenz96_rmse_t, print_lorenz96_rmse,plot_lorenz96_avg
        results_clim = eval_mno.eval_predictions(eval_model_digest,
                    dir_predictions=config['dir_predictions'],
                    dir_results=config['dir_results'])

        print_lorenz96_rmse(clim=results_clim['rmse'])
        plot_lorenz96_rmse_t(tgrid_clim=results_clim['grid'][0,...,0], 
            clim=results_clim['rmse_t'])
        plot_lorenz96_avg(
            tgrid_clim=results_clim['grid'][0,...,0], clim=results_clim['sol_pred'])

    return predictions['sol_true'], predictions['sol_pred']
