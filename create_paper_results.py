from pathlib import Path
import argparse
import numpy as np

from pce_pinns.solver.lorenz96 import load as loadLorenz96
from pce_pinns.solver.lorenz96 import measure_runtime_lorenz96
from pce_pinns.models.mno import eval as eval_mno 
from pce_pinns.models.mno import mno
from pce_pinns.utils.plotting import plot_lorenz96_runtimes
from pce_pinns.utils.plotting import plot_lorenz96_rmse_t
from pce_pinns.utils.plotting import print_lorenz96_rmse
from pce_pinns.utils.plotting import print_lorenz96_mse
from pce_pinns.utils.plotting import plot_lorenz96_predictions_avg
from pce_pinns.utils.plotting import plot_lorenz96_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='paper-results')
    parser.add_argument('--eval_runtime', action="store_true",
            help='Creates log-log plot of runtime vs. grid size')
    parser.add_argument('--eval_results', action="store_true",
            help='Creates results plot')
    parser.add_argument('--eval_mno_digest', default=None, 
            type=str, help='specifier of stored model and '\
            'config data, e.g., "a6a5acd859",')
    parser.add_argument('--eval_fcnn_digest', default=None, 
            type=str, help="e.g., fcnn_l02u0032e020lr01000n005000k4")
    parser.add_argument('--eval_clim_digest', default=None,  
            type=str, help='e.g., climatology_n5000')
    parser.add_argument('--eval_poly_digest', default=None,  
            type=str, help='e.g., poly_n5000')
    parser.add_argument('--dir_out', default="model/neurips23/lorenz96/mno", type=str,
            help='directory of stored models')

    args = parser.parse_args()

    # Create log-log plot runtime vs. grid size 
    if args.eval_runtime:
        runtimes_lorenz96, Ns_lorenz96 = measure_runtime_lorenz96()
        runtimes_mno, Ns_mno = eval_mno.measure_runtime_mno(
            eval_model_digest=args.eval_model_digest, 
            dir_out=Path('model/icml22/lorenz96/mno/'), coupled=True)
        plot_lorenz96y_runtimes(Ns_mno=Ns_mno, ts_mno=runtimes_mno,
            Ns_fully_resolved=Ns_lorenz96, ts_fully_resolved=runtimes_lorenz96,
            fname='runtimes')

    # Create results plot: RMSE over time
    if args.eval_results:
        # Create raw data
        # Train 
        # Run test

        # Evaluate
        results_None = {'grid': np.zeros((1,1,1)), 
            'sol_true': None, 'sol_pred': None, 
            'rmse': None, 'mse': None, 'rmse_t': None}
        results_mno = results_None
        results_fcnn = results_None
        results_clim = results_None
        results_poly = results_None

        if args.eval_mno_digest:
            results_mno = eval_mno.eval_predictions(args.eval_mno_digest, 
                dir_predictions='models/temp/lorenz96/mno/predictions/',
                dir_results='models/temp/lorenz96/mno/results/')
        if args.eval_fcnn_digest:
            results_fcnn = eval_mno.eval_predictions(args.eval_fcnn_digest,
                dir_predictions='models/temp/lorenz96/fcnn/predictions/',
                dir_results='models/temp/lorenz96/fcnn/results/')
            results_fcnn['grid']=results_fcnn['grid'][...,0]
        if args.eval_clim_digest:
            results_clim = eval_mno.eval_predictions(args.eval_clim_digest, 
                dir_predictions='models/temp/lorenz96/clim/predictions/',
                dir_results='models/temp/lorenz96/clim/results/')
        if args.eval_poly_digest:
            results_poly = eval_mno.eval_predictions(args.eval_poly_digest, 
                dir_predictions='models/temp/lorenz96/poly/predictions/',
                dir_results='models/temp/lorenz96/poly/results/')

        plot_lorenz96_samples(
            tgrid_mno=results_mno['grid'][0,...,0], mno_true=results_mno['sol_true'], mno_pred=results_mno['sol_pred'],
            tgrid_fcnn=results_fcnn['grid'][0,...,0], fcnn_true=results_fcnn['sol_true'], fcnn_pred=results_fcnn['sol_pred'],
            tgrid_clim=results_clim['grid'][0,...,0], clim_true=results_clim['sol_true'], clim_pred=results_clim['sol_pred'],
            tgrid_poly=results_poly['grid'][0,...,0], poly_true=results_poly['sol_true'], poly_pred=results_poly['sol_pred'],
            )
        plot_lorenz96_predictions_avg(tgrid_true=results_mno['grid'][0,...,0], sol_true=results_mno['sol_true'],
            tgrid_mno=results_mno['grid'][0,...,0], mno=results_mno['sol_pred'], 
            tgrid_fcnn=results_fcnn['grid'][0,...,0], fcnn=results_fcnn['sol_pred'],
            tgrid_clim=results_clim['grid'][0,...,0], clim=results_clim['sol_pred'],
            tgrid_poly=results_poly['grid'][0,...,0], poly=results_poly['sol_pred'],
            )
        print_lorenz96_rmse(mno=results_mno['rmse'], 
            fcnn=results_fcnn['rmse'], 
            clim=results_clim['rmse'],
            poly=results_poly['rmse'])
        print_lorenz96_mse(mno=results_mno['mse'], 
            fcnn=results_fcnn['mse'],
            clim=results_clim['mse'],
            poly=results_poly['mse'])
        plot_lorenz96_rmse_t(tgrid_mno=results_mno['grid'][0,...,0], mno=results_mno['rmse_t'], 
            tgrid_fcnn=results_fcnn['grid'][0,...,0], fcnn=results_fcnn['rmse_t'],
            tgrid_clim=results_clim['grid'][0,...,0], clim=results_clim['rmse_t'],
            tgrid_poly=results_poly['grid'][0,...,0], poly=results_poly['rmse_t'],
            )
