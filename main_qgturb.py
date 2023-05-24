import argparse
import numpy as np
import wandb
import h5py

import torch

from pce_pinns.utils.utils import store_interim_data
from pce_pinns.utils.utils import load_interim_data

from pce_pinns.utils.config import load_config
from pce_pinns.solver.qgturb import QgturbEq, reshape_qgturb_to_nn_model
from pce_pinns.models.mno.mno import interpolate_param_mno

import torchqg.workflow as workflow

def load_raw_data(load_data_path, system_name='\\mathcal{F}'):
    """
    Loads raw quasi-geostrophic turbulence data
    Returns:
        ft np.array([steps], dtype=np.float32): Temporal grid
        fr np.array([steps, Nyl, Nyx, 1], dtype=cfg.dtype): Ground-truth Parametrization
        fq np.array([steps, Nyl, Nyx, 1], dtype=cfg.dtype): Vorticity from filtered DNS
    """
    filename = load_data_path + '.h5'
    hf = h5py.File(filename, 'r')

    ft = hf.get('time')
    fr = hf.get(system_name + '_r')
    fq = hf.get(system_name + '_q')

    ft = ft[:]
    fr = fr[:][..., np.newaxis]
    fq = fq[:][..., np.newaxis]
    
    return ft, fr, fq

if __name__ == "__main__":
    # Quasi-Geostrophic turbulence equation
    parser = argparse.ArgumentParser(description='qgturb')
    # Differential equation
    parser.add_argument('--mode_target', default='no-scale-sep-param-no-mem', type=str,
        help='Name of solution variable that shall be estimated by neural net, e.g., '\
        '"no-scale-sep-param-no-mem"'\
        '                   for R_{0:Nyl,0:Nxl}(t+1)        = NN(Q_{0:Nyl,0:Nxl}(t))'\
        '"pure-ml-sol"'\
        '                   for Q_{0:Nyl,0:Nxl}(t+1)        = NN(Q_{0:Nyl,0:Nxl}(t))'
        )
    parser.add_argument('--model_type', default='mno', type=str,
        help='Name of model that will be used, "mno", "clim", "poly", "fcnn",'\
        '"fno-pure-ml-sol".')
    parser.add_argument('--n_samples', default=50, type=int,
            help='Number of samples in forward and inverse solution.')
    parser.add_argument('--load_data_path', default=None, type=str,
            help='Path to raw logged simulation data, e.g., data/raw/temp/qgturb/qgturb.')
    parser.add_argument('--load_interim_data_path', default=None, type=str,
            help='Path to interim ML-ready data, e.g., data/interim/temp/qgturb/qgturb.')
    parser.add_argument('--store_interim_data_path', default=None, type=str,
            help='Path to store interim ML-ready data, e.g., data/interim/temp/qgturb/qgturb_warmed_100')
    parser.add_argument('--load_processed_data_path', default=None, type=str,
            help='Path to load processed MNO-ready data, e.g., data/processed/temp/qgturb/mno/n5_t20')
    parser.add_argument('--eval_model_digest', default=None, type=str,
            help='file specifier of stored model and config data, e.g., "fcnn_l03u0512e003lr0.00100n000010"')
    parser.add_argument('--parallel', action="store_true",
            help='Enable parallel processing')
    parser.add_argument('--seed', default=1, type=int,
            help='Random seed')
    parser.add_argument('--no_plot', action="store_true",
            help='Deactivate plotting of results')
    parser.add_argument('--no_wandb', action="store_true",
        help='Disables wandb')
    #parser.add_argument('--no_test', action='store_true',
    #    help='Disable test run during model training')
    parser.add_argument('--sweep', action='store_true',
        help='Run wandb hyperparameter sweep. Requires online access on all nodes.')
    args = parser.parse_args()

    if args.parallel or args.sweep:
        raise NotImplementedError('Argument not supported for QG Turb')

    if args.parallel:
        assert args.no_plot==True, "Run model without parallel flag if plots are desired"
    np.random.seed(args.seed)

    # Init config and logger
    experiment_name = 'default'
    config_default = load_config(args, experiment_name=experiment_name,
        diffeq_name='qgturb', sweep=args.sweep)

    # Init grid and differential equation
    qgturbEq = QgturbEq()
    # Generate raw dataset
    # target: SGS parametrization: 
    #   FDNS -- torch.Tensor([steps, n_dims, Nyl, Nyx]), n_dims = 5, r q p u v in physical space
    #   DNS = Parametrization(DNS) + Filter(DNS) 
    # input: vorticity(t-1, x, y)

    ##################
    # Get interim ML-ready dataset
    ##################
    if args.eval_model_digest is not None:
        # Test works on raw data        
        pass
    elif args.load_data_path:
        assert args.load_interim_data_path is None, "Multiple data load paths. Only pass one."
        assert args.load_processed_data_path is None, "Multiple data load paths. Only pass one."
        # Load raw dataset
        tgrid, u_target, u_args = load_raw_data(args.load_data_path)

        # Convert raw to interim ML-ready dataset
        u_target, grid, y_args = reshape_qgturb_to_nn_model(sol=u_target, 
            tgrid=tgrid, u_args=u_args, 
            n_tsnippet=config_default['de']['n_tsnippet'], 
            est_qgturb=args.mode_target)
        config_default['de']['n_snippets'] = u_target.shape[0] 
 
        if args.store_interim_data_path:
            store_interim_data(args.store_interim_data_path, u_args=y_args, 
                u_target=u_target, grid=grid)

        # Delete raw data from memory
        del tgrid
        del u_args        
    elif args.load_interim_data_path:
        assert args.load_processed_data_path is None, "Multiple data load paths. Only pass one."
        # Load interim dataset
        y_args, u_target, _, _, grid  = load_interim_data(args.load_interim_data_path)
    elif args.load_processed_data_path:
        u_target = None
        grid = None
        y_args = None
    else:
        raise NotImplementedError("Need to pass path to load data.")

    """
    # Todo: Test if ML-ready dataset can be converted back into running the 
    # torchqg.sgs.predict(), such that -- filtered DNS = zero-param + param
    rs_tst = u_target.reshape((-1,)+u_target.shape[2:-1])
    fdns_tst = y_args.reshape((-1,)+y_args.shape[2:])
    time_tst = tgrid
    sgs_tst = Testparam(rs_tst=rs_tst, fdns_tst=fdns_tst, time_tst=time_tst)
    qgturbEq.init_les(sgs=sgs_tst)
    qgturbEq.solve()
    """
    # Train FNO2D
    if args.eval_model_digest is None:
      u, u_pred = interpolate_param_mno(grid=grid, 
        y=u_target, y_args=y_args,
        n_layers=config_default['model']['depth'], 
        n_units=config_default['model']['n_modes'], 
        lr=config_default['optimizer']['lr'],
        diffeq=qgturbEq,
        normalize=config_default['data_loader']['normalize'],
        grid_in_dims=(), # no grid information as input
        plot=(args.no_plot==False),
        config=config_default,
        eval_model_digest=args.eval_model_digest,
        run_parallel=args.parallel, no_test=True,
        load_processed_data_path=args.load_processed_data_path)# args.no_test)

    ###
    # Evaluation 
    ###
    from pathlib import Path
    from pce_pinns.models.mno.config import get_trained_model_cfg
    from pce_pinns.models.fno import fno2d_gym 
    from torchqg.sgs import MNOparam
    
    ## Load model
    def load_model(eval_model_digest, dir_out):
        # dir_out = config_default['dir_out']
        dir_out = Path(dir_out)
        cfg = get_trained_model_cfg(eval_model_digest, dir_out)
        model_load = fno2d_gym.make_model(cfg["model"])
        model_load.load_state_dict(torch.load(str(dir_out / "{}_modstat.pt".format(eval_model_digest))))
        return model_load

    eval_model_names = ['mno', 'fno-pure-ml-sol']
    ## Initialize coupled diffeq
    # Currently validating on continued last step of training data.
    # todo: replace args.load_data_path with validation data.
    # e.g., provide qgturb/main.py init_data_path = 'output/spinup_10000_dump' from a different random seed spinup. OR from the END of the training data.
    # Currently evaluating on last steps AFTER training
    qgturbEq = QgturbEq(init_data_path=args.load_data_path, init_tstep=-1)

    if 'mno' in eval_model_names:
        # Add model with MNO parametrization
        dir_out = Path('models/temp/qgturb/mno')
        eval_mno_digest = '03a25874ca' # b4e8334aa9'
        print(f'Please double-check if eval_mno_digest for evaluation was updated: {eval_mno_digest}')
        model_mno = load_model(eval_mno_digest, dir_out)
        sgs_mno = MNOparam(model=model_mno)
        qgturbEq.init_les(sgs=sgs_mno, name='mno')
    if 'fno-pure-ml-sol' in eval_model_names:
        # Add pure ML model
        dir_out = Path('models/temp/qgturb/fno-pure-ml-sol')
        model_pure_ml_sol = load_model(args.eval_model_digest, dir_out)
        qgturbEq.init_pure_ml(model=model_pure_ml_sol, name='pure-ml-sol')

    ## Run coupled model and plot workflow.diagnoses
    diagnostics = [workflow.diag_pred_vs_target_sol_err,
                workflow.diag_pred_vs_target_param,
                workflow.diag_fields,]
    qgturbEq.solve(dir=dir_out / 'predictions', name=args.eval_model_digest, 
        iters=2, store_iters=2, diags=diagnostics)

    # Evaluate
    ## RMSE over time.
    ## 
