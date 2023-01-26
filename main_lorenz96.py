import argparse
import numpy as np 
import wandb
import torch

#import pce_pinns.deepxde.deepxde.geometry.timedomain as timedomain # Define grid
#os.environ['DDEBACKEND'] = 'pytorch'

import pce_pinns.utils.plotting as plotting
from pce_pinns.utils.utils import store_interim_data
from pce_pinns.utils.utils import load_interim_data
import pce_pinns.utils.logging as logging
from pce_pinns.utils.config import load_config
from pce_pinns.utils.config import update_dynamic_cfg_lorenz96

from pce_pinns.solver.lorenz96 import Lorenz96Eq, reshape_lorenz96_to_nn_model
from pce_pinns.solver.sampler import sample_diffeq, sample_model
from pce_pinns.models.fcnn.neural_net import interpolate_param_nn
from pce_pinns.models.mno.mno import interpolate_param_mno
from pce_pinns.models.poly import interpolate_param_poly
from pce_pinns.models.climatology import interpolate_param_clim
from pce_pinns.grid.utils import get_uniform_tgrid, update_tgrid_w_warmup

def gen_prior_solution(param, model, model_args, args, config):
    """
    Generates a dataset of lorenz96 solutions
    """
    lorenz96Eq_prior = Lorenz96Eq(model_args["xgrid"], param=param, 
        random_ic=config['de']['random_ic'], plot=(args.no_plot==False), 
        seed=config['de']['seed'])
    model_args['diffeq'] = lorenz96Eq_prior
    load_data_path = args.load_data_path
    if args.load_data_path is not None:
        load_data_path = args.load_data_path + '_null'
    logs = sample_model(model=model, model_args=model_args, 
        n_tasks=config['de']['n_tasks'], run_parallel=args.parallel,
        load_data_path=load_data_path, store_data_path='data/raw/temp/lorenz96/lorenz96_null')
    u_args, u_target, rand_insts = logging.convert_log_dict_to_np_lorenz96(logs)
    return u_args, u_target, rand_insts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='lorenz96')
    # Differential equation
    parser.add_argument('--mode_target', default='no-scale-sep', type=str,
        help='Name of solution variable that shall be estimated by neural net, e.g., '\
        '"X"                for X_{0:K}(t)          = NN(t)), '\
        '"Y"                for Y_{0:J,0:K}(t)      = NN(X(t), t), '\
        '"Z",'\
        '"Xres"             for X_{0:K}(0:T)        = NN(X_{0:K}(0:T;h=0)); not implemented'
        '"no-scale-sep"     for Y_{0:J,0:K}(t+1)    = NN(X_{0:K}(t), Y_{0:J,0:K}(t)), '\
        '"superparam"       for Y_{0:J,k}(t)        = NN(X_k(t-1), Y_{0:J,k}(t-1))'\
        '"no-scale-sep-corr"for Y_{0:J,0:K}(t+1)    = NN(X_{0:K}(t;h=0), Y_{0:J,0:K}(t))' \
        '"param-no-mem"     for f_k(t+1)            = NN(X_k(t))'\
        '                                           = hc/b sum_{j=0}^J Y_{j,k}(t+1)'\
        '"no-scale-sep-param-no-mem"'\
        '                   for f_{0:K}(t+1)        = NN(X_{0:K}(t))'\
        )
    parser.add_argument('--model_type', default='mno', type=str,
        help='Name of model that will be used')
    parser.add_argument('--n_samples', default=50, type=int,
            help='Number of samples in forward and inverse solution.')
    # NN approximation
    parser.add_argument('--est_param_nn', default='u', type=str,
            help='Name of parameter that shall be estimated by neural net, e.g., "\
            "pce_coefs", "k", "k_eigvecs", "k_true", "u", "u_true"')
    parser.add_argument('--load_data_path', default=None, type=str,
            help='Path to logged simulation data, e.g., data/raw/temp/lorenz96/lorenz96.')
    parser.add_argument('--load_interim_data_path', default=None, type=str,
            help='Path to logged interim data, e.g., data/interim/temp/lorenz96/lorenz96.')
    parser.add_argument('--store_interim_data_path', default='data/interim/temp/lorenz96', type=str,
            help='Path to store interim data')
    parser.add_argument('--eval_model_digest', default=None, type=str,
            help='file specifier of stored model and config data, e.g., "fcnn_l03u0512e003lr0.00100n000010"')
    # General
    parser.add_argument('--parallel', action="store_true",
            help='Enable parallel processing.')
    parser.add_argument('--seed', default=1, type=int,
            help='Random seed')
    parser.add_argument('--debug', action="store_true",
            help='Query low runtime debug run, e.g., reduce dimensionality, number of samples, etc.')
    parser.add_argument('--no_plot', action="store_true",
            help='Deactivate plotting of results')
    parser.add_argument('--no_wandb', action="store_true",
        help='Disables wandb')
    parser.add_argument('--no_test', action='store_true',
        help='Disable test run during model training')
    parser.add_argument('--overwrite', action='store_true',
        help='Overwrites trained models and config files')
    parser.add_argument('--sweep', action='store_true',
        help='Run wandb hyperparameter sweep. Requires online access on all nodes.')

    args = parser.parse_args()
    if args.parallel:
        assert args.no_plot==True, "Run model without parallel flag if plots are desired"
    np.random.seed(args.seed)
    
    # Init config and logger
    experiment_name = 'default'
    config_default = load_config(args, experiment_name=experiment_name,
        diffeq_name='lorenz96', update_dynamic_cfg=update_dynamic_cfg_lorenz96,
        sweep=args.sweep)
    
    def _main(config_default, sweep_config=None, args=None):
        if args.no_wandb:
            config = config_default
        else:
            from pce_pinns.utils.utils import sync_nested_wandb_config 
            _, config = sync_nested_wandb_config(
                sweep_config, config_default, verbose=True)
            
        # Update variables if solver has warmup 
        print('warmup:', config['de']['warmup'])
        tgrid_warmup=None
        if not args.load_data_path and not args.load_interim_data_path:
            tgrid_warmup, config['de']['n_snippets'], config['de']['n_tsnippet'], config['de']['tmax'], config['de']['n_tasks'] = update_tgrid_w_warmup(
                warmup=config['de']['warmup'],
                n_snippets=config['de']['n_samples'],
                tmax=config['de']['tmax'],
                dt=config['de']['dt'],
                parallel=args.parallel,
                dtype=config['de']['dtype'],
                )
        else:
            from pce_pinns.grid.utils import get_snippet_lengths
            config['de']['n_tsnippet'] = int(config['de']['tmax']/config['de']['dt'])
            config['de']['n_snippets'], _, config['de']['n_tasks'] = get_snippet_lengths(
                config['de']['n_samples'], config['de']['n_tsnippet'], 
                config['de']['dt'], args.parallel)

        # Define grid
        tgrid = get_uniform_tgrid(config['de']['tmax'], config['de']['dt'], config['de']['dtype'],
            load_data_path=args.load_data_path, store_data_path=config['dir_raw_data'],
            load_interim_data_path=args.load_interim_data_path, 
            store_interim_data_path=config['dir_interim_data'])

        # Init differential equation
        lorenz96Eq = Lorenz96Eq(tgrid, param=config['de']['param'], 
            y_param_mode = config['de']['y_param_mode'],
            random_ic=config['de']['random_ic'], 
            random_ic_y=config['de']['random_ic_y'], random_ic_z=config['de']['random_ic_z'], 
            K=config['de']['K'], tgrid_warmup=tgrid_warmup,
            plot=(args.no_plot==False), seed=config['de']['seed'],
            dtype=config['de']['dtype'],
            use_rk4=config['de']['use_rk4'])

        # Init surrogate model of differential equation
        model = sample_diffeq
        model_args = {'diffeq':lorenz96Eq, 'xgrid': tgrid,
            'kl_dim': None, 'pce_dim': None}

        # Estimate various parameters with a neural network
        if args.est_param_nn == 'u':
            torch.manual_seed(config['de']['seed'])

            # Generate interim dataset
            if args.load_interim_data_path:
                u_args, u_target, rand_insts, u_prior, _ = load_interim_data(
                    args.load_interim_data_path, verbose=True)
            else:
                logs = sample_model(model=model, model_args=model_args, 
                    n_tasks=config['de']['n_tasks'], run_parallel=args.parallel,
                    load_data_path=args.load_data_path, store_data_path=config['dir_raw_data'])
                u_args, u_target, rand_insts = logging.convert_log_dict_to_np_lorenz96(logs)

                # Generate another set of lorenz96 solutions with null parametrization
                u_prior = None
                if args.mode_target == "no-scale-sep-corr":
                    _, u_prior, _ = gen_prior_solution(param='null', model=model, 
                                    model_args=model_args.copy(), args=args, config=config)
                    assert config['data_loader']['test_mode'] == 'batch'
                store_interim_data(config['dir_interim_data'], u_args=u_args, 
                    u_target=u_target, rand_insts=rand_insts, u_prior=u_prior)
        
            # Generate ML-ready dataset: Arrange input dimensions to fit NN inputs
            u_target, grid, y_args = reshape_lorenz96_to_nn_model(sol=u_target,
                tgrid=tgrid, est_lorenz=args.mode_target, 
                model_cfg=config['model'], sol_prior=u_prior, 
                u_args=u_args, n_snippets=config['de']['n_snippets'],
                n_tsnippet=config['de']['n_tsnippet'])

            # Test parametrization
            # lorenz96Eq.test_full_subgrid_forcing(x=y_args[0,:,:], y_param=u_target[0,:,:], tgrid=tgrid)
            # Select grid input dims:
            grid_in_dims = () # (-1,) # (-1,) for all grid as input
            if config['model']['type'] == 'fcnn' and not args.mode_target=='param-no-mem':
                plotting.plot_lorenz96_avg(grid[0,:,0], u_target[:,:,0])
            elif config['model']['type'] == 'mno' or args.mode_target=='param_no_mem':
                if config['model']['model_dims'] == 1:
                    plotting.plot_lorenz96_avg(grid[0,:,0], u_target[:,:,0,0], title='avg_y', ylabel='Y_{0,0}')
                    plotting.plot_lorenz96_avg(grid[0,:,0], y_args[:,:,0,0], title='avg_x', ylabel='X_0')
            elif config['model']['type'] == 'mno':
                if config['model']['model_dims'] == 2:
                    plotting.plot_lorenz96_avg(grid[0,:,0], u_target[:,:,0,0,0])

            # Use NN
            if args.est_param_nn=='u': # or args.est_param_nn=='u_true':
                if config['model']['type'] == 'fcnn':
                    u, u_pred = interpolate_param_nn(grid=grid, 
                        y=u_target, y_args=y_args, 
                        n_epochs=config['n_epochs'], 
                        n_layers=config['model']['n_layers'], 
                        n_units=config['model']['n_units'], 
                        lr=config['optimizer']['lr'],
                        target_name=args.est_param_nn,
                        alpha_indices=None, rand_insts=None,
                        n_test_samples=args.n_samples,
                        diffeq=lorenz96Eq, loss_name='mseloss', 
                        normalize=config['data_loader']['normalize'],
                        u_params=None, grid_in_dims=grid_in_dims,
                        test_mode=config['data_loader']['test_mode'],
                        plot=(args.no_plot==False), debug=args.debug,
                        config=config,
                        eval_model_digest=args.eval_model_digest, 
                        overwrite=args.overwrite,
                        run_parallel=args.parallel)
                elif 'mno' in config['model']['type']:
                    u, u_pred = interpolate_param_mno(grid=grid, 
                        y=u_target,y_args=y_args,
                        n_layers=config['model']['depth'], 
                        n_units=config['model']['n_modes'], 
                        lr=config['optimizer']['lr'],
                        target_name=args.est_param_nn,
                        alpha_indices=None, rand_insts=rand_insts,
                        n_test_samples=args.n_samples,
                        diffeq=lorenz96Eq, loss_name='mseloss', 
                        normalize=config['data_loader']['normalize'],
                        u_params=None, grid_in_dims=grid_in_dims,
                        plot=(args.no_plot==False), debug=args.debug,
                        config=config,
                        eval_model_digest=args.eval_model_digest,
                        run_parallel=args.parallel, no_test=args.no_test)
                elif config['model']['type'] == 'poly':
                    u, u_pred = interpolate_param_poly(grid=grid,
                        y=u_target, y_args=y_args,
                        diffeq=lorenz96Eq,
                        plot=(args.no_plot==False),
                        config=config)
                elif config['model']['type'] == 'climatology':
                    u, u_pred = interpolate_param_clim(grid=grid,
                        y=u_target, y_args=y_args, 
                        plot=(args.no_plot==False),
                        config=config) 
            else:
                raise NotImplementedError
             
            # plotting.plot_nn_lorenz96_solx(tgrid, u_target[0], u_pred[0])

            if not args.no_wandb:
                wandb.log({'runtime/tstep PDE.solve': lorenz96Eq.cumulative_runtime/float(config['de']['n_samples'])})

    if args.no_wandb:
        # wandb.init(mode='disabled')
        _main(config_default, args=args)
    else:
        import yaml
        # sweep_config = yaml.load(open(f'models/{experiment_name}/lorenz96/{args.model_type}/sweep.yaml'), Loader=yaml.FullLoader)
        sweep_config = yaml.load(open('models/%s/lorenz96/%s/sweep.yaml'.format(experiment_name, args.model_type)), Loader=yaml.FullLoader)
        def run_sweep():
            with wandb.init() as run:
                _main(config_default, sweep_config=run.config, args=args)
        sweep_id = wandb.sweep(sweep_config,
            project='mno-lorenz96-sweep-n64', entity='blutjens')
        wandb.agent(sweep_id, run_sweep, count=1000)
    print('Finished main_lorenz96.py')
