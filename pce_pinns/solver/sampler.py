"""
Functions to sample differential equation solver
Author 
"""
import numpy as np 
import pickle # For data storage

from pathlib import Path

import pce_pinns.utils.parallel as parallel

def sample_diffeq(diffeq, xgrid, 
    y_gp_mean=None, y_gp_cov=None, 
    kl_dim=None, expansion=None, z_candidate=None, pce_dim=None, 
    x_obs=None,
    sample_y=None,
    pce_coefs=None, deterministic=False, 
    solver_name=None):
    """Samples the differential equation

    Computes a sample of the differantial equation. The solution depends on the 
    stochastic, log-permeability, Y, given by sample_y. If sample_y is not given, we 
    assume it's modeled by a Gaussian process.

    Args: 
        diffeq pce_pinns.diffeq.DiffEq: Differential equation that shall be sampled, e.g., stochastic diffusion equation
        xgrid np.array(()): Evaluation grid
        y_gp_mean np.array(n_grid): Gaussian process mean as fn of x
        y_gp_cov np.array((n_grid,n_grid)): Gaussian process covariance matrix as fn of x
        kl_dim int: Number of non-truncated eigenvalues for Karhunen-Loeve expansion
        expansion
        z_candidate np.array(kl_dim): KL random variables
        pce_dim int: Maximum polynomial degree of PCE polynomial bases        
        x_obs np.array(n_msmts): Grid of observation points. If not None, only values at x_obs are returned.
        sample_y function()->y,exp_y,kl_trunc_err,coefs, rand_insts: 
            Function that draws samples of log-permeability, Y
        pce_coefs np.array(n_grid, n_alpha_indices): PCE coefficients
        solver_name bool: Name of numerical solver, if None default is used. 
    Returns:
        u_obs np.array(n_msmts): Solution, u, at measurement locations
        vals dict(
            'rand_insts': np.array(LocalAdvDiffEq.gp_stoch_dim)
            'rand_param': np.array(LocalAdvDiffEq.n_zgrid)
            'rand_insts_ic': np.array
            'rand_ic': np.array
            'u': See diffeq.solve; could be (np.array((diffeq.n_tgrid,) + diffeq.shape_xgrid + (diffeq.n_udim)))
        ): Dictionary of model results, e.g., Y, exp_Y, u, kl_trunc_err, coefs, rand_insts
        sample_y fn(): Function that samples log-permeability, y
    """
    vals = {}
    # Sample stochastic parameters
    vals['rand_insts'], vals['rand_param'] = diffeq.sample_random_params()
    vals['rand_insts_ic'], vals['rand_ic'] = diffeq.sample_random_ic()

    # Warmup solver
    if diffeq.tgrid_warmup is not None:
        # Warmup differential equation, e.g., to smooth out init conditions or find equilibrium/attractor 
        u_warmup = diffeq.solve(solver_name=solver_name, warmup=True, set_final_as_ic=True)

    # Compute solution
    vals['u'] = diffeq.solve(solver_name=solver_name)

    # Reduce solution to observed values
    if x_obs is None:
        u_obs = vals['u']
    else:
        # TODO: use pce_pinns.utils.utils.get_fn_at_x()
        obs_idxs = np.zeros((x_obs.shape[0]))
        for i, x_o in enumerate(x_obs):
            obs_idxs[i] = np.abs(xgrid - x_o).argmin().astype(int)
        obs_idxs = obs_idxs.astype(int)
        u_obs = vals['u'][obs_idxs]

    return u_obs, vals

def sample_model(model=sample_diffeq, model_args={}, 
    n_tasks=2, 
    run_parallel=False,
    load_data_path=None, store_data_path='data/temp'):
    """
    Returns n_tasks samples of the model
    
    Args:
        model fn:**model_args->u_obs, logs, sample_y: Continuous function returning model evaluations, u.
        model_args dict(): Dictionary of arguments to model() 
        run_parallel bool: If true, sample the model in parallel; TODO: implement and test
        load_data_path string: Path to stored n_tasks model samples
        store_data_path string: Path to store model samples
    
    Returns:
        logs n_tasks*[]: n_tasks samples of the model
    """
    # Load simulation data instead of 
    if load_data_path is not None:
        with open(load_data_path+'.pickle', 'rb') as handle:
            logs = pickle.load(handle)
    else:
        print('Generating '+str(n_tasks) + ' target model samples')
        logs = n_tasks*[None]
        # Initialize stochastic approximations to quickly generate model samples, e.g., PCE of the solution
        # _, _ = model(**model_args) # TODO: Do I really need this one warmup run? - I think yes for KLE and PCE
        
        model, model_tasks = parallel.init_preprocessing(fn=model, parallel=run_parallel)
        
        # Generate samples
        for i in range(n_tasks):
            if (i-1)%100==0: print('i', i)
            # Sample solution
            if not run_parallel:
                _, logs[i] = model(**model_args)
            else:
                model_tasks.append(model(**model_args))
                # Update random seed in model to avoid shared seed in parallel processes.
                try:
                    model_args['diffeq'].step_rng()
                except:
                    pass

        # Parse parallel tasks
        if run_parallel:
            model_tasks = parallel.get_parallel_fn(model_tasks)
            for i in range(n_tasks):
                _, logs[i] = model_tasks[i]
        
        # Store data
        print('Saving raw data. Load with --load_data_path \"'+store_data_path+'\"')
        folder = Path(*Path(store_data_path).parts[:-1])
        if not folder.exists():
            folder.mkdir(parents=True)
        with open(store_data_path+'.pickle', 'wb') as handle:
            pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return logs 
