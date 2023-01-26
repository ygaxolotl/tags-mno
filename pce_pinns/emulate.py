"""
Create emulator for different parts of a differential equation
Author: 
"""
import numpy as np 

import pce_pinns.utils.plotting as plotting
import pce_pinns.utils.logging as logging
import pce_pinns.utils.densities as densities

import pce_pinns.rom.pce as pce

def approx_sol_w_nn(grid, diffeq, u_target, y_args, u_true, u_params, grid_in_dims,
    est_param_nn, loss_name, normalize, kl_dim, pce_dim, rand_insts, n_samples, 
    test_mode, plot=True, debug=False, config_nn=None):
    """Learn diffeq surrogate with NN-based solution 
    
    Approximates a snapshot of the solution of a diffeq, e.g., temperature, T, with a 
    neural network (NN). The NN is trained on observations of u, u_target 
    or u_true. After the NN has been trained to estimate the solution multiple samples
    of diffeq are created.
    
    Args:
        grid np.array((n_samples, n_grid)) or 
            (np.array(n_xgrid), np.array(n_tgrid)) or 
            np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, ..., grid_dim)): grid for 1D, 2D, or ND input, respectively
        u_target np.array((n_samples, n_tgrid, n_xgrid, n_vars)): 
        y_args np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, ..., grid_dim))
        u_true: Measurement; not tested
        u_params dict(np.array(n_samples, n_tgrid, n_xgrid)): Parameters of each diffeq sample
        grid_in_dims tuple(int): Indices of grid dimensions that are used as input, e.g., 0, 1 for tgrid and x1grid. -1 for all grids
        debug bool: If True, use debug args, e.g., epochs=1 
        See neural_net.interpolate_param_nn for other args.
    Returns:
        nn_direct_pred np.array((n_samples, n_out)): Output of neural network on test grid
        nn_pred np.array((n_samples, n_vars)): Solution, based on NN output, on test grid
    """
    # TODO: remove import from here
    from pce_pinns.models.neural_net import interpolate_param_nn
    # TODO: try to move the query of multi_indices into pce.py
    if kl_dim is not None and pce_dim is not None:
        alpha_indices = pce.multi_indices(ndim=kl_dim, pce_dim=pce_dim, order='total_degree')
        #alpha_indices = alpha_indices[:,4:] # Omit constant pce coefficient
    else:
        alpha_indices = None

    # TODO: TEST IF BATCH SIZE CAN BE > n_grid
    nn_direct_pred, us_nn = interpolate_param_nn(grid=grid, y=u_target,
        y_args=y_args,
        epochs=20, n_layers=4, n_units=1024, lr=0.001,
        target_name=est_param_nn,
        alpha_indices=alpha_indices, rand_insts=rand_insts,
        n_test_samples=n_samples,
        diffeq=diffeq, loss_name=loss_name, normalize=normalize,
        u_params=u_params, grid_in_dims=grid_in_dims,
        test_mode=test_mode,
        plot=plot, debug=debug, config=config_nn)

    return nn_direct_pred, us_nn

def approx_pce_coefs_w_nn(xgrid, model, model_args, 
    coefs_target, y_gp_mean, y_gp_cov, kl_dim, pce_dim,
    n_samples, plot=True):
    """
    Approximate PCE coefficients with neural network
    """

    coefs_nn, _ = interpolate_param_nn(grid=xgrid, y=coefs_target[0,:,:], n_epochs=20, 
        target_name="pce_coefs", plot=plot)

    # Get PCE sampling function with given neural net coefs
    _, _, _, _, _, model_args['sample_y'] = densities.sample_approx_gaussian_process(xgrid, 
        mu_y=y_gp_mean, cov=y_gp_cov,
        kl_dim=kl_dim, expansion='polynomial_chaos', z_candidate=None, 
        pce_dim=pce_dim, pce_coefs=coefs_nn)

    # Draw samples from PCE with neural net-based PCE coefficients 
    logs = n_samples * [None]
    for n in range(n_samples):
        u_ests, logs[n], _ = model(**model_args)
        
    k, Y, u, _, _, _ = logging.parse_logs(logs)

    return k, Y, u

def approx_k_eigvecs_w_nn(xgrid, diffeq,
    k_target, kl_dim, plot=True, unit_tests=False):
    """
    Approximate KL-expansion with neural network
    Source: Zhang et al., 2019, Chapter 3.2.1
    Args:
        xgrid np.array(n_grid): Grid where function was evaluated
        diffeq class: 
        k_target np.array(n_samples, n_grid): Measurements
        kl_dim int: Number of non-truncated eigenvalues for Karhunen-Loeve expansion
        plot bool: if true, plot
    """
    n_samples, n_grid = k_target.shape

    # Interpolate mu_k
    mu_k_target =  np.mean(k_target,axis=0)[np.newaxis,:,np.newaxis]
    mu_k_target_norm = (mu_k_target - np.mean(mu_k_target))# / np.std(mu_k_target) # Normalize (todo: integrate in NN)
    mu_k_nn, _ = interpolate_param_nn(grid=xgrid[np.newaxis,:], y=mu_k_target_norm, 
        n_epochs=2000, lr=0.000005, batch_size=2,  target_name="mu_k", plot=plot)
    mu_k_nn = mu_k_nn + np.mean(mu_k_target)##(mu_k_nn * np.std(mu_k_target)) + np.mean(mu_k_target) #
    # TODO WHY DOES A BATCH_SIZE > 1 largelgyEXTEND LEARNING TIME ?
    # TODO: use wandb or MLflow for hyperparam opt. 
    # Best: n_epochs=2000, lr=0.000005, batch_size=2, n_layers=2, n_units=128; only mean normalization
    # 2nd best: n_epochs=2000, lr=0.00002, batch_size=2, n_layers=2, n_units=128; normal normalization
    # good & quick: n_epochs=100, lr=0.002, batch_size=10, n_layers=2, n_units=128, normal normalization

    # Interpolate k_eigvecs
    cov = np.cov(k_target.T) # np.cov(k_target.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvecs_target = eigvecs[np.newaxis, :, :kl_dim]
    #eigvecs_target = np.repeat(eigvecs_target[:,:], repeats=n_samples, axis=0)
    k_eigvecs_nn, _ = interpolate_param_nn(grid=xgrid[np.newaxis,:], y=eigvecs_target, batch_size=n_grid, 
        n_epochs=900, lr=0.0005, target_name="k_eigvecs", plot=plot)

    # Extract random instances via PCA on covariance of k measurement matrix; Zhang et al., eq. (9)
    k_norm = k_target-np.mean(k_target,axis=0)
    cov_norm = np.cov(k_norm.T) 
    eigvals_norm, eigvecs_norm = np.linalg.eig(cov)
    eig_inv = 1./np.sqrt(eigvals_norm[:kl_dim])
    z_msmt = np.matmul(np.multiply(eigvecs_norm[:,:kl_dim], eig_inv).T, k_norm.T[:,:]).T

    # Sanity checks
    if unit_tests:
        # z_msmt centered; E[z_msmt] = 0
        assert(np.allclose(np.mean(z_msmt,axis=0),0))  # E[z_msmt] = 0
        # z_msmt uncorrelated; E[z_msmt_i z_msmt_j] = \delta_ij
        assert(np.allclose(1./float(n_samples)*np.matmul(z_msmt.T,z_msmt), np.identity(kl_dim),atol=0.001))
        # Check if there's numerical error in eigvals 
        assert(np.all(eigvals[:kl_dim]>0.))
        # Check that more than 95\% variance is explained in eigenvalues
        var_explained = []
        for eig_i in eigvals:
            var_explained.append((float(eig_i)/np.sum(eigvals))*100.)
        if np.cumsum(var_explained)[kl_dim]<95.:
            print('Less than 95\% variance is captured in eigenvalues up to kl_dim')
        
    # Compute solution
    import pdb;pdb.set_trace()
    ks_nn = np.zeros((n_samples, n_grid))
    ys_nn = np.zeros((n_samples, n_grid))
    us_nn = np.zeros((n_samples, n_grid))
    for n in range(n_samples):
        # Sample from NN-based KL-expansion
        ks_nn[n,:] = (mu_k_nn + np.matmul(np.multiply(k_eigvecs_nn[:,:kl_dim], z_msmt[n,:]), np.sqrt(eigvals[:kl_dim,np.newaxis])))[:,0]
        ys_nn[n,:] = np.log(np.where(ks_nn[n,:]>0, ks_nn[n,:], 1.)) # = np.log(k_sample)
        us_nn[n,:] = diffeq.solve(xgrid, k=ks_nn[n,:])[:,0]

    if plot:
        plotting.plot_mu_k_vs_ks_nn(xgrid, mu_k_target[0,:,0], mu_k_nn)
        plotting.plot_std_k_vs_ks_nn(xgrid, k_target, ks_nn)
        plotting.plot_k_eigvecs_vs_ks_nn(xgrid, eigvecs_target[0,:,:], k_eigvecs_nn)
        plotting.plot_kl_k_vs_ks_nn(xgrid, k_target, ks_nn)

    return ks_nn, ys_nn, us_nn

def approx_model_param_w_nn(xgrid, diffeq, k_target, k_true, 
    est_param_nn, kl_dim, pce_dim, rand_insts, n_samples, plot=True):
    """Solves diffeq with NN-based parameters 

    Approximates a model parameter in diffeq, e.g., permeability, k, with a 
    neural network (NN). The NN is trained on observations of k, k_target 
    or k_true. After the NN estimates the parameter multiple samples of diffeq are created.
    
    Args:
    Returns:
    """
    # TODO: try to move the query of multi_indices into pce.py
    alpha_indices = pce.multi_indices(ndim=kl_dim, pce_dim=pce_dim, order='total_degree')

    x_in = np.repeat(xgrid[np.newaxis,:], repeats=n_samples, axis=0)

    # Approximate k measurements
    if est_param_nn=='k_true':
        k_target = np.repeat(k_true[np.newaxis,:],repeats=n_samples, axis=0) # Observations
    #alpha_indices = alpha_indices[:,4:] # Omit constant pce coefficient
    # TODO: TEST IF BATCH SIZE CAN BE > n_grid
    pce_coefs, ks_nn = interpolate_param_nn(grid=x_in, y=k_target, target_name=est_param_nn, 
        alpha_indices=alpha_indices, rand_insts=rand_insts,
        batch_size=151, n_epochs=100, n_test_samples=n_samples, plot=plot)
    if True:
        plotting.plot_k_vs_ks_nn(xgrid, k_target, ks_nn)

    # Compute log-permeability
    ys_nn = np.log(np.where(ks_nn>0, ks_nn, 1.))

    # Compute solution
    us_nn = np.zeros((n_samples, n_grid))
    for n in range(n_samples):
        us_nn[n,:] = diffeq.solve(xgrid, k=ks_nn[n,:])[:,0]

    return ks_nn, ys_nn, us_nn
