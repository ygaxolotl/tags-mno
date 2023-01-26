"""
Density functions
"""
import numpy as np
import scipy.stats as st

import pce_pinns.rom.kl_expansion as kl # Reduced order models
import pce_pinns.rom.pce as pce

def sample_approx_gaussian_process(xgrid, 
    mu_y, cov, expansion='KL', 
    kl_dim=3, z_candidate=None, 
    pce_dim=5, pce_coefs=None, 
    plot=True):
    '''Samples from a gaussian process

    Computes a function that samples a gaussian process, given by mu_y and cov,
    via a reduced order model, e.g., a Karhunen-Loeve expansion

    Args:
        xgrid np.array(n_grid): 1D grid points
        mu_y np.array(n_grid): Mean over x
        cov_y np.array((n_grid,n_grid)): Covariance matrix of gaussian process
        expansion string: Chosen reduced order model
        kl_dim int: Number of non-truncated eigenvalues for Karhunen-Loeve expansion
        z_candidate np.array(kl_dim): If not None, these are candidates for the new random variables of the KL-exp
        pce_dim int: Maximum polynomial degree of polynomial chaos expansion, typically equal to kl_dim
        pce_coefs np.array(n_grid, pce_dim: If PCE coefficients are given, they will not be computed

    Returns:
        Y np.array(n_grid): Sample of low dim gaussian process
        exp_Y np.array(n_grid): Sample of low dim exp-gaussian process
        kl_trunc_err float: KL truncation error 
        coefs np.array((n_grid, pce_dim)): Polynomial chaos expansion coefficients 
        rand_insts np.array(kl_dim or pce_dim): Random instances used in expansion 
    '''
    n_grid = xgrid.shape[0]
    # Compute expansion
    Y, exp_Y, kl_trunc_err, kl_eigvals, kl_eigvecs, rand_insts, sampling_fn = kl.kl_expansion(xgrid, mu_y, cov, kl_dim, z_candidate=z_candidate)
    if expansion == 'KL':
        kl_trunc_err = kl_trunc_err
        coefs=(kl_eigvals, kl_eigvecs)
    elif expansion == 'polynomial_chaos':
        Y, exp_Y, kl_trunc_err, coefs, rand_insts, sampling_fn = pce.polynomial_chaos_expansion(xgrid, 
            kl_dim=kl_dim, kl_mu_y=mu_y, kl_eigvals=kl_eigvals, kl_eigvecs=kl_eigvecs,
            pce_dim=kl_dim, plot=plot, c_alphas=pce_coefs)
    return Y, exp_Y, kl_trunc_err, coefs, rand_insts, sampling_fn

def power_kernel(xgrid, var_y=0.3, lengthscale=0.3, p=1.):
    '''
    Computes the power kernel (TODO: find out actual name)
    
    Args:
        xgrid np.array(n_grid): Array of grid points
        var_y (int): Variance of the output stochastic process
        lengthscale (int): Lengthscale of the kernel
        p (int): Order of the kernel
    
    Returns:
        cov (np.array((n_grid,n_grid))): Covariance matrix
    '''
    n_grid = xgrid.shape[0]
    x1 = np.repeat(xgrid[:,np.newaxis], n_grid, axis=1)
    x2 = np.repeat(xgrid[np.newaxis,:], n_grid, axis=0)
    cov = var_y * np.exp(-1./p * np.power(np.absolute(x1 - x2) / lengthscale, p))

    return cov

def init_gaussian_process(xgrid, y_mean, y_var, lengthscale=0.3, order=1.):
    """
    Initializes parameters of a gaussian process

    Args:
        xgrid np.array(n_grid): 1D grid points
        y_mean float: Constant mean
        y_var float: Constant
        lengthscale float: kernel lengthscale
        order float: kernel order
    Returns:
        mu_y np.array(n_grid): Mean over x
        cov_y np.array((n_grid,n_grid)): Covariance matrix
    """
    #import GPy
    #vis_kernel1 = GPy.kern.RBF(input_dim=1, variance=3., lengthscale=1)
    n_grid = xgrid.shape[0]
    mu_y = np.repeat(y_mean, n_grid)

    # Compute covariance matrix
    cov = power_kernel(xgrid, var_y=y_var, lengthscale=lengthscale, p=order)

    return mu_y, cov

def calc_stats(u):
    """
    Calculates stats of set of samples u
    
    Args:
        u np.array(n_samples): Function of x
    
    Returns:
        u_stats dict(): Statistics of u
    """
    n_samples = u.shape[0]
    
    # Calculate statistics
    conf_bnds = 0.95
    u_stats = {
        'ux_cum_mean': np.zeros((n_samples)),
        'ux_cum_std' : np.zeros((n_samples)),
        'ux_cum_sem' : np.zeros((n_samples)),
        'ux_cum_var' : np.zeros((n_samples)),
        'ux_conf_int' : np.zeros((n_samples, 2)),
    }

    for n in range(n_samples):
        u_stats['ux_cum_mean'][n] = np.mean(u[:n+1])
        u_stats['ux_cum_std'][n] = np.std(u[:n+1], ddof=1) # Use n-1 in denominator for unbiased estimate. 
        u_stats['ux_cum_sem'][n] = st.sem(u[:n+1])
        u_stats['ux_cum_var'][n] = np.var(u[:n+1])
        if n>0:
            u_stats['ux_conf_int'][n,:] = st.t.interval(conf_bnds, n-1, loc=u_stats['ux_cum_mean'][n], scale=st.sem(u[:n+1]))

    """ Compute confidence interval                        
    for n in range(n_samples):
        est_n = u_stats['ux_cum_mean'][:n+1]
        #if n==0:
        #    ux_conf_int[n,:] = np.array([u_stats['ux_cum_mean'][:n+1], u_stats['ux_cum_mean'][:n+1]])[0]
        #else:
        ux_conf_int[n,:] = st.t.interval(conf_bnds, est_n.shape[0]-1, loc=np.mean(est_n), scale=st.sem(est_n))
    """
    return u_stats
