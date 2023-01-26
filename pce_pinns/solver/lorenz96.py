"""
Differential equation solver and boundary conditions for Lorenz 96
Author: 
"""
import os
import time
import timeit
import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

from pce_pinns.utils import plotting
from pce_pinns.utils.shaping import split_sequences_into_snippets
from pce_pinns.solver.diffeq import DiffEq

class Lorenz96Eq(DiffEq):
    def __init__(self, tgrid, param='fully_resolved', 
        y_param_mode='superparam', random_ic=True, 
        random_ic_y=True, random_ic_z=True, 
        dummy_init=False,
        plot=False, K=4, 
        tgrid_warmup=None, seed=0,
        dtype=np.float64,
        use_rk4=True):
        """
        Sets up Lorenz '96 equation as initial value problem. The equation
        models chaotic dynamics in three dimensions and is based on 
        Ch.2.1 in https://doi.org/10.1002/qj.2974 

        Args:
            tgrid np.array(n_grid): Grid points in time 
            param string: Type of subgrid parametrization, e.g., 'null' or 'fully_resolved'
            y_param_mode string: Mode of learned subgrid parametrization for y, e.g., 'superparam'
            random_ic bool: If true, use random initial conditions
            dummy_init bool: If true, uses computationally fast dummy initialization
            plot bool: If true, generates plots
            tgrid_warmup np.array(n_warmup): Grid points in time used to warmup equation, i.e., reach the attractor 
            dtype np.dtype: Datatype for storing Lorenz96 solution
            seed int: Seed for random number generator
            use_rk4 bool: If true, use RK4 time-stepping scheme. 
        Attributes:
            
        """
        self.name = 'lorenz96'
        self.tgrid = tgrid
        self.tgrid_warmup = tgrid_warmup
        self.dtype = dtype

        self.seed = seed
        self.rng = default_rng(self.seed)
        self.cumulative_runtime = 0. # Cumulative runtime for solve(). Can be divided by number of fn calls
        self.plot = plot

        # Coupling Parameters
        self.param = param
        self.y_param_mode = y_param_mode
        self.nyargs = 1 # Number of returned parametrizations in self.step()

        self.K = K # Number of large-scale variables, x
        self.J = self.K # Number of middle-scale variables, y
        self.I = self.K # Number of small-scale variables, z
        self.h = 0.5 # Coupling strength between spatial scales; no coupling for h=0
        self.b = 10. # rel. magnitude of x, y
        self.c = 8. # rel. evolution speed of x, y
        self.d = 10. # rel. magnitude ratio of y, z
        self.e = 10. # rel. evolution speed of y, z
        self.F = 20. # Large-scale forcing

        self.hc_b = self.h * self.c / self.b # Coupling strength and ratio x, y
        self.he_d = self.h * self.e / self.d # Coupling strength and ratio y, z

        # Finite difference schemes in space; periodic boundaries
        if self.K != self.I or self.K != self.J:
            raise NotImplementedError("K, I, and J are assumed to be equal.")
        self.minus = np.roll(np.arange(0, self.K), 1) # e.g., [1, 2, 3, 0] for self.K = 4
        self.minus2 = np.roll(np.arange(0, self.K), 2)
        self.plus = np.roll(np.arange(0, self.K),-1)
        self.plus2 = np.roll(np.arange(0, self.K), -2)
        self.use_rk4 = use_rk4 # Use Runge-Kutta 4 time-stepping 

        self.dummy_init = dummy_init
        self.random_ic = random_ic
        self.random_ic_y = random_ic_y # If false, fixes y0 when resampling x0, z0
        self.random_ic_z = random_ic_z # If false, fixes z0 when resampling x0, y0
        self.sample_random_ic(init=True, init_ones=self.dummy_init)
        self.sample_random_params()

    def step_rng(self):
        """ Iterate the random number generator 

        Iterates the random number generator, s.t., parallel processes have different output
        """
        self.rng.uniform(0, 1, size=(1))
        return 0

    def sample_random_params(self):
        """ Sample random parameters
    
        Sets all stochastic parameters to a new sample

        Returns:
            rand_insts np.array(stoch_dim): Random instances
            rand_param np.array(?): Random parameters
        """
        return np.ones(1,dtype=self.dtype), np.ones(1,dtype=self.dtype)

    def sample_random_ic(self, init=False, init_ones=False):
        """ Sample random initial condition

        Sets a new initial conditions to a new sample
    
        Args:
            init_ones bool: See Lorenz96Eq()
        Returns:
            rand_insts np.array(stoch_dim): Random instances
            rand_ic (np.array(K,),
                np.array(J,K),
                np.array(I,J,K): Random initial conditions
        """ 
        # Initial conditions
        def set_ic(random_ic_y=True, random_ic_z=True):
            self.x0 = self.rng.integers(-15, 15, size=(self.K,)).astype(self.dtype) # -5, 6
            if random_ic_y:
                self.y0 = 1. * self.rng.standard_normal((self.J, self.K)).astype(self.dtype) # 0.25
            if random_ic_z:
                self.z0 = .05 * self.rng.standard_normal((self.I, self.J, self.K)).astype(dtype=self.dtype)

        if self.random_ic and not init:
            set_ic(random_ic_y=self.random_ic_y, random_ic_z=self.random_ic_z)
        elif init and self.param=='z_null' and not init_ones:
            set_ic(random_ic_z=False)
            self.z0 = np.zeros((1, 1, 1), dtype=self.dtype)# self.I,self.J,self.K))
        elif init and not init_ones:
            set_ic()
            # Set fixed initial conditions only during initialization
            #try:
            #   self.x0
            #except:
            #    set_ic()
            #pass
        # Initalize with all 1 array to avoid comp. cost of random sampling
        elif init_ones:             
            self.x0 = np.ones((self.K), dtype=self.dtype)
            if self.y_param_mode == 'full_subgrid_forcing':
                # No need to init y0, as the y parametrization 
                # will be passed by external fn
                self.y0 = np.ones((1,1), dtype=self.dtype)
            else:
                self.y0 = np.ones((self.J, self.K), dtype=self.dtype)        
            if self.param=='z_null':
                self.z0 = np.ones((1,1,1), dtype=self.dtype)
            else:
                self.z0 = np.ones((self.I,self.J,self.K), dtype=self.dtype)

        return np.ones(1, dtype=self.dtype), (self.x0, self.y0, self.z0)

    def step(self, x, y, z, verbose=False): # , b, c, d, e, h
        """
        Compute 1st time derivative
        Source: Ch.2.1 in https://doi.org/10.1002/qj.2974 

        Args:
            x np.array(K,): Large-scale variables
            y np.array(J,K): Medium-scale variables
            z np.array(I,J,K): Small-scale variables
        Returns:
            dx np.array(K,): Derivative of large-scale variables over time
            dy np.array(J,K): Derivative of medium-scale variables over time
            dz np.array(I,J,K): Derivative of small-scale variables over time
        """
        if self.param == 'null':
            y_param = np.zeros(x.shape, dtype=self.dtype)
            z_param = np.zeros(y.shape, dtype=self.dtype)
        elif self.param == 'z_null':
            y_param = - self.hc_b * np.sum(y,axis=0)
            z_param = np.zeros(y.shape, dtype=self.dtype)
        elif self.param == 'fully_resolved':
            z_param = - self.he_d * np.sum(z,axis=0)
            y_param = - self.hc_b * np.sum(y,axis=0)
        else:
            raise NotImplementedError('Lorenz96.param unknown: ', self.param)

        # Calculate large-scale variables
        dx = np.multiply(x[self.minus],(x[self.plus] - x[self.minus2])) - x + self.F + y_param
        # Calculate medium-scale variables
        if not self.param == 'null':
            dy = - self.c * self.b * np.multiply(y[self.plus,:], (y[self.plus2,:] - y[self.minus,:])) - self.c * y + self.hc_b * x + z_param
            # dy = - self.c * self.b * np.multiply(y[self.plus,:], (y[self.plus2,:] - y[self.minus,:])) # - self.c * y + self.hc_b * x + z_param
            # dy = - self.c * y + self.hc_b * x + z_param
            # dy = self.hc_b * x + z_param
            # dy = np.zeros(y.shape)
        else:
            dy = np.zeros(y.shape, dtype=self.dtype)
        # Calculate high-res. veriables
        if self.param =='fully_resolved': 
            dz = self.e * self.d * np.multiply(z[self.minus,:,:],(z[self.plus,:,:] - z[self.minus2,:,:])) - self.e * z + self.he_d * y
        else:
            dz = np.zeros(z.shape, dtype=self.dtype)
            
        return (dx, dy, dz), (y_param)

    def step_large_scale(self, x, f_y, verbose=False):
        """
        Steps large-scale model with predicted subgrid forcing.
        Computes 1st time derivative and integrates with RK4 in time. 
        In torch.

        Args:
            x torch.Tensor(K,): Large-scale variables
            f_y torch.Tensor(see below): Subgrid forcing from medium-scale variables
        Returns:
            xnext torch.Tensor(K,): Next timestep's large-scale variable
        """
        import torch
        dt = self.tgrid[-1] - self.tgrid[-2]

        if self.y_param_mode == 'full_subgrid_forcing':
            # Args: f_y np.array(K)
            y_param = f_y
        elif self.y_param_mode == 'mean_superparam':
            # Args: f_y np.array(K)
            y_param = - self.hc_b * f_y
        elif self.y_param_mode == 'superparam':
            # Args: f_y np.array(J*K)
            y_param = - self.hc_b * torch.sum(f_y.reshape(self.J,self.K),axis=0)
        else:
            raise NotImplementedError(f'Unknown mode {y_param_mode} of predicted parametrization') 

        # Calculate large-scale derivative
        def dx(x):
            return torch.multiply(x[self.minus],(x[self.plus] - x[self.minus2])) - x + self.F

        # Integrate in time 
        dx1 = dx(x)
        xnext = x + dt * (dx1 + y_param)
        """
        if self.use_rk4:

            Rx1 = x + 0.5 * dt * dx1
            dx2 = dx(Rx1)
            
            Rx3 = x + 0.5 * dt * dx2
            dx3 = dx(Rx3)
            
            Rx4 = x + dt * dx3
            dx4 = dx(Rx4)

            xnext = x + dt / 6. * (dx1 + 2. * dx2 + 2. * dx3 + dx4)
        """

        return xnext

    def residual(self, x, y, z, t):
        """
        Computes autodifferentiable squared residual of proposed solution.
        Args:
        """
        raise NotImplementedError
        return 0

    def solve(self, solver_name=None, warmup=False, set_final_as_ic=False, verbose=False):
        """
        Solves the equation

        Args:
            warmup bool: If true, warms up the equation by using tgrid_warmup.
            set_final_as_ic: If true, sets terminal/final state as initial condition.
        Returns:
            sol (np.array(n_tgrid, K),
                np.array(n_tgrid, J, K),
                np.array(n_tgrid, I, J, K),
                np.array(n_tgrid, K): Parametrization of medium-scale onto large-scale effects, y_param. 
                    If warmup only terminal state is returned with n_tgrid==1.
        """
        if solver_name is None or solver_name=='RK4': # Use Runge-Kutta 4
            tgrid = self.tgrid if not warmup else self.tgrid_warmup
            dt = tgrid[-1] - tgrid[-2]
            if not warmup:
                solx = np.zeros((tgrid.shape[0], self.K), dtype=self.dtype)
                soly = np.zeros((tgrid.shape[0], self.J, self.K), dtype=self.dtype)
                if self.param=='z_null':
                    solz = np.zeros((tgrid.shape[0], 1, 1, 1), dtype=self.dtype)# self.I,self.J,self.K))
                else:
                    solz = np.zeros((tgrid.shape[0], self.I, self. J, self.K), dtype=self.dtype)
                solx[0,:] = self.x0
                soly[0,:,:] = self.y0
                solz[0,:,:,:] = self.z0
            soli = (self.x0, self.y0, self.z0)
            yargs = np.zeros((tgrid.shape[0], self.K), dtype=self.dtype)

            nvars = len(soli)
            nyargs = self.nyargs
            start = time.time()
            for i, t in enumerate(tgrid[1:], 1):
                dsol1, yargs1 = self.step(*soli)

                if self.use_rk4:
                    Rsol2 = [soli[k] + 0.5 * dt * dsol1[k] for k in range(nvars)]
                    dsol2, yargs2 = self.step(*Rsol2)
                    
                    Rsol3 = [soli[k] + 0.5 * dt * dsol2[k] for k in range(nvars)]
                    dsol3, yargs3 = self.step(*Rsol3)
                    
                    Rsol4 = [soli[k] + dt * dsol3[k] for k in range(nvars)] 
                    dsol4, yargs4 = self.step(*Rsol4)

                    dsoli = [1. / 6. * (dsol1[k] + 2. * dsol2[k] + 2. * dsol3[k] + dsol4[k]) for k in range(nvars)]
                    
                    soli = [soli[k] + dt * dsoli[k] for k in range(nvars)]
                    
                    if not warmup: # Do not record states during warmup
                        solx[i,:], soly[i,:,:], solz[i,:,:,:] = [soli[k] for k in range(nvars)]
                    
                    # yargs[i-1] = np.asarray([1. / 6. * (yargs1[k] + 2. * yargs2[k] + 2. * yargs3[k] + yargs4[k]) for k in range(nyargs)])
                    # (dsol1[0] - yargs1) returns 1-step solution with null parametrization
                    # dsoli[0] is RK4 time-stepped solution of filtered high-resolution
                    # yargs[i-1] is SGS parametrization, i.e., difference between filtered high-res. and null parametrization.
                    yargs[i-1] = dsoli[0] - (dsol1[0] - yargs1)
                    if nyargs > 1:
                        raise Warning('RK4 time-integration only implemented for one parametrization.')
                else:
                    soli = [soli[k] + dt * dsoli[k] for k in range(nvars)]

                    if not warmup: # Do not record states during warmup
                        solx[i,:], soly[i,:,:], solz[i,:,:,:] = [soli[k] for k in range(nvars)]
                    
                    yargs[i-1] = np.asarray([yargs1[k] for k in range(nyargs)])
                    
                # soli = (solx[i,:], soly[i,:,:], solz[i,:,:,:])

                if i%100 == 0 and verbose: print(i)
            if warmup:
                # TODO check if soli.shape[0]==1
                sol = (soli[0][None,...], soli[1][None,...], soli[2][None,...], yargs[-2:-1,...])
            else:
                sol = (solx, soly, solz, yargs)

            self.cumulative_runtime += (time.time() - start)/float(len(tgrid[1:]))
        else:
            raise NotImplementedError(f'Solver {solver_name} not implemented for Lorenz96')
        if self.plot:
            plotting.plot_lorenz96(tgrid, solx, soly, solz, self.K)

        if set_final_as_ic:
            self.x0 = soli[0][...]
            self.y0 = soli[1][...]
            self.z0 = soli[2][...]

        return sol

    def get_y_param(self, y):
        """
        Returns full y parametrization term, given y over time
        Args:   
            y np.array(n_tgrid, J, K)
        Returns:
            y_param np.array(n_tgrid, K)
        """
        y_param = - self.hc_b * np.sum(y,axis=1)

    def test_full_subgrid_forcing(self, x_target, y_param, tgrid, plot=False):
        """
        Tests accuracy of ground-truth solution vs. predicted parametrization 
        Args:
            x_target np.array(n_tgrid, K): Low res Solution 
            y_param np.array(n_tgrid, K): Full subgrid forcing parametrization
            tgrid np.array(n_tgrid): Temporal grid
        Returns:
            x np.array(n_tgrid, K): Predicted low-res solution
        """
        import torch
        assert y_param.dtype == self.dtype
        assert x_target.dtype == self.dtype

        y_param_mode_ori = self.y_param_mode
        self.y_param_mode = 'full_subgrid_forcing'
        x = torch.from_numpy(np.zeros(y_param.shape, dtype=self.dtype))
        y_param = torch.from_numpy(y_param)
        x[0,:] = torch.from_numpy(x_target[0,:])
        for t in range(1,x_target.shape[0]):
            x[t,:] = self.step_large_scale(x[t-1,:], y_param[t-1,:], verbose=True)

        if plot:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            # Plot predicted vs. ground-truth large-scale solution 
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16,8), dpi=100)
            colors = cm.get_cmap('tab20')
            for k in range(self.K):
                if k>3:
                    break
                axs.plot(tgrid[::2], x[::2,k], color=colors.colors[2*k], linewidth=1, label=rf'$k={k}$')# label='Predicted, $X_'+str(k)+'$')
                axs.plot(tgrid[::2], x_target[::2,k], color=colors.colors[2*k+1], linewidth=2, linestyle=':')#, label='Ground-Truth, $X_'+str(k)+'$')

            axs.set_xlabel(r'time, $t$')
            axs.set_ylabel(r'solution $X_k$')
            axs.legend()
            fig.savefig('doc/figures/lorenz96/sol_w_full_subgrid_forcing.png')
            plt.close(fig)
            print('Target low-res solution matches reconstructed solution '\
                ' w parametrization target :', np.all(x[:,k] == x_target[:,k]))
            self.y_param_mode = y_param_mode_ori
            import pdb;pdb.set_trace()

        return x.cpu().numpy()

    def dump(self, folder='data/lorenz96/config/', filename='diffeq.obj'):
        """
        Dumps diffeq config to file
        """
        path = Path(folder,filename)
        print('Saving Lorenz96 object to file at ', path)
        if not path.exists():
            path.mkdir()
        with open(path, 'wb') as filehandler:
          pickle.dump(self, filehandler)
        return 1

def load(folder='data/lorenz96/config/', filename='diffeq.obj'):
    """
    Loads diffeq object from file
    """
    filehandler = open(Path(folder, filename), 'r') 
    lorenz96Eq = pickle.load(filehandler)
    return lorenz96eq

def reshape_lorenz96_to_snippets(tgrid, sol, sol_prior, u_args, n_snippets, n_tsnippet):
    """
    Args:
        tgrid np.array((n_tgrid))
        sol (np.array((n_sequences, n_tgrid, K))
            np.array((n_sequences, n_tgrid, J, K))
            np.array((n_sequences, n_tgrid, I, J, K))
        )
        sol_prior (np.array((n_sequences, n_tgrid, K))
            np.array((n_sequences, n_tgrid, J, K))
            np.array((n_sequences, n_tgrid, I, J, K))
        ):  Approximate solution that is known ahead of time
        u_args dict(): see logging.convert_log_dict_to_np_lorenz96()
        n_snippets int: see utils.shaping.split_sequences_into_snippets()
        n_tsnippet int: time length of individual snippet 
   Returns:
        tgrid_snippet = np.array((n_tsnippet))
        sol (np.array((n_snippets, n_tsnippet, K))
            np.array((n_snippets, n_tsnippet, J, K))
            np.array((n_snippets, n_tsnippet, I, J, K))
        )
        sol_prior (np.array((n_snippets, n_tsnippet, K))
            np.array((n_snippets, n_tsnippet, J, K))
            np.array((n_snippets, n_tsnippet, I, J, K))
        ):  Approximate solution that is known ahead of time
        u_args {
            'y_param': np.array((n_snippets, n_tgrid, K)): Parametrization of middle-res onto low-res effects
            'rand_param': np.array((n_snippets, 1)) : Random parameter, currently undefined
            'rand_ic': (np.array((n_snippets, K))
                np.array((n_snippets, J, K))
                np.array((n_snippets, I, J, K))
                ): Random initial condition
        }         
    """
    import math
    n_sequences = sol[0].shape[0]
       
    # Create new tgrid by taking first n_tsnippet values. This assumes that 
    # tgrid is periodic after n_tsnippet time steps
    tgrid_snippet = tgrid[:n_tsnippet]

    # Convert tuple to list to support item assignment
    sol = [*sol]
    u_args['rand_ic'] = [*u_args['rand_ic']]

    # Split sol, sol_prior, y_param of length n_tgrid into snippets of length n_tsnippet
    for i, soli in enumerate(sol):
        sol[i] = split_sequences_into_snippets(sol[i], n_snippets, n_tsnippet)
    u_args['y_param'] = split_sequences_into_snippets(u_args['y_param'], 
                           n_snippets, n_tsnippet)
    if sol_prior is not None:
        sol_prior = [*sol_prior]
        for i, sol_priori in enumerate(sol_prior):
            sol_prior[i] = split_sequences_into_snippets(sol_prior[i], 
                           n_snippets, n_tsnippet) 
        sol_prior = *sol_prior,

    # Repeat random parameters as all snippets in a sequence share the same random parameters
    u_args['rand_param'] = np.repeat(u_args['rand_param'], repeats=(math.ceil(n_snippets/n_sequences)), axis=0)[:n_snippets]
    
    # Set initial conditions to initial states in snippets 
    for i, rand_ic in enumerate(u_args['rand_ic']):
        u_args['rand_ic'][i] = np.repeat(rand_ic, repeats=(math.ceil(n_snippets/n_sequences)), axis=0)[:n_snippets]
        try:
            u_args['rand_ic'][i][:,...] = sol[i][:,0,...]
        except:
            print(f'u_args didnt work for i={i}')
            # If except is hit, the initial condition remains the same. This is the case, e.g., for 
            # diffeq.z0 which might not be fully stored. 
            pass

    # Convert lists back to tuple
    sol = *sol,
    u_args['rand_ic'] = *u_args['rand_ic'],

    return tgrid_snippet, sol, sol_prior, u_args

def reshape_lorenz96_to_nn_model(sol, tgrid, est_lorenz, 
    model_cfg={'type':'fcnn'},sol_prior=None, u_args=None, 
    n_snippets=None, n_tsnippet=None):
    """
    Reshapes lorenz96.solve() output and LorenzEq.tgrid into the desired in-/output

    Args:
        sol (np.array((n_samples, n_tgrid, K))
            np.array((n_samples, n_tgrid, J, K))
            np.array((n_samples, n_tgrid, I, J, K))
        )
        tgrid np.array((n_tgrid))
        est_lorenz string: Variable to estimate, e.g., low-res "X" or middle-res, "Y"
        model_cfg {
            'type': string: Desired model type, e.g., 'fno', 'fcnn'
            'model_dims': int: Number of desired feature dimension (opt)
        sol_prior (np.array((n_samples, n_tgrid, K))
            np.array((n_samples, n_tgrid, J, K))
            np.array((n_samples, n_tgrid, I, J, K))
        ):  Approximate solution that is known ahead of time
        u_args dict(): see logging.convert_log_dict_to_np_lorenz96()
        n_snippets int: utils.shaping.split_sequences_into_snippets()
        n_tsnippet int: utils.shaping.split_sequences_into_snippets()
    Returns:
        see subfunctions 'reshape_lorenz96_to_*'
    """
    if n_snippets is not None:
        tgrid, sol, sol_prior, u_args = reshape_lorenz96_to_snippets(tgrid=tgrid, 
            sol=sol, sol_prior=sol_prior, u_args=u_args, n_snippets=n_snippets,
            n_tsnippet=n_tsnippet)

    # Select function to reshape.  
    if model_cfg['type'] == 'mno':
        if model_cfg['model_dims'] == 1:
            reshape_fn = reshape_lorenz96_to_fno1d
    elif model_cfg['type'] == 'mno':
        if model_cfg['model_dims'] == 2:
            reshape_fn = reshape_lorenz96_to_fno2d
    elif model_cfg['type'] == 'fcnn':
        reshape_fn = reshape_lorenz96_to_nn
    elif model_cfg['type'] == 'poly':
        reshape_fn = reshape_lorenz96_to_poly
    elif model_cfg['type'] == 'climatology':
        reshape_fn = reshape_lorenz96_to_nn
    else:
        raise NotImplementedError('Model type {:s} not implemented. '\
            'Use e.g., "fcnn" instead.'.format(model_cfg['type']) )

    u_target, grid, y_args = reshape_fn(sol=sol,
        tgrid=tgrid, est_lorenz=est_lorenz, 
        sol_prior=sol_prior, u_args=u_args)

    return u_target, grid, y_args

def reshape_lorenz96_to_nn(sol, tgrid, est_lorenz, sol_prior=None, u_args=None,
        shared_grid=False):
    """
    Reshapes lorenz96.solve() output and LorenzEq.tgrid into flexible NN in-/output

    Args:
        see reshape_lorenz96_to_model_type
        shared_grid bool: If true, assumes that grid is shared among n_samples
    Returns:
        u_target np.array(n_samples, n_tgrid, {K or J*K or I*J*K})
        grid np.array(n_samples, n_tgrid, dim_grid): dim_grid = 1
        y_args np.array((n_samples, n_tgrid, dim_y_args)): Function arguments, e.g., X_t, Y_t-1
    """
    n_samples = sol[0].shape[0]
    # Reshape grid
    n_tgrid = tgrid.shape[0]
    if shared_grid:
        grid = np.repeat(tgrid[np.newaxis,:,np.newaxis], repeats=1, axis=0)
    else:
        # todo
        grid = np.repeat(tgrid[np.newaxis,:,np.newaxis], repeats=n_samples, axis=0)

    # Reshape solution
    # See main_lorenz96.py for description
    if est_lorenz == "X": # Select low-res variable X as target
        u_target = sol[0][...,None]
        y_args = None
    elif est_lorenz == "Y": # Select mid-res variable Y as target
        u_target = sol[1].reshape(n_samples, n_tgrid, -1)
        y_args = sol[0]
    elif est_lorenz == "no-scale-sep": # Predict Y_{0:J,0:k}(t) = NN(X_{0:K}(t-1), Y_{0:J,0:K}(t-1]))
        dt = 1 # Timesteps into past
        y_t = sol[1].reshape(n_samples, n_tgrid, -1) # flatten J,K
        y_tplus = np.roll(y_t, shift=-dt, axis=1) # shift future values back.
        x_t = sol[0]
        u_target = y_tplus
        y_args = np.concatenate((x_t, y_t), axis=2)
        assert np.all(y_t[0,5,:]==u_target[0,5-dt,:])
    elif est_lorenz == "superparam": 
        dt = 1
        K = sol[1].shape[-1]
        J = sol[1].shape[-2]
        # Merge k-channel into different samples
        y_t = np.moveaxis(sol[1],source=-1,destination=1)
        y_t = y_t.reshape(n_samples*K,n_tgrid,J)
        y_tplus = np.roll(y_t, shift=-dt, axis=1) # shift future values back
        x_t = np.repeat(sol[0], repeats=K, axis=0)
        u_target = y_tplus
        y_args = np.concatenate((x_t,y_t), axis=2)
    elif est_lorenz == "no-scale-sep-corr":
        print('This case doesn\"t really make sense. todo: reimplement.')
        dt = 1
        y_t = sol[1].reshape(n_samples, n_tgrid, -1) # flatten J,K
        y_tplus = np.roll(y_t, shift=-dt, axis=1) # shift future values back.
        u_target = y_tplus
        x_prior = sol_prior[0] # Concatenate with prior coarse solution
        y_args = np.concatenate((x_prior, y_t), axis=-1)
    elif est_lorenz == "no-scale-sep-param-no-mem": # for f_{0:K}(t+1)        = NN(X_{0:K}(t))'\
        dt = 1
        # TODO: find out if I need to roll y_param
        u_target = u_args['y_param']
        y_args = sol[0] # x_t
        # dt = 1 # Timesteps into past
        # x_t = sol[0] # x_t
        # y_args = x_t
        # param_t = sol[TODO]
        # u_target = np.roll(param_t, shift=-dt, axis=1) # x_tplus
    elif est_lorenz == "param-no-mem":
        # for f_k(t+1)            = NN(X_k(t)) = hc/b sum_{j=0}^J Y_{j,k}(t+1)
        """
        Returns:
            u_target np.array(n_samples, n_tgrid, n_x1grid,...,n_xngrid, 1)
            y_args np.array((n_samples, n_tgrid, n_x1grid,...,n_xngrid, 1))
        """
        u_target = u_args['y_param'][...,None]
        y_args = sol[0][...,None]
    else:
        raise NotImplementedError()

    return u_target, grid, y_args

def reshape_lorenz96_to_poly(sol, tgrid, est_lorenz, sol_prior=None, u_args=None):
    n_samples = sol[0].shape[0]

    # Reshape grid
    n_tgrid = tgrid.shape[0]
    grid = np.repeat(tgrid[np.newaxis,:,np.newaxis], repeats=n_samples, axis=0)

    if est_lorenz == "param-no-mem":
        # for f_k(t+1)            = NN(X_k(t)) = hc/b sum_{j=0}^J Y_{j,k}(t+1)
        u_target = u_args['y_param']
        y_args = sol[0] # x_0:K
    else:
        raise NotImplementedError()

    return u_target, grid, y_args

def reshape_lorenz96_to_fno1d(sol, tgrid, est_lorenz, sol_prior=None, u_args=None,
    shared_grid=False):
    """
    Reshapes lorenz96.solve() output and LorenzEq.tgrid into 1D FNO in-/output

    Args:
        see reshape_lorenz96_to_model_type
    Returns:
        u_target np.array(n_samples, n_tgrid, K, 1)
        grid np.array(n_samples, n_tgrid, dim_grid): dim_grid = 1
        y_args np.array((n_samples, n_tgrid, K, dim_y_args)): Function arguments, e.g., X_t, Y_t-1
    """
    u_target, grid, y_args = reshape_lorenz96_to_nn(sol=sol,
        tgrid=tgrid, est_lorenz=est_lorenz, 
        sol_prior=sol_prior, u_args=u_args,
        shared_grid=shared_grid)
    u_target = u_target[...,np.newaxis]
    y_args = y_args[...,np.newaxis]

    return u_target, grid, y_args

def reshape_lorenz96_to_fno2d(sol, tgrid, est_lorenz, sol_prior=None, u_args=None):
    """
    Reshapes lorenz96.solve() output and LorenzEq.tgrid into 1D or 2D CNN in-/output

    Args:
        see reshape_lorenz96_to_model_type
    Returns:
        u_target np.array(n_samples, n_tgrid, J, K, 1)
        grid np.array(n_samples, n_tgrid, dim_grid): dim_grid = 1
        y_args np.array((n_samples, n_tgrid, J, K, dim_y_args)): Function arguments, e.g., X_t, Y_t-1
    """
    n_samples = sol[0].shape[0]
    K = sol[0].shape[-1]
    J = sol[1].shape[-2]

    # Reshape grid
    n_tgrid = tgrid.shape[0]
    grid = np.repeat(tgrid[np.newaxis,:,np.newaxis], repeats=n_samples, axis=0)

    # Reshape solution
    # See main_lorenz96.py for description
    if est_lorenz == "X": # Select low-res variable X as target
        raise NotImplementedError()
    elif est_lorenz == "Y": # Select mid-res variable Y as target
        raise NotImplementedError()
    elif est_lorenz == "no-scale-sep": # Predict Y_{0:J,0:k}(t) = NN(X_{0:K}(t-1), Y_{0:J,0:K}(t-1]))
        dt = 1 # Timesteps into past
        y_t = sol[1][...,np.newaxis] 
        y_tplus = np.roll(y_t, shift=-dt, axis=1) # shift future values back.
        u_target = y_tplus
        x_t = sol[0][:,:,np.newaxis,:,np.newaxis]
        x_t = np.repeat(x_t, repeats=J, axis=2)
        y_args = np.concatenate((x_t, y_t), axis=-1)
        assert np.all(y_t[0,5,:,:,:]==u_target[0,5-dt,:,:,:])
    elif est_lorenz == "superparam": 
        raise NotImplementedError()
    elif est_lorenz == "no-scale-sep-corr":
        raise NotImplementedError()
    elif est_lorenz == "no-scale-sep-param-no-mem": # for f_{0:K}(t+1)        = NN(X_{0:K}(t))'\
        y_param = u_args['y_param']
        y_param = y_param[:,:,np.newaxis,:,np.newaxis]
        y_param = np.repeat(y_param, repeats=K, axis=2)
        u_target = y_param
        y_args = sol[0][:,:,np.newaxis,:,np.newaxis] # x_t
        y_args = np.repeat(y_args, repeats=K, axis=2)
    else:
        raise NotImplementedError()

    return u_target, grid, y_args

def init_sample_diffeq_instance(grid_size, do_init=False, dummy_init=False, dtype=np.float64):
    """
    Initializes a sample instance of Lorenz96 equation that can 
    be used to measure runtimes.
    Args:
        grid_size int: Grid size, e.g., K
        do_init bool: Sample new initial conditions for every instance.
        dummy_init bool: If true, initializes all ones to save comp. cost.        
    Returns:
        lorenz96 : Sample instance
        [x: Sample large-scale output
        y:
        z]
    """
    do_init = False
    # Create sample tgrid. Note: values not relevant as
    # they do not impact the runtime of step() or step_large_scale
    dt = 0.005
    tmax = 10.
    tgrid = np.linspace(0.,tmax, int(tmax/dt), dtype=dtype)

    lorenz96 = Lorenz96Eq(tgrid, param='z_null', 
        y_param_mode='full_subgrid_forcing', 
        random_ic_z=False, dummy_init=dummy_init,
        plot=False, K=grid_size, dtype=dtype, seed=1)

    # Set initial value
    if do_init:
        _, (x, y, z) = lorenz96.sample_random_ic(init=True, init_ones=False)
    else:
        x = lorenz96.x0
        y = lorenz96.y0
        z = lorenz96.z0
    
    return lorenz96, [x, y, z]

def measure_runtime_lorenz96(dtype=np.float64):
    """
    Creates log-log runtime vs. grid-size plot of Lorenz96 Equation. The runtimes
    are averaged across m_samples runs. Running the large gridsizes K>4096 requries 
    >8GB RAM. Only the runtime of Lorenz96.step() is timed.

    Args:
        tgrid np.array(n_tgrid): Temporal grid to initialize Lorenz96 Equation
    Returns:
        runtimes list(): List of runtimes         
        Ks list(): List of grid sizes that have been sampled.
    """
    print("Creating runtime plot for Lorenz96. This might take >10min.")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    runtimes = []
    # Todo: adapt maximum grid-size and m_samples according to RAM and CPU processing time. 
    Ks = [32768, 16384, 8192, 4096, 2048, 1024, 939, 861, 724, 663, 609, 512, 256, 128, 64, 32, 16, 8, 4]
    m_samples = (1*np.ones(len(Ks),dtype=int)).astype(int) # Repeats time measurement m-times
    m_samples[Ks.index(32768)] = 1
    m_samples[Ks.index(16384)] = 1
    m_samples[Ks.index(4096)] = 2
    m_samples[Ks.index(2048)] = 10
    m_samples[Ks.index(1024)] = 1000
    m_samples[Ks.index(939)] = 1000
    m_samples[Ks.index(861)] = 1000
    m_samples[Ks.index(724)] = 1000
    m_samples[Ks.index(663)] = 1000
    m_samples[Ks.index(609)] = 1000
    m_samples[Ks.index(512)] = 2000
    m_samples[Ks.index(256)] = 10000
    m_samples[Ks.index(128)] = 10000
    m_samples[Ks.index(64)] = 10000
    m_samples[Ks.index(32)] = 10000
    m_samples[Ks.index(16)] = 10000
    m_samples[Ks.index(8)] = 100000
    m_samples[Ks.index(4)] = 100000 
    for i,K in enumerate(Ks):
        lorenz96, sample = init_sample_diffeq_instance(grid_size=K, dtype=dtype)

        t = timeit.Timer(lambda: lorenz96.step(sample[0],sample[1],sample[2]))
        runtime = min(t.repeat(repeat=m_samples[i], number=1))
        runtimes.append(runtime)
        print(f'Average runtime K={K:04d}: {runtime:.8f}s')

    print('runtimes', runtimes)
    plotting.plot_lorenz96_runtimes(Ns=Ks, ts=runtimes, fname='runtimes_lorenz96_float16')
    os.environ["OMP_NUM_THREADS"] = ""
    os.environ["NUMEXPR_NUM_THREADS"] = ""
    os.environ["MKL_NUM_THREADS"] = ""

    return runtimes, Ks

def test_dtype(dtype=np.float64):
    """Tests if Lorenz96Eq.solve() maintains given dtype
    """
    lorenz96, _ = init_sample_diffeq_instance(grid_size=4, dtype=dtype)
    sol = lorenz96.solve(verbose=True)

    for soli in sol:
        assert soli.dtype==dtype, "Lorenz96 equation produces "\
        f"wrong data type. Check that data type {dtype} is "\
        "correctly propagated throught solution process."

    return 0

if __name__ == "__main__":
    """
    Create plot of test lorenz96 equation. See plotting.plot_lorenz96 for directory
    """
    parser = argparse.ArgumentParser(description='Lorenz 96')
    args = parser.parse_args()  

    dtype = np.float16
    dt = 0.005
    tmax = 10.
    tgrid = np.linspace(0.,tmax, int(tmax/dt), dtype=dtype)

    # Create log-log plot of runtime vs. grid-size 
    msr_runtime=True
    if msr_runtime:
        measure_runtime_lorenz96(dtype=dtype)
    else: # Plot the data from previos runs
        plotting.plot_lorenz96_runtimes(Ns=None, ts=None, fname='runtimes_lorenz96')
    import sys;sys.exit()

    # Test if dtype is correctly implemented
    test_dtype(dtype=dtype)

    # Test solution
    lorenz96 = Lorenz96Eq(tgrid, param='z_null', plot=True, seed=0)
    print('Ground-Truth:')
    sol = lorenz96.solve(verbose=True)

    # Test parametrization
    lorenz96.test_full_subgrid_forcing(x_target=sol[0], y_param=sol[3], tgrid=tgrid)
 
