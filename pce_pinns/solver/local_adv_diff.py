import argparse
import numpy as np

from scipy.integrate import odeint

# Import torch to compute autodifferentiable residual
import torch
import torch.nn.functional as F

import pce_pinns.utils.plotting as plotting
import pce_pinns.utils.densities as densities
from pce_pinns.solver.diffeq import DiffEq

class LocalAdvDiffEq(DiffEq):
    def __init__(self, tgrid, zgrid, random_ic=False, plot=False):
        """
        Sets up local advection diffusion equation, also called horizontally
        averaged Boussinesq approximation as initial value problem. The equation
        models the temperature distribution over time in a vertical ocean column.

        f = dT/dt + d(wT)/dz - d/dz(k dT/dz)

        Args:
            tgrid np.array(n_grid): Grid points in time 
            zgrid np.array(n_grid): Grid points in height
            random_ic bool: If true, use random initial conditions

        Attributes:
            source np.array(n_grid) or float: Source term, either a vector of values at points in xgrid or a constant
        """
        self.name = 'localAdvDiff'

        self.tgrid = tgrid
        self.zgrid = zgrid

        self.plot = plot
        self.verbose = False

        self.source = lambda: np.zeros(zgrid.shape[0])
        self.diffusivity = self.sample_diffusivity # lambda: np.ones(zgrid.shape[0]) # Sample diffusivity parameter
        self.sample_ln_diffusivity = None # Function to sample log-diffusivity
        self.vel = self.sample_vel # Sample wT parametrization

        self.random_ic = random_ic
        self.sample_random_ic()
        self.sample_random_params()

        self.check_stability()
        
    def check_stability(self):
        """
        Checks stabibility criteria of local advection diffusion equation
        stability criteria: dx < 2k/w and dt < dx^2/2k

        Returns:
            stable bool: True if stable, else False
        """
        dt = np.diff(self.tgrid, n=1, axis=0, append=2*self.tgrid[-1]-self.tgrid[-2])
        dx = np.diff(self.zgrid, n=1, axis=0, append=2*self.zgrid[-1]-self.zgrid[-2])

        peclet_x = self.w * dx / self.k
        peclet_t = np.mean(dt)*np.mean(self.k) / np.mean(np.power(dx,2.)) # Is taking the average correct to mix dx and dt in this calculation?
        print('Peclet x: ', np.mean(peclet_x))
        print('Peclet t: ', np.mean(peclet_t))

        stable = False
        if not np.all(peclet_x < 2.):
            print('[WARNING] Reduce spatial step size. Function likely instable. Peclet_x: ', peclet_x)
        elif not np.mean(peclet_t)< 2.:
            print('[WARNING] Reduce temporal step size. Function likely instable. Peclet_t: ', peclet_t)
        else:
            stable = True

        return stable

    def sample_random_params(self):
        """ Sample random parameters

        Sets all stochastic parameters to a new sample

        Returns:
            k_rand_insts np.array(gp_stoch_dim): Random instances used in approximation of GP 
            rand_param np.array(n_zgrid): Values of random parameter
        """
        self.w = self.vel()
        self.k, k_rand_insts = self.diffusivity()
        self.f = self.source()

        return k_rand_insts, np.copy(self.k)

    def sample_random_ic(self):
        """ TODO
        """ 
        # Initial conditions
        def set_ic():
            # TODO: add stochasticity
            # self.T0 = np.exp(-np.power((self.zgrid-0.5), 2)/(2.*0.25**2))
            self.T0 = -4.*np.power((self.zgrid-0.5),2.)+1.

        if self.random_ic:
            set_ic()
        else:
            # Set fixed initial conditions only during initialization
            try:
                self.T0
            except:
                set_ic()
            pass

        return np.ones(1), self.T0


    def sample_tempVel(self, T, wT_var=0.0005, lengthscale=0.3, order=1):
        """
        Args:
            T np.array(n_zgrid): Temperature profile at time t
            wT_var float: Variance of temperature-velocity
            lengthscale float: Kernel lengthscale
            order float: Kernel order
        Returns: 
            wT np.array(n_zgrid): Temperature-velocity profile at time t 
        """
        # TODO: this currently assumes that each time-step is independent!

        wT_mean = np.cos(np.sin(np.power(T,3.))) + np.sin(np.cos(np.power(T,2))) # dim: n_zgrid

        wT_cov = densities.power_kernel(self.zgrid, var_y=wT_var, lengthscale=lengthscale, p=order)

        wT, exp_wT, _,_,_,_ = densities.sample_approx_gaussian_process(xgrid=self.zgrid, 
            mu_y=wT_mean, cov=wT_cov, expansion='KL', 
            kl_dim=3, plot=True)

        return wT

    def sample_vel(self, w_var=0.0, lengthscale=0.3, order=1):
        """
        Args:
            w_var float: Variance of vertical velocity
            lengthscale float: Kernel lengthscale
            order float: Kernel order
        Returns: 
            w np.array(n_zgrid): vertical velocity profile at time t 
        """
        # TODO: this currently assumes that each time-step is independent!

        #w_mean = np.cos(np.sin(np.power(self.zgrid,3.))) + np.sin(np.cos(np.power(self.zgrid,2))) # dim: n_zgrid
        y_mean = np.log(100) * np.ones(self.zgrid.shape[0]) 
        #y_mean = -self.zgrid/6.

        if w_var>0:
            y_cov = densities.power_kernel(self.zgrid, var_y=w_var, lengthscale=lengthscale, p=order)

            w, exp_w, _,_,_,_ = densities.sample_approx_gaussian_process(xgrid=self.zgrid, 
                mu_y=y_mean, cov=y_cov, expansion='KL', 
                kl_dim=5, plot=True)
        else:
            w = y_mean
            exp_w = np.exp(y_mean)
        if self.plot:
            plotting.plot_w(self.zgrid, exp_w)
        return exp_w

    def step_rng(self):
        """ TODO: Check if i need to implement this.Iterate the random number generator 

        Iterates the random number generator, s.t., parallel processes have different output
        """
        raise NotImplementedError('step_rng not implemented.')
        self.rng.uniform(0, 1, size=(1))
        return 0


    def sample_diffusivity(self, k_var=1., lengthscale=0.3, order=1, gp_stoch_dim=3):
        """
        Args:
            k_var float: Variance of temperature-velocity
            lengthscale float: Kernel lengthscale
            order float: Kernel order
            gp_stoch_dim int: Dimensionality of Gaussian process approximation
        Returns: 
            k np.array(n_zgrid): diffusivity profile at time t 
            rand_insts np.array(gp_stoch_dim): Random instances used in approximation of GP 
        """
        # TODO: this currently assumes that each time-step is independent!

        #w_mean = np.cos(np.sin(np.power(self.zgrid,3.))) + np.sin(np.cos(np.power(self.zgrid,2))) # dim: n_zgrid
        y_mean = np.log(1000) * np.ones(self.zgrid.shape[0]) 
        #y_mean = np.log(100.)*np.cos(self.zgrid*2.*np.pi)
        #y_mean = -(self.zgrid-60) / 6.
        y_cov = densities.power_kernel(self.zgrid, var_y=k_var, lengthscale=lengthscale, p=order)

        # Sample log-diffusivity, Y, assuming it's a Gaussian process
        if self.sample_ln_diffusivity is None:
            # Calling sample_approx_gaussian_process computes the PCE-expansion during first call
            Y, exp_Y, _, coefs, rand_insts, self.sample_ln_diffusivity = densities.sample_approx_gaussian_process(
                xgrid=self.zgrid, mu_y=y_mean, cov=y_cov,
                kl_dim=gp_stoch_dim, pce_dim=gp_stoch_dim, 
                expansion='polynomial_chaos', plot=False)
        else:
            Y, exp_Y, _, coefs, rand_insts = self.sample_ln_diffusivity()

        k = exp_Y  
        if self.plot:
            plotting.plot_k_diff(self.zgrid, k)
        return k, rand_insts

    def step(self, T, t, w, k, f, verbose=False):
        """
        Computes 1st time derivative of T
        Args:
            T np.array(n_zgrid): Temperature in K
            t float: time in seconds
            w np.array(n_zgrid): vertical velocity in m/s
            k np.array(n_zgrid): thermal diffusivity in m^2/s
            f np.array(n_zgrid): source in K
        Returns:

        """
        if verbose: print(t, T)

        T_x = np.gradient(T) # Uses 2nd order central diff in interior and forw/backw at boundaries
        kT_xx = np.gradient(np.multiply(k,T_x))

        wT = np.multiply(w, T)
        wT_x = np.gradient(wT)

        T_t = -wT_x + kT_xx + f
        
        return T_t 

    def torch_1st_order_forw_diff(self, t, axis=0):
        """
        Calculates first order forward difference along given axis
        # TODO make more beautiful and compute 2nd order diff
        t np.array(n_tgrid, n_xgrid): Input array; only tested with 2D
        """
        if axis==0:
            diff = F.pad(t,(0,0,-1,1)) - t # 1st order forward difference 
            diff[-1,:] = t[-1,:] - t[-2,:] # 1st order backward difference at last element
        elif axis==1:
            diff = F.pad(t,(-1,1,0,0)) - t # 1st order forward difference 
            diff[:,-1] = t[:,-1] - t[:,-2] # 1st order backward difference at last element
        else:
            raise NotImplementedError
        return diff

    def residual(self, T, u_params, w=1, k=1, f=0):
        """
        Computes squared residual of proposed solution
        Args:
            T torch.tensor(n_tgrid, n_zgrid): Temperature profile
            # xt domain
            u_params dict(param: np.array(n_tgrid, n_zgrid)): Parameters of each differential equation sample; only used for PINN loss
            w np.array(n_zgrid): vertical velocity in m/s
            k np.array(n_zgrid): thermal diffusivity in m^2/s
            f np.array(n_zgrid): source in K            
        """
        # TODO: implement with torch autograd. 
        # import torch.autograd as autograd
        #print(T.shape, xt.shape)
        #print(autograd.grad(T[:], xt, retain_graph=True).shape)
        #print(autograd.grad(T[:], xt, retain_graph=True)[0].data)
        #torch.autograd.grad(output[:, i].sum(), input, retain_graph=True)[0].data
        n_tgrid, n_xgrid = u_params['rand_param'].shape
        k = torch.tensor(u_params['rand_param']) 
        # TODO: pass parameters into function from u_params
        w = torch.tensor(np.repeat(self.w[np.newaxis,:], repeats=n_tgrid, axis=0))
        f = torch.tensor(np.repeat(self.f[np.newaxis,:], repeats=n_tgrid, axis=0))

        T_z = self.torch_1st_order_forw_diff(T, axis=1)
        kT_z = torch.mul(k, T_z)
        kT_zz = self.torch_1st_order_forw_diff(kT_z, axis=1)

        wT = torch.mul(w, T)
        wT_z = self.torch_1st_order_forw_diff(wT, axis=1)

        T_t = self.torch_1st_order_forw_diff(T, axis=0)

        res = -wT_z + kT_zz + f - T_t
        # res = torch.nan_to_num(res)

        return res

    def solve(self, solver_name=None):
        """
        Solves the local adv. diffusion equation

        Args:
            None
        Returns:
            Tsol np.array((n_tgrid, n_zgrid)): Temperature solution
        """
        if solver_name is None or solver_name=='explicit':
            dt = self.tgrid[-1] - self.tgrid[-2]
            Tsol = np.zeros((self.tgrid.shape[0], self.zgrid.shape[0]))
            Tsol[0,:] = self.T0
            for i, t in enumerate(self.tgrid[1:]):
                if i==self.tgrid.shape[0]:
                    break
                Tsol[i+1,:] = Tsol[i,:] + dt * self.step(Tsol[i,:], t, self.w, self.k, self.f)

                # Dirichlet boundary conditions
                # TODO: define BCs in better way.
                Tsol[i+1, 0] = 0.
                Tsol[i+1, -1] = 0.
        elif solver_name=='odeint':
            Tsol = odeint(func=self.step, y0=self.T0, t=self.tgrid, args=(self.w, self.k, self.f), printmessg=self.verbose)
            # Todo: implement BCs        
        else:
            raise NotImplementedError(f'Solver {solver_name} not implemented for Loca Adv Diff Eq')

        if self.plot:
            plotting.plot_2d_sol(self.zgrid, self.tgrid, Tsol)
        return Tsol

def reshape_localAdvDiff_to_nn(Tsol, tgrid, zgrid, rand_insts):
    """
    Reshapes localAdvDiff solution, Tsol, and grid into flexible NN in-/output

    Args:
        u_params dict(
            'rand_param': np.array((n_samples_after_warmup, n_tgrid, n_zgrid)): Random parameter, here, k
            )
        Tsol np.array((n_samples, n_tgrid, n_zgrid))
        tgrid np.array((n_tgrid))
        zgrid np.array((n_zgrid))
        rand_insts np.array((n_samples, LocalAdvDiffEq.gp_stoch_dim)): 
    Returns:
        Tsol np.array(n_samples, n_tgrid, n_zgrid, 1)
        grid np.array(n_samples, n_tgrid, n_zgrid, 2)
        rand_insts np.array(n_samples, n_tgrid, n_zgrid, LocalAdvDiffEq.gp_stoch_dim)
    """
    n_samples = Tsol.shape[0]

    # Reshape grid
    n_tgrid = tgrid.shape[0]
    n_zgrid = zgrid.shape[0]
    # Create flattened meshgrid over spatio-temporal domain t, x
    tt, zz = np.meshgrid(tgrid, zgrid, indexing='ij')
    grid = np.concatenate((tt[:,:, np.newaxis], zz[:,:, np.newaxis]), axis=2) # dim: n_tgrid, n_zgrid, dim_grid
    grid = np.repeat(grid[np.newaxis,:,:,:], repeats=n_samples, axis=0) # dim: n_samples, n_tgrid, n_zgrid, dim_grid

    # Reshape solution
    Tsol = Tsol[:,:,:,np.newaxis]

    # Reshape rand_insts
    rand_insts = np.repeat(rand_insts[:,np.newaxis,:], n_tgrid, axis=1)
    rand_insts = np.repeat(rand_insts[:,:,np.newaxis,:], n_zgrid, axis=2)

    return Tsol, grid, rand_insts

if __name__ == "__main__":
    """
    Create plot of test local advection diffusion equation. See plotting.plot_2d_sol for directory
    """
    parser = argparse.ArgumentParser(description='Local Adv. Diff.')
    args = parser.parse_args()  

    seed = 1
    np.random.seed(seed)

    # Define grid
    n_tgrid = 128
    n_xgrid = 16
    xmax = 1.
    tmax = .01

    #if args.debug:
    #    n_tgrid, n_xgrid, tmax, xmax = np.multiply(.25, (n_tgrid, n_xgrid, tmax, xmax))
    #    args.nsamples = 2
    tgrid = np.linspace(0., tmax, int(n_tgrid)) # 1.5, n_tgrid)
    xgrid = np.linspace(0., xmax, int(n_xgrid)) # 30, n_xgrid)

    # Init stochastic differential equation
    localAdvDiffEq = LocalAdvDiffEq(tgrid=tgrid, zgrid=xgrid, plot=True)
    print(f'Diffusivity {np.mean(localAdvDiffEq.k):.6f}+-{np.std(localAdvDiffEq.k):.6f}, vertical velocity {np.mean(localAdvDiffEq.w):.6f}')

    sol = localAdvDiffEq.solve()
