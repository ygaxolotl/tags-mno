import numpy as np

import pce_pinns.utils.plotting as plotting
from pce_pinns.solver.diffeq import DiffEq

class StochDiffEq(DiffEq):
    def __init__(self, xgrid, rand_flux_bc=False, injection_wells=False):
        """
        Sets up 1D stochastic diffusion eqn's initial conditions and parameters
        
        -s = d/dx(k(x;w) du(x;w)/dx)

        Args:
            xgrid np.array(n_grid): Grid points
            rand_flux_bc bool: If true, use random flux for van neumann condition on left boundary
            inverse_problem bool: If true, sets source term to be a function  

        Attributes:
            flux function()->float: Flux at left-hand boundary, k*du/dx = -F; could be stochastic or deterministic  
            source np.array(n_grid) or float: Source term, either a vector of values at points in xgrid or a constant
            rightbc float: Dirichlet BC on right-hand boundary
        """
        self.rightbc = 1.
        self.flux = lambda: -1. 
        self.source = 5.
        if rand_flux_bc:
            mu_f = -1. #-2.
            var_f = .2 #.5
            self.flux = lambda: np.random.normal(mu_f, var_f, 1) # 
        elif injection_wells:
            self.source = injection_wells(xgrid, n_wells=4, strength=0.8, width=0.05)

    def sample_random_params(self):
        """ Sample random parameters

        Sets all stochastic parameters to a new sample

        Returns:
            k_rand_insts np.array(gp_stoch_dim): Random instances used in approximation of GP 
            rand_param np.array(): Random parameter
        """
        raise NotImplementedError("todo")

        return 0, 0

    def solve(self, xgrid, k):
        """
        Solves 1-D diffusion equation with given diffusivity field k
        and left-hand flux F. Domain is given by xgrid (should be [0,1])
        
        Args:
            xgrid np.array(n_grid): Grid points
        
        Returns:
            usolution np.array(xgrid): Solution
        """
        # Sample stochastic parameters
        F = self.flux() 

        N = xgrid.shape[0] # Number of grid points
        h = xgrid[N-1]-xgrid[N-2] # step size; assuming uniform grid

        # Set up discrete system f = Au + b using second-order finite difference scheme
        A = np.zeros((N-1, N-1)) 
        b = np.zeros((N-1,1)) 
        if np.isscalar(self.source): 
            f = -self.source * np.ones((N-1,1))
        else:
            f = -self.source[:N-1,np.newaxis] # [:N] 

        # diagonal entries
        A = A - 2.*np.diag(k[:N-1]) - np.diag(k[1:N]) - np.diag(np.hstack((k[0], k[:N-2]))) 

        # superdiagonal
        A = A + np.diag(k[:N-2], 1) + np.diag(k[1:N-1], 1) 

        # subdiagonal
        A = A + np.diag(k[:N-2], -1) + np.diag(k[1:N-1], -1)

        A = A / (2. * np.power(h,2))

        # Treat Neumann BC on left side
        A[0,1] = A[0,1] + k[0] / np.power(h,2)
        b[0] = 2.*F / h # b(1) = 2*F/h;

        # Treat Dirichlet BC on right side
        b[N-2] = self.rightbc * (k[N-1] + k[N-2]) / (2.*np.power(h,2)) 

        # Solve it: Au = f-b
        uinternal = np.linalg.solve(A, (f-b))

        usolution = np.vstack((uinternal, self.rightbc)) 

        return usolution

def injection_wells(xgrid, n_wells=4, strength=0.8, width=0.05, plot=True):
    """
    Returns a function over xgrid that models injection wells
    
    Args: 
        xgrid np.array(n_grid): Grid
        n_wells int: Number of equidistantly placed wells
        strength float: Source strength, theta
        width float: Source width, delta
    
    Returns:
        source np.array(n_grid): Function that models injection wells
    """
    n_grid=xgrid.shape[0]

    pos_wells = np.linspace(xgrid.min(), xgrid.max(), num=n_wells+2) # equidistant wells
    pos_wells = pos_wells[1:-1] # remove wells at borders
    pos_wells = np.repeat(pos_wells[np.newaxis,:], repeats=n_grid,axis=0) # repeat for matrix math
    xgrid_w = np.repeat(xgrid[:,np.newaxis], repeats=n_wells,axis=1) # repeat for matrix math
    
    amp = strength / (width * np.sqrt(2.*np.pi)) # amplitude
    source = np.sum(amp * np.exp(-(xgrid_w - pos_wells)**2 / (2.*width**2)),axis=1) # sum over wells

    if plot:
        plotting.plot_source(xgrid, source)

    return source
