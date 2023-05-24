import math
import argparse
import numpy as np
import torch
import h5py

from pce_pinns.utils.shaping import split_sequences_into_snippets
from pce_pinns.solver.diffeq import DiffEq

from torchqg.qg import to_spectral, to_physical, QgModel, QgModelPureML
from torchqg.sgs import Constant
import torchqg.workflow as workflow

class QgturbEq(DiffEq):
    def __init__(self, param='null', 
      init_data_path=None,
      init_tstep=-1):
        """
        Args:
          init_data_path str: Path with data from which to load initial state
          init_tstep int: idx of timestep within init_data that should be used as initial state.
            if -1 takes last timestep. 
        """
        self.name = 'qgturb'

        def t_unit():
          # T_d(x) = 1.2 x 10^6/pi s
          return 1.2e6
        self.t_unit = t_unit

        def l_unit():
          # L_d(x) = 504 x 10^4/pi m
          return (504e4 / math.pi) 

        self.Lx = 2*math.pi
        self.Ly = 2*math.pi
        self.Nx = 512
        self.Ny = 512

        self.dt = 480 / self.t_unit() # 480s
        self.mu = 1.25e-8 / l_unit()**(-1) # 1.25e-8m^-1
        self.nu = 352 / l_unit()**2 / self.t_unit()**(-1) # 352m^2/s for (512^2) and 22m^2s^-1 for (2048^2)

        eta = torch.zeros([self.Ny, self.Nx], dtype=torch.float64, requires_grad=True)

        # High res model.
        self.h = QgModel(
          name='\\mathcal{F}',
          Nx=self.Nx,
          Ny=self.Ny,
          Lx=self.Lx,
          Ly=self.Ly,
          dt=self.dt,
          t0=0.0,
          B=0.0,    # Planetary vorticity y-gradient, aka, Rossby parameter?
          mu=self.mu,    # Linear drag coefficient
          nu=self.nu,    # Viscosity coefficient
          nv=1,     # Hyperviscous order (nv=1 is viscosity)
          eta=eta,  # Topographic PV
          source=self.Fs # Source term
        )

        # Initial conditions.
        self.init_dns(init_data_path, init_tstep)

        # Set up spectral filter kernel.
        self.h.kernel = self.h.grid.cutoff

        print(self.h)

        self.les_models = []
        self.param = param
        self.y_param_mode = 'full_subgrid_forcing'
        sgs_null = Constant(c=0.0)
        self.init_les(sgs=sgs_null, name='null') # Init LES models with null parametrization

    def init_dns(self, init_data_path=None, init_tstep=-1):
        """
        Initializes DNS model
        """
        if init_data_path is not None:
            filename = init_data_path + '.h5'
            hf = h5py.File(filename, 'r')

            init_tstep = hf.get('time').shape[0] - 1 if init_tstep == -1 else init_tstep
            ft = hf.get('time')[init_tstep]
            fq = hf.get('dns_q')[init_tstep]
            
            fq = torch.from_numpy(fq)
            fqh = to_spectral(fq)

            self.h.pde.cur.t = ft
            self.h.pde.sol = fqh
        else:
            self.h.init_randn(0.01, [3.0, 5.0])
        return 1

    def init_les(self, sgs, name=''):
        """
        Initializes LES model(s)
        Args:
          sgs: torchqg.sgs, e.g., MNOparam
        """
        # Low res model(s).
        self.scale = 4

        Nxl = int(self.Nx / self.scale)
        Nyl = int(self.Ny / self.scale)

        eta_m = torch.zeros([Nyl, Nxl], dtype=torch.float64, requires_grad=True)
        
        # Init parametrization
        if sgs is None:
          # Default null parametrization
          sgs = Constant(c=0.0)

        # No model.
        m1 = QgModel(
          name=name,
          Nx=Nxl,
          Ny=Nyl,
          Lx=self.Lx,
          Ly=self.Ly,
          dt=self.dt*self.scale,
          t0=0.0,
          B=0.0,     # Planetary vorticity y-gradient
          mu=self.mu,     # Linear drag
          nu=self.nu,     # Viscosity coefficient
          nv=1,      # Hyperviscous order (nv=1 is viscosity)
          eta=eta_m, # Topographic PV
          source=self.Fs, # Source term
          sgs=sgs # Subgrid-scale term (replace with yours)
        )

        # Initialize from DNS vorticity field.
        m1.pde.sol = self.h.filter(m1.grid, self.scale, self.h.pde.sol)
        print('LES Model: ', m1)

        self.les_models.append(m1)

    def init_pure_ml(self, model, name=''):
      """
      Initializes LES that uses ML model to forecast large-scale dynamics
      Args:
        model torch.model
      """
      # Low res model(s).
      self.scale = 4
      Nxl = int(self.Nx / self.scale)
      Nyl = int(self.Ny / self.scale)
      
      m_pure_ml = QgModelPureML(
        name=name,
        Nx=Nxl,
        Ny=Nyl,
        Lx=self.Lx,
        Ly=self.Ly,
        dt=self.dt,
        t0=0.0,
        model=model,
        
      )

      # Initialize from DNS vorticity field.
      m_pure_ml.pde.sol = self.h.filter(m_pure_ml.grid, self.scale, self.h.pde.sol)
      print('LES Model: ', m_pure_ml)

      self.les_models.append(m_pure_ml)
    def step(self):
        pass

    def solve(self,dir='data/raw/temp/qgturb/',
      name='geo', iters=100, store_iters=100, diags=[workflow.diag_fields]):
        # Will produce two images in folder `dir` with the final fields after <iters> iterations.
        workflow.workflow(
          dir=dir,
          name=name,
          iters=iters, # 10000,  # Model iterations; 6.15min/1K iters on 1CPU
          steps=store_iters, # Total number of iterations that will be saved
          scale=self.scale,  # Kernel scale
          diags=diags, # Diagnostics
          system=self.h,       # DNS system
          models=self.les_models, # LES models; or empty []
          dump=True
        )

    def residual(self):
        """
        Computes autodifferentiable squared residual of proposed solution.
        """
        pass

    def sample_random_ic(self, init=False, init_ones=False):
        """ Dummy
        """ 
        raise NotImplementedError('qgturb.QgturbEq.sample_random_ic not implemented')
        return np.ones(1, dtype=self.dtype), np.ones(1,dtype=self.dtype)

    def sample_random_params(self):
        """ Dummy
        """
        raise NotImplementedError('qgturb.QgturbEq.sample_random_params not implemented')
        return np.ones(1,dtype=self.dtype), np.ones(1,dtype=self.dtype)

    # Wind stress forcing.
    def Fs(self, i, sol, dt, t, grid):
      """
      Note: Forcing is independent of time-step dt. Only depends on t_unit. 
      """
      phi_x = math.pi * math.sin(1.2e-6 / self.t_unit()**(-1) * t)
      phi_y = math.pi * math.sin(1.2e-6 * math.pi / self.t_unit()**(-1) * t / 3)
      y = torch.cos(4 * grid.y + phi_y).view(grid.Ny, 1) - torch.cos(4 * grid.x + phi_x).view(1, grid.Nx)

      yh = to_spectral(y)
      K = torch.sqrt(grid.krsq)
      yh[K < 3.0] = 0
      yh[K > 5.0] = 0
      yh[0, 0] = 0

      e0 = 1.75e-18 / self.t_unit()**(-3)
      ei = 0.5 * grid.int_sq(yh) / (grid.Lx * grid.Ly)
      yh *= torch.sqrt(e0 / ei)
      return yh

def reshape_qgturb_to_snippets(tgrid, sol, u_args, n_tsnippet, n_snippets=None, sol_prior=None):
    """   
    Args:
        tgrid np.array((n_tgrid))
        sol np.array((n_tgrid, n_x1grid, n_x2grid, n_dims))
        u_args np.array((n_tgrid, n_x1grid, n_x2grid, n_dims))
        n_snippets int: Maximum number of total snippets. Only full-length snippets will be created. The rest of the data will be discarded. .
        n_tsnippet int: time length of individual snippet 
     
    Returns:
        tgrid_snippet np.array((n_tsnippet,))
        sol_snippets np.array((n_snippets, n_tsnippet, n_x1grid, n_x2grid, n_dims))
        u_args_snippets np.array((n_snippets, n_tsnippet, n_x1grid, n_x2grid, n_dims))
    """
    # Create new tgrid by taking first n_tsnippet values. This assumes that 
    # tgrid is periodic after n_tsnippet time steps
    tgrid_snippet = tgrid[:n_tsnippet]

    # Split sol and u_args of length n_tgrid into snippets of length n_tsnippet
    sol_snippets = split_sequences_into_snippets(sol[np.newaxis,...,0], n_snippets, n_tsnippet)[...,np.newaxis] # Init shape
    for i in range(sol.shape[-1]-1):
        sol_snippets[...,i] = split_sequences_into_snippets(sol[np.newaxis,...,i], n_snippets, n_tsnippet)
    u_args_snippets = split_sequences_into_snippets(u_args[np.newaxis,...,0], n_snippets, n_tsnippet)[...,np.newaxis] # Init shape
    for i in range(sol.shape[-1]-1):
        u_args_snippets[...,i] = split_sequences_into_snippets(u_args[np.newaxis,...,i], n_snippets, n_tsnippet)

    if sol_prior is not None:
        raise NotImplementedError('sol_prior argument not supported in reshape_qgturb_to_snippets')

    return tgrid_snippet, sol_snippets, sol_prior, u_args_snippets
    
def reshape_qgturb_to_fno2d(sol, tgrid, est_qgturb=None,
        sol_prior=None, u_args=None):
    """
    Reshapes QG turbulence from snippets to FNO2D input 
    Args:
        tgrid_snippet np.array((n_tsnippet,))
        sol_snippets np.array((n_snippets, n_tsnippet, n_x1grid, n_x2grid, n_dims))
        u_args_snippets np.array((n_snippets, n_tsnippet, n_x1grid, n_x2grid, n_dims))
    Returns:
        u_target np.array(n_samples, n_tgrid, n_x1grid, n_x2grid, 1)
        grid np.array(n_samples, n_tgrid, n_x1grid, n_x2grid, dim_grid): dim_grid = 1
        y_args np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, dim_y_args))
    """
    u_target = sol
    y_args = u_args
    grid = tgrid[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    grid = np.repeat(grid, axis=2, repeats=u_target.shape[2]) # n_x1grid
    grid = np.repeat(grid, axis=3, repeats=u_target.shape[3]) # n_x2grid

    return u_target, grid, y_args

def reshape_qgturb_to_nn_model(sol,
    tgrid, est_qgturb=None, 
    model_cfg={'type':'mno'}, sol_prior=None, 
    u_args=None, n_snippets=None,
    n_tsnippet=None):
    """
    Reshapes torchqg.workflow() output grid into the desired in-/output
    Args:
        sol np.array([steps, Nyl, Nyx, 1]): Parametrization
        tgrid np.array([steps, 1]): Temporal from filtered DNS
        u_args np.array([steps, Nyl, Nyx, 1]): Vorticity from filtered DNS 
    """
    # Align 
    if n_tsnippet is not None:
        tgrid, sol, sol_prior, u_args = reshape_qgturb_to_snippets(tgrid=tgrid, 
            sol=sol, sol_prior=sol_prior, u_args=u_args, n_snippets=n_snippets,
            n_tsnippet=n_tsnippet)

    # Select in-/output relation that shall be learned.
    if est_qgturb == "no-scale-sep-param-no-mem":
      # Estimates exact parametrization, r, from filtered DNS over the full state
      pass
    elif est_qgturb == "pure-ml-sol":
      # Target is the vorticity, but shifted forward by t_offset time steps along
      # the n_tsnippet dimension. The last time step is filled with all zeroes.
      # This is assuming that all steps within one snippet come from the same auto-
      # regressive process. Steps from distinct snippet can be generated by differing
      # processes.
      del sol # parametrization is not needed and can be discarded from memory.
      t_offset = 1
      sol = u_args.copy()
      sol = sol[:,t_offset:,...]
      u_blank = np.zeros(sol[:,:t_offset,...].shape, dtype=u_args.dtype)
      sol = np.concatenate((sol, u_blank), axis=1)
      # Input is the vorticity, but last time step filled with zeros.
      u_args = u_args[:,:-t_offset,...]
      u_args = np.concatenate((u_args, u_blank), axis=1)

    # Select function to reshape.  
    if model_cfg['type'] == 'mno':
        reshape_fn = reshape_qgturb_to_fno2d
    elif model_cfg['type'] == 'fno-pure-ml-sol':
        reshape_fn = reshape_qgturb_to_fno2d
    elif model_cfg['type'] == 'poly':
        reshape_fn = reshape_qgturb_to_poly
    elif model_cfg['type'] == 'climatology':
        reshape_fn = reshape_qgturb_to_nn
    else:
        raise NotImplementedError('Model type {:s} not implemented. '\
            'Use e.g., "fcnn" instead.'.format(model_cfg['type']) )

    u_target, tgrid, y_args = reshape_fn(sol=sol,
        tgrid=tgrid, est_qgturb=est_qgturb,
        sol_prior=sol_prior, u_args=u_args)

    return u_target, tgrid, y_args


if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser(description='Quasi-geostrophic turbulence')
    args = parser.parse_args()  

    qgturbEq = QgturbEq()

    qgturbEq.solve()
