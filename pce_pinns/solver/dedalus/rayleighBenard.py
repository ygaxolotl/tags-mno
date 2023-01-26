"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
This script demonstrates solving a 2D cartesian initial value problem. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_snapshots.py` script can be used to
produce plots from the saved data. It should take about a cpu-minute to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""
import os
import time
import timeit
import h5py
import argparse
import pickle
from pathlib import Path
import numpy as np
from numpy.random import default_rng
from skimage.measure import block_reduce

import dedalus.public as d3
import dedalus.core as core
from dedalus.core.timesteppers import RK222
from dedalus.core.coords import CartesianCoordinates
from dedalus.core.basis import RealFourier, ChebyshevT
from dedalus.core.distributor import Distributor
from dedalus.core.operators import Lift, grad, div, skew
from dedalus.core.problems import IVP
from dedalus.tools import post
    
from pce_pinns.solver.diffeq import DiffEq
from pce_pinns.utils import plotting

import logging
logger = logging.getLogger(__name__)

class RayleighBenardGrid:
    def __init__(self,
        Nx=64, Nz=16,
        dtype=np.float64):

        # Parameters
        self.Nx, self.Nz = Nx, Nz
        self.Lx, self.Lz = 4, 1 # Domain size
        self.dealias = 3/2
        self.dtype = dtype

        # Bases
        self.coords = CartesianCoordinates('x', 'z')
        self.dist = Distributor(self.coords, dtype=self.dtype)
        self.xbasis = RealFourier(self.coords['x'], size=self.Nx, bounds=(0, self.Lx), dealias=self.dealias)
        self.zbasis = ChebyshevT(self.coords['z'], size=self.Nz, bounds=(0, self.Lz), dealias=self.dealias)

    def get_grid():
        return self

class RayleighBenardEq(): # DiffEq
    def __init__(self, grid, 
        random_ic=True, 
        dummy_init=False,
        plot=False, 
        tgrid_warmup=None, 
        seed=0,
        fixed_timestep=0.01,
        stop_sim_time=50,
        dtype=np.float64,
        dir_hr=None,
        Nx_hr=None,
        Nz_hr=None):
        """
        Sets up rayleigh benard equation as initial value problem. The equation
        models chaotic 2D convection and is based on 
        https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_rayleigh_benard.html 

        Args:
            grid : Grid 
            param string: Type of subgrid parametrization, e.g., 'null' or 'fully_resolved'
            y_param_mode string: Mode of learned subgrid parametrization for y, e.g., 'superparam'
            random_ic bool: If true, use random initial conditions
            dummy_init bool: If true, uses computationally fast dummy initialization
            plot bool: If true, generates plots
            tgrid_warmup np.array(n_warmup): Grid points in time used to warmup equation, i.e., reach the attractor 
            fixed_timestep float: If not None, used instead of CFL condition
            dtype np.dtype: Datatype for storing Lorenz96 solution
            seed int: Seed for random number generator
            dir_hr pathlib.Path: If not None, solver updates state with HR data on every time step
        Attributes:
        """
        # Grid
        self.grid = grid
        Lx, Lz = grid.Lx, grid.Lz
        self.n_tsnippet = 50 # Number of time steps in one snippet
        self.fixed_timestep = fixed_timestep
        self.stop_iteration = None

        # Random 
        self.seed = seed
        self.rng = default_rng(self.seed)

        # Parameters
        self.Rayleigh = 2e6
        self.Prandtl = 1
        self.stop_sim_time = stop_sim_time
        self.max_timestep = 0.125
        self.timestepper = RK222
        self.dtype = dtype

        # Fields
        p = self.grid.dist.Field(name='p', bases=(self.grid.xbasis,self.grid.zbasis))
        self.b = self.grid.dist.Field(name='b', bases=(self.grid.xbasis,self.grid.zbasis))
        self.u = self.grid.dist.VectorField(self.grid.coords, name='u', bases=(self.grid.xbasis,self.grid.zbasis))
        tau_p = self.grid.dist.Field(name='tau_p')
        tau_b1 = self.grid.dist.Field(name='tau_b1', bases=self.grid.xbasis)
        tau_b2 = self.grid.dist.Field(name='tau_b2', bases=self.grid.xbasis)
        tau_u1 = self.grid.dist.VectorField(self.grid.coords, name='tau_u1', bases=self.grid.xbasis)
        tau_u2 = self.grid.dist.VectorField(self.grid.coords, name='tau_u2', bases=self.grid.xbasis)

        # Set multiscale parameters (for MNO paper)
        self.dir_hr = dir_hr
        self.Nx_hr = Nx_hr
        self.Nz_hr = Nz_hr
        self.lr_factor = 2
        # Indices to grab stored data. Todo: make more beautiful 
        self.buoyancy_idx = 0
        self.velocity_idx = [1,2]
        self.param = None # Storage for subgrid parametrization
        if self.dir_hr is not None:
            # Storage for coarsened HR data
            self.coarse_hr_np = self.init_coarse_hr() 

        # Substitutions
        kappa = (self.Rayleigh * self.Prandtl)**(-1/2)
        nu = (self.Rayleigh / self.Prandtl)**(-1/2)
        # x, z = self.grid.dist.local_self.grids(self.grid.xbasis, self.grid.zbasis)
        x, z = self.grid.dist.local_grids(self.grid.xbasis, self.grid.zbasis)
        ex, ez = self.grid.coords.unit_vector_fields(self.grid.dist)
        lift_basis = self.grid.zbasis.derivative_basis(1)
        lift = lambda A: Lift(A, lift_basis, -1)
        grad_u = grad(self.u) + ez*lift(tau_u1) # First-order reduction
        grad_b = grad(self.b) + ez*lift(tau_b1) # First-order reduction
        self.z = z

        # Problem
        # First-order form: "div(f)" becomes "trace(grad_f)"
        # First-order form: "lap(f)" becomes "div(grad_f)"
        self.problem = IVP([p, self.b, self.u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
        self.problem.add_equation("trace(grad_u) + tau_p = 0")
        self.problem.add_equation("dt(self.b) - kappa*div(grad_b) + lift(tau_b2) = - self.u@grad(self.b)")
        self.problem.add_equation("dt(self.u) - nu*div(grad_u) + grad(p) - self.b*ez + lift(tau_u2) = - self.u@grad(self.u)")
        self.problem.add_equation("self.b(z=0) = Lz")
        self.problem.add_equation("self.u(z=0) = 0")
        self.problem.add_equation("self.b(z=Lz) = 0")
        self.problem.add_equation("self.u(z=Lz) = 0")
        self.problem.add_equation("integ(p) = 0") # Pressure gauge

        # Solver
        self.solver = self.problem.build_solver(self.timestepper)
        self.solver.stop_sim_time = self.stop_sim_time
        if self.stop_iteration is not None:
            self.solver.stop_iteration = self.stop_iteration

        # Set initial condition
        self.dummy_init = dummy_init
        self.random_ic = random_ic
        self.sample_random_ic(init=True, init_ones=self.dummy_init)
        self.sample_random_params()

        # Init data storage
        if self.fixed_timestep:
            write_dt = self.fixed_timestep # record every tstep
        else:
            write_dt = 0.25
        data_dir = Path(f"data/raw/temp/rayleighBenard/"\
            f"evolution_{self.grid.Nx:04}_{self.grid.Nz:04}/snapshots")
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        snapshots = self.solver.evaluator.add_file_handler(
                data_dir, sim_dt=write_dt, max_writes=self.n_tsnippet)
        snapshots.add_task(self.b, layout='g', name='buoyancy')
        # snapshots.add_task(-div(skew(u)), name='vorticity')
        snapshots.add_task(self.u, layout='g', name='velocity')
        ## HOW TO STORE u_x and u_y separately?

        # CFL
        self.fixed_timestep = fixed_timestep
        self.CFL = d3.CFL(self.solver, initial_dt=self.max_timestep, 
                cadence=10, safety=0.5, threshold=0.05,
                max_change=1.5, min_change=0.5, max_dt=self.max_timestep)
        self.CFL.add_velocity(self.u)

        # Flow properties
        self.flow = d3.GlobalFlowProperty(self.solver, cadence=10)
        self.flow.add_property(np.sqrt(self.u@self.u)/nu, name='Re')
    
    def sample_random_ic(self, init=True, init_ones=False):
        """ Sample random initial condition

        Sets a new initial conditions to a new sample
    
        Args:
            init_ones bool: See Lorenz96Eq()
        Returns:
            rand_insts np.array(stoch_dim): Random instances
            rand_ic : Random initial conditions
        """ 
        # Initial conditions
        def set_ic():
            # Initial conditions
            # todo: replace sampler with self.rng 
            # self.rng.normal(-15, 15, size=(self.K,)).astype(self.dtype)
            self.b.fill_random('g', seed=self.seed, distribution='normal', scale=1e-3) # Random noise
            self.b['g'] *= self.z * (self.grid.Lz - self.z) # Damp noise at walls
            self.b['g'] += self.grid.Lz - self.z # Add linear background

        if self.random_ic and not init:
            set_ic()
        elif init and not init_ones:
            set_ic()
        # Initalize with all 1 array to avoid comp. cost of random sampling during runtime msmt
        elif init_ones:      
            self.b['g'] = self.grid.z
        else: # TODO: get rid of this else case.
            set_ic()

        return np.ones(1, dtype=self.dtype), (self.b['g'])

    def get_timestep(self):
        """ return timestep
        """
        if self.fixed_timestep is not None:
            timestep = self.fixed_timestep
        else:
            timestep = self.CFL.compute_timestep() # min ~ 0.01
        return timestep

    def step(self, timestep):
        self.solver.step(timestep)
        return 1

    def init_coarse_hr(self):
        # Init auxiliary vector fields. Used to transform coarse_hr from grid into coefficient space.
        self.u_aux = self.grid.dist.VectorField(self.grid.coords, name='u_aux', bases=(self.grid.xbasis,self.grid.zbasis))
        self.b_aux = self.grid.dist.Field(name='b_aux', bases=(self.grid.xbasis,self.grid.zbasis))

        # Create coarsened HR data
        hr_filename = self.dir_hr / f'evolution_{self.Nx_hr:04}_{self.Nz_hr:04}/' / 'data.npy'
        hr_data = np.load(hr_filename)
        self.coarse_hr_np = self.coarse_grain(hr_data)
        
        # TODO: set initial time of solver based on HR data.
        self.stop_iteration = self.coarse_hr_np.shape[1]-1
        return self.coarse_hr_np

    def coarse_grain(self, hr):
        """
        Args:
            np.array(n_samples, n_tgrid, Nx, Nz, n_xdim)
        Returns
            np.array(n_samples, n_tgrid, Nx / lr_factor, Nz / lr_factor, n_xdim)
        """
        lr = block_reduce(hr, (1, 1, self.lr_factor,self.lr_factor, 1), np.mean)
        return lr

    def step_large_scale(self, sol_t, param_t):
        """
        Steps large-scale model with predicted subgrid forcing.
        -->In torch<--
        \bar u_t+1 = N(\bar u_t) + h_t

        Args:
            sol_t torch.Tensor(n_x1grid, n_x2grid, n_xdim): Large-scale variables, \bar u_t
            param_t torch.Tensor(n_x1grid, n_x2grid, n_xdim): Subgrid forcing from fine-scale variables, h_t
        Returns:
            xnext torch.Tensor(n_x1grid, n_x2grid, n_xdim): Next timestep's large-scale variable
        """
        pass

    def step_large_scale_np(self, sol_t, param_t):
        """
        Steps large-scale model with predicted subgrid forcing.
        -->In torch<--
        \bar u_t+1 = N(\bar u_t) + h_t

        Args:
            sol_t np.array(n_x1grid, n_x2grid, n_xdim): Large-scale variables, \bar u_t
            param_t np.array(n_x1grid, n_x2grid, n_xdim): Subgrid forcing from fine-scale variables, h_t
        Returns:
            sol_tnext np.array(n_x1grid, n_x2grid, n_xdim): Next timestep's large-scale variable
        """
        self.update_state(sol_t)
        
        # Step LR large-scale
        self.step(self.get_timestep())

        # Calculate next large-scale state 
        self.b_aux['c'] = self.b['c']
        self.u_aux['c'] = self.u['c']
        b_tnext = self.b_aux['g'] + param_t[...,self.buoyancy_idx]
        u_tnext = np.moveaxis(self.u_aux['g'],0,-1) + param_t[...,self.velocity_idx]
        sol_tnext = np.concatenate((b_tnext[...,None], u_tnext), axis=-1)

        return sol_tnext        

    def load_mno_train_data(self,
        dir_processed=Path(f"data/processed/temp/rayleighBenard/mno/")):
        """
        returns:
            sol_target np.array(n_tgrid, n_x1grid, ..., n_xngrid, n_xdim): Target coarse-grained HR Solution 
            param np.array(n_tgrid, n_x1grid, ..., n_xngrid, n_xdim): Full subgrid forcing parametrization
        """
        specifier = f'evolution_{self.grid.Nx:04}_{self.grid.Nz:04}'
        xtrain_filename = dir_processed / specifier / 'xtrain.npy'
        ytrain_filename = dir_processed / specifier / 'ytrain.npy'
        sol_target = np.load(ytrain_filename)
        param = np.load(xtrain_filename)
        return sol_target, param

    def test_full_subgrid_forcing(self, sol_target, param):
        """
        Tests accuracy of ground-truth solution vs. predicted parametrization 
        Args:
            sol_target np.array(n_tgrid, n_x1grid, ..., n_xngrid, n_xdim): Target coarse-grained HR Solution 
            param np.array(n_tgrid, n_x1grid, ..., n_xngrid, n_xdim): Full subgrid forcing parametrization
            # tgrid np.array(n_tgrid): Temporal grid
        Returns:
            success bool: True if reconstructed solution matches target solution
        """
        # Initialize arrayns
        sol_pred = np.zeros(param.shape, dtype=self.dtype)
        self.u_aux = self.grid.dist.VectorField(self.grid.coords, name='u_aux', bases=(self.grid.xbasis,self.grid.zbasis))
        self.b_aux = self.grid.dist.Field(name='b_aux', bases=(self.grid.xbasis,self.grid.zbasis))

        # Init condition
        sol_pred[0,...] = sol_target[0,...]

        # Feed-forward inference
        for t in range(1,sol_target.shape[0]):
            sol_pred[t,...] = self.step_large_scale_np(sol_pred[t-1,...], param[t-1,...])
            
            if (self.solver.iteration-1) % 10 == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(
                    self.solver.iteration, self.solver.sim_time, t, self.flow.max('Re')))

        success = np.all(sol_pred == sol_target)
        print('MAE over time:', np.mean(np.absolute(sol_pred-sol_target), axis=(1,2)))
        print('Reconstruction error MAE:', np.mean(np.absolute(sol_pred-sol_target)[:-2,...], axis=(0,1,2)))
        print('Target low-res solution matches reconstructed solution '\
            ' w parametrization target :', success)

        return success

    def update_state(self,
        new_state
        ):
        """
        Updates the current state
        
        Args:
            new_state np.array(n_x1grid, ..., n_xngrid, n_xdim)
        """
        # Update current state
        self.b_aux['g'] = new_state[...,self.buoyancy_idx]
        self.u_aux['g'] = np.moveaxis(new_state[...,self.velocity_idx], -1, 0)
        self.b.data = self.b_aux['c']
        self.u.data = self.u_aux['c']

        return 1

    def update_coarse_grained_state(self,
        t_idx=None,
        grid=None,
        sample_idx=0):
        """
        Args:
            t_idx: Time index of HR that has to match LR data 
            sample_idx: Index of HR data that has to match LR data
        Returns:
            Field
            Vectorfield    
        """
        # Select \bar u_t
        coarse_hr_t = self.coarse_hr_np[sample_idx,t_idx, ...]
        self.update_state(coarse_hr_t)

        return 1

    def calculate_subgrid_param(self, 
        tnext=None,
        sample_idx=0):
        """
        Calculates subgrid parametrization / commutation error 
        param_t = \bar u_{t+1} - N(\bar u_t)

        Args:
            tnext int: index to next time, t+1
            sample_idx: Index of HR data that has to match LR data
        """
        # Initialize logger
        if self.param is None:
            self.param = np.zeros((self.coarse_hr_np.shape))

        if tnext < self.solver.stop_iteration:
            coarse_hr_tnext = self.coarse_hr_np[sample_idx,tnext, ...]
        else: # Predict identity for the last element.  
            coarse_hr_tnext = self.coarse_hr_np[sample_idx,tnext-1,...]

        # Calculate subgrid parametrization error, param_t
        self.b_aux['c'] = self.b['c']
        self.u_aux['c'] = self.u['c']
        b_param_t = coarse_hr_tnext[...,self.buoyancy_idx] - self.b_aux['g'] 
        u_param_t = coarse_hr_tnext[...,self.velocity_idx] - np.moveaxis(self.u_aux['g'], 0,-1)

        # Log
        self.param[sample_idx, tnext-1, ..., self.buoyancy_idx] = b_param_t
        self.param[sample_idx, tnext-1, ..., self.velocity_idx[0]:self.velocity_idx[-1]+1] = u_param_t

        return 1

    def dump_mno_train_set(self,
        dir_processed=Path(f"data/processed/temp/rayleighBenard/mno/")):
        """
        Store dataset to train multiscale neural operators MNO: \bar u_t -> h_t
            s.t., \bar u_t+1 = N(\bar u_t) + h_t

        Args:
            dir_processed pathlib.Path: Parent directory for storing ML-ready files

        Stores:
            xtrain np.array(n_samples, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim): 
                Training input; current large-scale state \bar u_t
            ytrain np.array(n_samples, n_tgrid, n_x1grid, ..., n_xngrid, n_xdim):
                Training target; parametrization that captures influence of fine- onto large-scale 
                state: h_t = \bar u_{t+1} - N(\bar u_t)
        """
        specifier = f'evolution_{self.grid.Nx:04}_{self.grid.Nz:04}'
        target_dir = dir_processed / specifier

        xtrain_filename = target_dir / 'xtrain.npy'
        ytrain_filename = target_dir / 'ytrain.npy'

        print('Saving ML-ready data at ', target_dir)
        if not target_dir.exists():
            target_dir.mkdir(parents=True)
        np.save(xtrain_filename, self.param)
        np.save(ytrain_filename, self.coarse_hr_np)

        return 1

    def solve(self):
        # Main loop
        # warmup_iter = 10
        try:
            logger.info('Starting main loop')
            while self.solver.proceed:
                # Overwrite LR solver with coarse-grained HR data; load \bar u_t
                if self.dir_hr is not None:
                    self.update_coarse_grained_state(t_idx=self.solver.iteration)

                timestep = self.get_timestep()
                self.step(timestep)
                
                if (self.solver.iteration-1) % 10 == 0:
                    max_Re = self.flow.max('Re')
                    logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(
                        self.solver.iteration, self.solver.sim_time, timestep, max_Re))

                if self.dir_hr is not None:
                    self.calculate_subgrid_param(tnext=self.solver.iteration)
        except:
            logger.error('Exception raised, triggering end of main loop.')
            raise
        finally:
            self.solver.log_stats()
            if self.dir_hr is not None:
                self.dump_mno_train_set()

        return 1

    def sample_random_params(self):
        """ Sample random parameters
    
        Sets all stochastic parameters to a new sample

        Returns:
            rand_insts np.array(stoch_dim): Random instances
            rand_param np.array(?): Random parameters
        """
        return np.ones(1,dtype=self.dtype), np.ones(1,dtype=self.dtype)

def reshape_rayleighBenard_to_nn_dist(filename, start, count, output):
    tasks = ['buoyancy', 'velocity']

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                print(f'index: {index:04d}, file: {file}, task: {task}')
                dset = file['tasks'][task]
                return dset
                
def reshape_rayleighBenard_to_nn(
    dir_raw=Path(f"data/raw/temp/rayleighBenard/"),
    dir_interim = Path(f"data/interim/temp/rayleighBenard/"),
    snapshot_range=range(20,32),
    tasks=['buoyancy', 'velocity']):
    """
    Args:
        dir_raw pathlib.Path: Source directory for raw snapshot data, as recorded by Dedalus 
        dir_interim pathlib.Path: Target directory for processed snapshot data, in .npy 
        snapshot_range range: indices of snapshot files
        tasks [str]: Keys to recorded variables
    Stores and returns: 
        # grid np.array(n_snippets, n_tgrid, n_x1grid, ..., n_xngrid, n_dimgrd)
        dset np.array(n_snippets, n_tgrid, n_x1grid,..., n_xngrid, n_xdim): vorticity, buoyancy
    """
    Nx, Nz = 64,16
    n_griddims = 3 # t, x1, x2
    specifier = f'evolution_{Nx:04}_{Nz:04}'

    src_dir = dir_raw / specifier / "snapshots"
    print('Loading raw data from:', src_dir)
    files = []
    for i in snapshot_range:
        files.append(f'snapshots_s{i}.h5')
    files = [Path(src_dir,file) for file in files]
    files = post.natural_sort(str(sp) for sp in files)
    
    # Merge analysis
    # post.merge_analysis(src_dir, cleanup=True)

    # Init data array
    # Get data dimensions
    task_nxdims = [] # [1,2] # Number of x dimensions of each recorded variable
    for task in tasks:
        shape_snapshot  =  h5py.File(files[0], 'r')['tasks'][task].shape
        if len(shape_snapshot) > n_griddims:
            task_nxdims.append(shape_snapshot[1])
            shape_snapshot = (shape_snapshot[0],)+(shape_snapshot[2:])
        else:
            task_nxdims.append(1)
    n_xdims = sum(task_nxdims)
    n_tgrid = len(files) * shape_snapshot[0]
    n_xngrid = shape_snapshot[1:]
    dset = np.empty((1, n_tgrid,)+ n_xngrid + (n_xdims,))
    grid = np.empty((1, n_tgrid,)+ n_xngrid + (n_griddims,))

    # Iterate over distributed snapshots and concatenate into one npy dataset 
    sample_idx = 0
    for file_idx, filename in enumerate(files):
        with h5py.File(filename, mode='r') as file:
            for task_idx, task in enumerate(tasks):
                print(f'file: {file}, task: {task}')
                t_ids = np.arange(file_idx*shape_snapshot[0],(file_idx+1)*shape_snapshot[0])
                # Store grid, assuming constant grid across tasks and times in snapshot 
                if task_idx == 0: 
                    sim_time = np.array(file['tasks'][task].dims[0]['sim_time'])
                    sim_time = np.repeat(np.repeat(sim_time[:, None,None],repeats=Nx,axis=1),repeats=Nz, axis=-1)

                    grid_x = np.array(file['tasks'][task].dims[1][''])
                    grid_x = np.repeat(np.repeat(grid_x[None,:,None],repeats=len(t_ids),axis=0),repeats=Nz, axis=-1)
                    grid_z = np.array(file['tasks'][task].dims[2][''])
                    grid_z = np.repeat(np.repeat(grid_z[None,None,:],repeats=len(t_ids),axis=0),repeats=Nx, axis=1)

                    grid[0, t_ids,..., 0] = sim_time
                    grid[0, t_ids,..., 1] = grid_x
                    grid[0, t_ids,..., 2] = grid_z

                # Store task data 
                task_idstart = sum(task_nxdims[:task_idx])
                task_idend = sum(task_nxdims[:(task_idx+1)])
                data = np.array(file['tasks'][task])
                if task_nxdims[task_idx] > 1:
                    data = np.moveaxis(data, source=1, destination=-1)
                else:
                    data = data[...,None]
                dset[sample_idx, t_ids,..., task_idstart:task_idend] = data

    # Dump data
    target_dir = dir_interim / specifier
    print('Saving Rayleigh-Benard data and grid at ', target_dir)
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
    np.save(target_dir / 'data.npy', dset)
    np.save(target_dir / 'grid.npy', grid)

    #dset = post.visit_writes(files, reshape_rayleighBenard_to_nn_dist, output=dir_interim)
    
    return 1

if __name__ == "__main__":
    # todo
    # [x] Why is LR integration, N(\bar u_t) so small? 
        # [x] Why does step() add 12,4 dimensions? --> possibly to satisfy boundary conditions??
        # -> I am inconsistent about update grid or spectral values.. find out!
    # [x] save at constant time
    # [x] save at 0.01dt after 40 writes, i.e., t>10 
    # [x] Create parametrization from ( downsample(HR) - LR ) data
    # [x] Test MNO train data. 
        # [] -> Why does the integration error grow from e-22 to e-14? Error from double precision? spectral decomposition?
    # [x] test at 32 - 8 then do 256 - 64
    # [] Generate 10k samples in parallel
    # [] Generate train-val split
    # [] train FNO2D on 1-point sample with 32-8 data and check if train err reaches zero
    # [] train and hyperparam tune FNO2d
    # [] evaluate coupled FNO2D + large-scale sim 
    # [] Set init_sim_time in LR self.solver
    # [x] train over fixed init condition.
    #   [] then sample over initial conditions
    # [x] understand the equation
    #   [x] write down the equations
    parser = argparse.ArgumentParser(description='Rayleigh-Benard Convection')
    args = parser.parse_args()

    dtype = np.float64

    # Generate HR data
    np.random.seed(0)
    grid = RayleighBenardGrid(Nx=64, Nz=16, dtype=dtype)
    rayleighBenardEq = RayleighBenardEq(grid=grid, 
        dtype=dtype, stop_sim_time=50)
    rayleighBenardEq.solve()

    # Store HR data 
    snapshot_range = range(20,32)
    reshape_rayleighBenard_to_nn(snapshot_range=range(20,32))

    # Generate training target with LR
    # TODO: set correct init and stop_sim_time
    np.random.seed(0)
    grid_lr = RayleighBenardGrid(Nx=32, Nz=8, dtype=dtype)
    rayleighBenardEq_lr = RayleighBenardEq(grid=grid_lr, dtype=dtype,
        dir_hr=Path(f"data/interim/temp/rayleighBenard/"),
        Nx_hr=64, Nz_hr=16)
    rayleighBenardEq_lr.solve()
    
    # Test training dataset
    np.random.seed(0)
    grid_lr = RayleighBenardGrid(Nx=32, Nz=8, dtype=dtype)
    rayleighBenardEq_lr = RayleighBenardEq(grid=grid_lr, dtype=dtype)
    sol_target, param = rayleighBenardEq_lr.load_mno_train_data()
    assert True == rayleighBenardEq_lr.test_full_subgrid_forcing(sol_target[0], param[0])