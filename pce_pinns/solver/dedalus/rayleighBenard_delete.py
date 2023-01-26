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
import argparse
import pathlib
import numpy as np
import dedalus.public as d3
import dedalus.core as core
from dedalus.core.timesteppers import RK222
from dedalus.core.coords import CartesianCoordinates
from dedalus.core.basis import RealFourier, ChebyshevT
from dedalus.core.distributor import Distributor
from dedalus.core.operators import Lift, grad, div, skew
from dedalus.core.problems import IVP

from pce_pinns.solver.diffeq import DiffEq
from pce_pinns.utils import plotting

import logging
logger = logging.getLogger(__name__)


class RayleighBenardEq(DiffEq):
    def __init__(self, tgrid, param='fully_resolved', 
        y_param_mode='superparam', random_ic=True, 
        random_ic_y=True, random_ic_z=True, 
        dummy_init=False,
        plot=False, 
        tgrid_warmup=None, seed=0,
        dtype=np.float32):
        """
        Sets up rayleigh benard equation as initial value problem. The equation
        models chaotic 2D convection and is based on 
        https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_rayleigh_benard.html 

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
        Attributes:
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rayleigh-Benard Convection')
    args = parser.parse_args()

    # Parameters
    Lx, Lz = 4, 1
    # Nx, Nz = 256, 64
    # Nx, Nz = 128, 32
    Nx, Nz = 64, 16
    Rayleigh = 2e6
    Prandtl = 1
    dealias = 3/2
    stop_sim_time = 50
    timestepper = RK222
    max_timestep = 0.125
    dtype = np.float64

    # Bases
    coords = CartesianCoordinates('x', 'z')
    dist = Distributor(coords, dtype=dtype)
    xbasis = RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    zbasis = ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

    # Fields
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    b = dist.Field(name='b', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
    tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    # Substitutions
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: Lift(A, lift_basis, -1)
    grad_u = grad(u) + ez*lift(tau_u1) # First-order reduction
    grad_b = grad(b) + ez*lift(tau_b1) # First-order reduction

    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    problem = IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
    problem.add_equation("b(z=0) = Lz")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("b(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # Initial conditions
    b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
    b['g'] *= z * (Lz - z) # Damp noise at walls
    b['g'] += Lz - z # Add linear background

    # Analysis
    data_dir = pathlib.Path(f"data/raw/temp/rayleighBenard/evolution_{Nx:04}_{Nz:04}/snapshots")
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    snapshots = solver.evaluator.add_file_handler(data_dir, sim_dt=0.25, max_writes=50)
    snapshots.add_task(b, name='buoyancy')
    snapshots.add_task(-div(skew(u)), name='vorticity')

    # CFL
    CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
                 max_change=1.5, min_change=0.5, max_dt=max_timestep)
    CFL.add_velocity(u)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(np.sqrt(u@u)/nu, name='Re')

    # Main loop
    startup_iter = 10
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep() # min ~ 0.01
            solver.step(timestep)
            if (solver.iteration-1) % 10 == 0:
                max_Re = flow.max('Re')
                logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()

