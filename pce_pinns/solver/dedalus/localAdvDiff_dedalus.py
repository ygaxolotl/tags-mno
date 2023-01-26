import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import argparse

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Local Advection-Diffusion')
    args = parser.parse_args()

    # Aspect ratio 
    Lz = (1.)
    nz = (96)

    # Create bases and domain
    z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
    domain = de.Domain([z_basis], grid_dtype=np.float64)

    # Define parameters
    diff = 1.
    vel = 1.
    source = 0.

    # Define Equations
    problem = de.IVP(domain, variables=['T', 'Tz'])
    problem.parameters['k'] = diff
    problem.parameters['w'] = vel
    problem.parameters['f'] = source
    problem.add_equation("dt(T) + w*dz(T) - k*dz(Tz) = f")
    problem.add_equation("Tz - dz(T) = 0")

    # Boundary conditions
    problem.add_bc("left(T) = 0.5")
    problem.add_bc("right(T) = 0.1")

    # Time-stepping
    ts = de.timesteppers.RK443

    # Initialize IVP solver
    solver =  problem.build_solver(ts)

    # Set initial conditions
    z = domain.grid(0)
    T = solver.state['T']
    Tz = solver.state['Tz']

    T['g'] = np.exp(-np.power((z-0.5), 2)/(2.*0.25**2))
    T.differentiate('z',out=Tz)

    # Set integration parameters and CFL
    solver.stop_sim_time = 2.01
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf

    initial_dt = 0.2*Lz/nz
    cfl = flow_tools.CFL(solver,initial_dt,safety=0.8)
    cfl.add_velocities(('T'))

    ## Main loop
    # Make plot of scalar field
    z = domain.grid(0,scales=domain.dealias)
    fig, axis = plt.subplots(figsize=(10,5))
    axis.plot(z, T['g'])
    plt.savefig("../doc/figures/local_ade_T0_dedalus.png")

    logger.info('Starting loop')
    start_time = time.time()
    dt = 0.005
    while solver.ok:
        # dt = cfl.compute_dt()
        solver.step(dt)
        if solver.iteration % 10 == 0:
            # Update plot of scalar field
            axis.plot(z, T['g'], label='t='+str(solver.iteration*dt))
            #plt.savefig("../doc/figures/local_ade_T0_dedalus.png")
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

    end_time = time.time()
    import pdb; pdb.set_trace()
    axis.plot(z, T['g'])

    # Print statistics
    logger.info('Run time: %f' %(end_time-start_time))
    logger.info('Iterations: %i' %solver.iteration)

    plt.savefig("../doc/figures/local_ade_dedalus.png")