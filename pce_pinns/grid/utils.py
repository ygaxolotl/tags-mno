import os
import psutil
import numpy as np

from pathlib import Path



def get_uniform_tgrid(tmax, dt, dtype, 
    load_data_path=None, store_data_path=None,
    load_interim_data_path=None, store_interim_data_path=None):
    """
    Generates a uniform timegrid
    """
    if load_interim_data_path is not None:
        tgrid = np.load(load_interim_data_path+'_tgrid.npy')
    elif load_data_path is not None:
        tgrid = np.load(load_data_path+'_tgrid.npy')
    else:
        n_tgrid = int(tmax/dt)
        try:
            import pce_pinns.deepxde.deepxdegeometry.timedomain as timedomain # Define grid
            os.environ['DDEBACKEND'] = 'pytorch'
            tdomain = timedomain.TimeDomain(0., tmax)
            tgrid = tdomain.uniform_points(n=n_tgrid, boundary=True) # dim: n_tgrid, dim_grid
            tgrid = tgrid[:,0].astype(dtype)
        except:
            tgrid = np.linspace(0., tmax, num=n_tgrid, dtype=dtype)
        # Dump grid
        if store_data_path is not None:
            folder = Path(*Path(store_data_path).parts[:-1])
            if not folder.exists():
                folder.mkdir(parents=True)
            np.save(store_data_path+'_tgrid.npy', tgrid)
        if store_interim_data_path is not None:
            folder = Path(*Path(store_interim_data_path).parts[:-1])
            if not folder.exists():
                folder.mkdir(parents=True)
            np.save(store_interim_data_path+'_tgrid.npy', tgrid)

    return tgrid

def get_snippet_lengths(n_snippets, n_tsnippet, dt, parallel):
    """
    Each processor computes one long time sequence that is chopped
    up into snippets. This function calculates the time each processor
    has to run the model s.t., n_snippets snippets are generated.

    Args:
        n_snippets int: Total desired number of snippets. 
        n_tsnippet int: Number of time steps per snippet
        dt float: Time discretization
        parallel bool: If true, assumes that data is generated in parallel
    Returns:
        n_snippets int: Number of total snippets. Rounded down, s.t., 
            each processor computes the same number of snippets.
        tmax float: Modeling time on each processor, after warmup, 
            s.t., n_snippets are produced in tmax time
        n_processors int: Number of processors
    """
    # Each parallel process will create one long sample/sequence
    if parallel:
        n_processors = int(psutil.cpu_count(logical=False))
    else:
        n_processors = 1
    n_snippets_per_processor = int(np.floor(n_snippets/n_processors))
    # Make sure n_snippets is multiple of n_processors
    n_snippets = int(n_snippets_per_processor*n_processors)
    tmax = n_tsnippet * dt * n_snippets_per_processor

    return n_snippets, tmax, n_processors

def update_tgrid_w_warmup(
    warmup,
    n_snippets,
    tmax,
    dt,
    parallel,
    dtype='float64',
    n_tmax_warmup = 10.,
    ):
    """
    Updates variables to respect warmup.

    Args:
        warmup bool: 
        n_snippets: See get_snippet_lengths()
        n_tsnippets: See get_snippet_lengths()
        dt: See get_snippet_lengths()
        parallel: See get_snippet_lengths()
    Returns:
        tgrid_warmup
        n_snippets: See get_snippet_lengths()
        n_tsnippet
        tmax: See get_snippet_lengths()      
        n_tasks int: Number of model runs that will be executed via parallel processing.       
    """
    n_tsnippet = int(tmax / dt)
    if warmup:
        n_snippets, tmax, n_processors = get_snippet_lengths(n_snippets, n_tsnippet, dt, parallel)
        n_tasks = n_processors

    if warmup:
        # Get grid for warmup
        tgrid_warmup = get_uniform_tgrid(n_tmax_warmup, dt, dtype)
        print(f"Warming up {n_processors} processors for {n_tmax_warmup} time "\
            f"and then creating a total of {n_snippets} snippets "\
            f"of length {n_tsnippet} steps.")
    else:
        tgrid_warmup=None
        n_tasks = n_snippets

    return tgrid_warmup, n_snippets, n_tsnippet, tmax, n_tasks
