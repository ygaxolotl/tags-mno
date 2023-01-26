import os
from pathlib import Path
import numpy as np

from pce_pinns.models.fno.fno_dataloader import make_big_lazy_cat

def get_paths(n_snippets=None, dir_store_ml_data=None, n_chunks=1, full_path=None):
    """
    n_chunks int: Number of time steps in each chunk, i.e., length of each data snippets
    full_path string: Path to directory that contains all files
    """
    if full_path:
        d_proc = Path(full_path)
    else:
        assert n_snippets is not None
        assert dir_store_ml_data is not None
        d_proc = Path(dir_store_ml_data + f'/n{n_snippets:d}_t{int(n_chunks):d}')
    paths = dict()
    paths['f_xtrain'] = d_proc / "xtrain.npy"
    paths['f_xtest'] = d_proc / "xtest.npy"
    paths['f_xactualtest'] = d_proc / "xactualtest.npy"
    paths['f_ytrain'] = d_proc / "ytrain.npy"
    paths['f_ytest'] = d_proc / "ytest.npy"
    paths['f_yactualtest'] = d_proc / "yactualtest.npy"
    paths['f_lossmsk'] = d_proc / "notlandbool.npy"

    return paths, d_proc

def save_fno_data(x, y, n_snippets, val_size, dir_store_ml_data, test_size=0.05, split_style="sequential"):
    """
    Args:
        x np.array((n_snippets*n_tgrid, n_x1grid, n_x2grid, dim_in)): 2D input array, see shaping.shape_2D_to_2D_data for info
        y np.array((n_snippets*n_tgrid, n_x1grid, n_x2grid, dim_y)): 2D output array, see shaping.shape_2D_to_2D_data for info
                            dim_in = grid_in_dims + dim_y_args
        val_size float: size of validation set as percentage/100% of full data, e.g., 0.20
        test_size float: size of test set, e.g., 0.05
        n_snippets: Number of snippets. The input x is a sequence with multiple 'snippets'.
        split_style str: Style to split into train, val, test
    Saves:
        f_xtrain np.array((n*train_size, n_x1grid, n_x2grid, dim_in): 
        f_xtest np.array((n*val_size, n_x1grid, n_x2grid, dim_in)):
        f_ytrain np.array((n*train_size, n_x1grid, n_x2grid, dim_y))): y
        f_ytest np.array((n*val_size, n_x1grid, n_x2grid, dim_y)): 
        f_lossmsk np.array((n_x1grid, n_x2grid, dim_y)): Loss mask; zero for masked values
    """

    lossmsk = np.ones(y[0].shape, dtype=bool)
    
    # Define paths
    paths, d_proc = get_paths(n_snippets, dir_store_ml_data, n_chunks=int(x.shape[0]/n_snippets))
    # Create paths
    if not os.path.exists(d_proc): 
        os.makedirs(d_proc)
    print('Saving processed ML-ready data at: ', d_proc)

    # Create train, val, test split
    ## Calculate length of each snippet so the same snippet is not in different splits
    n_tgrid = int(x.shape[0]/n_snippets)
    if split_style == 'sequential':
        n_val = int(int(n_snippets*val_size)*n_tgrid)
        n_train = int(int(n_snippets*(1-val_size))*n_tgrid)
        # We currently only create val and train. No test split. It says 'test', but it's actually val.
        np.save(paths['f_xtrain'] , x[:n_train])
        np.save(paths['f_xtest'] , x[-n_val:])
        np.save(paths['f_ytrain'] , y[:n_train])
        np.save(paths['f_ytest'] , y[-n_val:])
    elif split_style=='random_snippets':
        # Creates train, val, test splits by randomly taking snippets from a long sequence. 
        #  Every snippet is fully associated to one split.

        # Index the start of every snippet in the sequence. 
        init_idx = np.arange(0, x.shape[0], n_tgrid) # Indexes start of all snippets
        np.random.shuffle(init_idx) # Shuffling is now done across the start idx of snippets 
        
        # Create test split by grabbing snippets randomly from sequence
        n_test_snippets = int(n_snippets*test_size)
        if n_test_snippets >= 1:
            # Ensure that first snippet is in test data.
            init_idx = init_idx[(init_idx != 0)]
            init_idx_test = np.array([0])
        else:
            init_idx_test = np.array([])
        end_id_test = max(n_test_snippets-1, 0)
        init_idx_test = np.concatenate((init_idx_test, init_idx[:end_id_test])) # Indexes test snippet starts
        idx_test = np.r_[[range(init,init+n_tgrid) for init in init_idx_test]].flatten() # Indexes all test elements
        print('Leaving the following initial indices for test: ', init_idx_test)

        # Create validation split
        n_val_snippets = int(n_snippets*val_size)
        end_id_val = end_id_test + n_val_snippets
        init_idx_val = init_idx[end_id_test:end_id_val] # Indexes val snippet starts
        idx_val = np.r_[[range(init,init+n_tgrid) for init in init_idx_val]].flatten() # Indexes all val elements

        # Create train split
        n_train_snippets = int(n_snippets - n_test_snippets - n_val_snippets)
        end_id_train = end_id_val + n_train_snippets
        init_idx_train = init_idx[end_id_val:end_id_train]
        idx_train = np.r_[[range(init,init+n_tgrid) for init in init_idx_train]].flatten()

        # Save data
        if len(idx_test) > 0:
            np.save(paths['f_xactualtest'] , x[idx_test])
            np.save(paths['f_yactualtest'] , y[idx_test])
        if len(idx_val) > 0:
            np.save(paths['f_xtest'] , x[idx_val])
            np.save(paths['f_ytest'] , y[idx_val])
        np.save(paths['f_xtrain'] , x[idx_train])
        np.save(paths['f_ytrain'] , y[idx_train])
    else:
        raise ValueError("Invalid split_style argument to dataloader.save_fno_data")

    # Save loss mask
    np.save(paths['f_lossmsk'] , lossmsk)

    return paths

def init_val_dataloader(n_tgrid, full_cfg, saved_cfg):
    """
    Initializes dataloader with locally stored data.

    Args:
        full_cfg dict(
                'de': {
                    'n_snippets'
                }
                'data_loader': {
                    'dir_store_ml_data'
                    'batch_size'
                }
            ): Config dictionary, created in current run. 
        saved_cfg dict(): Saved model dictionary
    Returns
        val_loader 
    """
    # Get data paths
    paths, d_proc = get_paths(full_cfg['de']['n_snippets'],
        full_cfg['data_loader']['dir_store_ml_data'], 
        n_chunks=full_cfg['data_loader']['batch_size'])

    # Set full length of temporal sequence as batch_size
    if 'sequential' in saved_cfg['data_loader']['test_mode']:
        saved_cfg['data_loader']['batch_size'] = n_tgrid
        val_seed = None # Disable shuffling 
    else:
        val_seed = dl_cfg['seed']
    x_val = np.load(str(paths['f_xtest']), mmap_mode = 'r')
    y_val = np.load(str(paths['f_ytest']), mmap_mode = 'r')
    dl_cfg = saved_cfg["data_loader"]
    dl_cfg["chunk_size"] = int(dl_cfg['batch_size'])
    # TODO: DISABLE SHUFFLING!!
    val_loader = make_big_lazy_cat(x_val, y_val, device="cpu", 
        statics=dl_cfg['statics'], chunk_size=dl_cfg['chunk_size'],
        n_hist=dl_cfg['n_hist'], n_fut=dl_cfg['n_fut'],
        n_repeat=dl_cfg['n_repeat'], batch_size=dl_cfg['batch_size'],
        seed=val_seed)

    return val_loader
