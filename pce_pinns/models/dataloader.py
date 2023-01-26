import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

class RegressionDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def get_paths(n_snippets, dir_store_ml_data, n_chunks=1,val=True,test=False):
    """
    n_chunks int: Number of data chunks , e.g., int(x.shape[0]/n_snippets)
    """
    d_proc = Path(dir_store_ml_data + f'/n{n_snippets:d}_t{int(n_chunks):d}')
    paths = dict()
    paths['f_xtrain'] = d_proc / "xtrain.npy"
    paths['f_ytrain'] = d_proc / "ytrain.npy"
    if val:
        paths['f_xval'] = d_proc / "xval.npy"
        paths['f_yval'] = d_proc / "yval.npy"
    if test:
        paths['f_xtest'] = d_proc / "xtest.npy"
        paths['f_ytest'] = d_proc / "ytest.npy"
    # paths['f_lossmsk'] = d_proc / "notlandbool.npy"

    return paths, d_proc

def create_and_store_splits(X, Y, batch_size, dir_store_ml_data='data/processed/lorenz96/fcnn/', 
    test_size=0.1, val_size=0.1, seed=0):
    """
    Creates train, val, test data splits
    Args:
        X np.array((n_data, dim_in))
        Y np.array((n_data, dim_out))
        dir_store_ml_data str: Path to store ML ready data
        batch_size int: Batch size
        test_size float: Size of test dataset in percent of total dataset
        val_size float: Size of validation dataset in percent of total dataset
    
    """
    n_data = X.shape[0]
    # Check split sizes
    if test_size*X.shape[0]%batch_size != 0. or val_size*X.shape[0]%batch_size != 0.:
        print('test size %0.3f, val_size %0.3f, batch_size %d, n_samples %d'%(test_size, val_size, batch_size, X.shape[0]))
        raise ValueError('Enter test or val size as valid percentage')

    # Create splits
    if test_size > 0:
        X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, shuffle=False, random_state=seed)
        val_size = val_size/(1.-test_size)
    else:
        X_test = None
        Y_test = None

    if val_size > 0:
        X, X_val, Y, Y_val = train_test_split(X, Y, test_size=val_size, shuffle=False, random_state=seed+1)
    else:
        X_val = None
        Y_val = None

    # Create paths
    paths, d_proc = get_paths(n_data, dir_store_ml_data, n_chunks=1, val=(val_size>0),test=(test_size>0))
    if not os.path.exists(d_proc): 
        os.makedirs(d_proc)

    # Store ML-ready splits
    print('Saving processed ML-ready data at: ', d_proc)
    np.save(paths['f_xtrain'] , X)
    np.save(paths['f_ytrain'] , Y)
    if test_size > 0:
        np.save(paths['f_xtest'] , X_test)
        np.save(paths['f_ytest'] , Y_test)
    if val_size > 0:
        np.save(paths['f_xval'] , X_val)
        np.save(paths['f_yval'] , Y_val)

    return paths

def init_dataloader(paths, batch_size, test_size=0.1, val_size=0.1, shuffle=False, seed=0):
    """
    Initialize the dataloader
    
    Note that we're taking care that data is not shuffled within one batch to, e.g., retain flattened spatiotemporal dimensions

    Args:
        paths
        batch_size int: Batch size
        test_size float: Size of test dataset in percent of total dataset
        val_size float: Size of validation dataset in percent of total dataset
        shuffle bool: If True, shuffles dataset. The option is not true, s.t., the rand instance in train and test can be correlated
    
    Returns:
        train_loader DataLoader
        val_loader DataLoader
        test_loader DataLoader
    """
    # Load data
    X = np.load(str(paths['f_xtrain']), mmap_mode = 'r')
    Y = np.load(str(paths['f_ytrain']), mmap_mode = 'r')
    if test_size > 0:
        X_test = np.load(str(paths['f_xtest']), mmap_mode = 'r')
        Y_test = np.load(str(paths['f_ytest']), mmap_mode = 'r')
    if val_size > 0:
        X_val = np.load(str(paths['f_xval']), mmap_mode = 'r')
        Y_val = np.load(str(paths['f_yval']), mmap_mode = 'r')

    # Init data loaders
    if test_size > 0:
        test_dataset = RegressionDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)
        val_size = val_size/(1.-test_size)
    else:
        test_loader = None
    if val_size > 0:
        val_dataset = RegressionDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        val_loader = None
    train_dataset = RegressionDataset(torch.from_numpy(X), torch.from_numpy(Y))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)


    return train_loader, val_loader, test_loader
