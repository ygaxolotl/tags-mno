from typing import Any, Dict, Optional, Union
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as T

from pce_pinns.models.fno.fno_dataloader import make_big_lazy_cat
from pce_pinns.models.fno import fno2d
from pce_pinns.utils.adam import Adam

TORCH_TYPE_LOOKUP = { 
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
    'complex64': torch.complex64,
    'complex128': torch.complex128,
}

def weighted_avg(x):
    y, n = zip(*x)
    return np.sum(np.multiply(y, n)) / np.sum(n)


def loss_batch(
    model,
    loss_func,
    x,
    y,
    opt=None,
    *,
    model_args=None,
    n_done=None,
    cb=None,
    del_x=False,
    del_y=False
):
    nx = len(x)

    if model_args is None:
        model_args = []

    loss = loss_func(model(x, *model_args), y)
    
    if del_x:
        del x

    if del_y:
        del y

    if cb:
        cb(loss / nx, n_done)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), nx


def fit_epoch(
    model,
    loss_func,
    opt,
    *,
    train_dl,
    test_dl,
    model_args=None,
    train_cb=None,
    test_cb=None,
    device=None,
    quiet=False,
    progress_bar_prefix=""
):
    desc_train = progress_bar_prefix + "  Training"
    desc_test = progress_bar_prefix + "Validation"
    
    model.train()
    loss_train = weighted_avg(
        loss_batch(
            model,
            loss_func,
            x.to(device) if device else x,
            y.to(device) if device else y,
            opt,
            model_args=model_args,
            n_done=i,
            cb=train_cb,
            del_x=device and x.device != device,
            del_y=device and y.device != device,
        )
        for i, (x, y) in enumerate(tqdm(train_dl, desc=desc_train, disable=quiet))
    )

    model.eval()
    with torch.no_grad():
        loss_test = weighted_avg(
            loss_batch(
                model,
                loss_func,
                x.to(device) if device else x,
                y.to(device) if device else y,
                model_args=model_args,
                n_done=i,
                cb=test_cb,
            )
            for i, (x, y) in enumerate(tqdm(test_dl, desc=desc_test, disable=quiet))
        )

    return loss_train, loss_test

class UnNormalize(T.Normalize):
    # Undoes the normalization and returns the reconstructed images in the input domain.
    # Source: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

class UnNormalizeDuringTest(nn.Module):
    def __init__(self, mean, std):
        super(UnNormalizeDuringTest, self).__init__()
        self.unnormalize = UnNormalize(mean=mean, std=std)
        # self.apply_unnormalize = False # Flag to turn unnormalize off during training and on during test.
        self.testing = False

    def forward(self, x):
        if self.testing:
            return self.unnormalize(x)
        else:
            return x

    def test(self, mode=True):
        self.testing=mode

class NormalizeDuringTest(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeDuringTest, self).__init__()
        self.normalize = T.Normalize(mean=mean, std=std)
        self.testing = False

    def forward(self, x):
        if self.testing:
            return self.normalize(x)
        else:
            return x

    def test(self, mode=True):
        self.testing=mode

def test(self,mode=True):
    r"""Sets a module in testing mode."""
    # Is not automatically deactivate upon train() or eval() 
    self.testing = mode
    for module in self.children():
        if hasattr(module, 'test'):
            module.test(mode)
    return self

def make_model(
    config: Dict[str, Any], *, device: Optional[Union[str, torch.device]] = None
) -> Any:
    """Creates and initializes FNO, stacked with a linear layer.

    The linear layer has a single output neuron.

    :param config: Configuration used to initialize the FNO.
    :param device: The ``torch.device`` on which tensors should be stored.
    :return: The (JIT) compiled model.
    """
    torch.set_default_dtype(TORCH_TYPE_LOOKUP[config['dtype']])    
    model = nn.Sequential(
        NormalizeDuringTest(mean=config["means_x"], std=config["stdevs_x"]),
        fno2d.FNO(device=device, #**config),
            depth=config['depth'], n_features=config['n_features'],
            n_channels=config['n_channels'], n_modes=config['n_modes'],
            use_bnorm=config['use_bnorm'], model_dims=config['model_dims']),
        nn.Linear(config["n_channels"], 1), #, device=device),
        UnNormalizeDuringTest(mean=config["means_y"], std=config["stdevs_y"])
    )

    # Add testing mode to control normalization.
    model.test = test
    model.test(model, mode=False)

    #return torch.jit.script(model)
    return model

def train_model(
    *,
    x_train,
    x_test,
    y_train,
    y_test,
    loss_mask,
    epoch_cb,
    dl_config=None,
    model_config=None,
    opt_config=None,
    meta=None,
    seed=None,
    device=None,
):
    model_dims = len(x_train.shape[1:-1]) # 1 for 1D and 2 for 2D FNO

    if seed is not None:
        torch.manual_seed(seed)

    if not (type(loss_mask) is torch.Tensor):
        loss_mask = torch.tensor(loss_mask)

    loss_mask = loss_mask.to(device).unsqueeze(0)

    if len(loss_mask.shape) == 3 and model_dims==2:
        loss_mask = loss_mask.unsqueeze(-1)
    if (
        x_train.shape[:-1] != y_train.shape[:-1]
        or x_test.shape[:-1] != y_test.shape[:-1]
        or x_train.shape[1:] != x_test.shape[1:]
        or y_train.shape[1:] != y_test.shape[1:]
        or y_train.shape[1:-1] != loss_mask.shape[1:-1]
    ):
        raise ValueError("Shapes of input data are inconsistent.")

    if not (y_train.shape[-1] == y_test.shape[-1] == loss_mask.shape[-1] == 1):
        raise ValueError(
            "Last dimension of targets and loss mask has to be of size one."
        )

    if dl_config is None:
        dl_config = dict()

    if model_config is None:
        model_config = dict()

    if opt_config is None:
        opt_config = dict()

    if meta is None:
        meta = dict()

    dl_config = {
        "statics": (0,), #TODO: Number of static input variables ??
        "chunk_size": 4096,
        "n_hist": 0,
        "n_fut": 0,
        "n_repeat": 0,
        "batch_size": 64,
        "seed": seed,
        "p_hflip": 0.,
        "p_vflip": 0.,
        **dl_config, # Default values above are overwritten by **dl_config
    }
    n_statics = len(dl_config["statics"])
    n_hist = dl_config["n_hist"]
    if dl_config['chunk_size'] is None or dl_config['chunk_size'] == 'None':
        dl_config['chunk_size'] = dl_config['batch_size']

    n_features = x_train.shape[-1]
    n_features += (n_features - n_statics) * n_hist

    model_config = {
        "depth": 1,
        "n_features": n_features,
        "n_channels": 7,
        "n_modes": (5, 5),
        "use_bnorm": True,
        "dtype": 'float32',
        "means_x": [],
        "means_y": [],
        "stdevs_x": [],
        "stdevs_y": [],
        **model_config,
    }
    opt_config = {
        "lr": 0.1,
        "step_size": 20,
        "gamma": 0.1,
        **opt_config,
    }
    train_dl = make_big_lazy_cat(x_train, y_train, device="cpu", 
        statics=dl_config['statics'], chunk_size=dl_config['chunk_size'],
        n_hist=dl_config['n_hist'], n_fut=dl_config['n_fut'],
        n_repeat=dl_config['n_repeat'], batch_size=dl_config['batch_size'],
        seed=dl_config['seed'], 
        means_x=model_config['means_x'], stdevs_x=model_config['stdevs_x'], 
        means_y=model_config['means_y'], stdevs_y=model_config['stdevs_y'],
        p_hflip=dl_config['p_hflip'], p_vflip=dl_config['p_vflip'])
    test_dl = make_big_lazy_cat(x_test, y_test, device="cpu", 
        statics=dl_config['statics'], chunk_size=dl_config['chunk_size'],
        n_hist=dl_config['n_hist'], n_fut=dl_config['n_fut'],
        n_repeat=dl_config['n_repeat'], batch_size=dl_config['batch_size'],
        seed=dl_config['seed'], 
        means_x=model_config['means_x'], stdevs_x=model_config['stdevs_x'], 
        means_y=model_config['means_y'], stdevs_y=model_config['stdevs_y'],
        p_hflip=dl_config['p_hflip'], p_vflip=dl_config['p_vflip'])
    model = make_model(model_config, device=device)
    
    # opt = Adam(model.parameters(), lr=opt_config["lr"])# , weight_decay=1e-4)
    opt = torch.optim.Adam(model.parameters(), lr=opt_config["lr"])# , weight_decay=opt_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=opt_config["step_size"], gamma=opt_config["gamma"]
    )

    def loss_fct(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(((x - y) * loss_mask) ** 2)

    loss_train = []
    loss_test = []

    n_epoch = 0
    stop = False
    while not stop:
        n_epoch += 1

        loss = fit_epoch(
            model,
            loss_fct,
            opt,
            train_dl=train_dl,
            test_dl=test_dl,
            device=device,
            progress_bar_prefix=f"[Epoch #{n_epoch}] ",
        )
        loss_train.append(loss[0])
        loss_test.append(loss[1])
        print(f"\n[Epoch #{n_epoch}] Loss - Train {loss[0]:.6f} - Test {loss[1]:.6f}")
        stop = epoch_cb(loss_train, loss_test)

        scheduler.step()

    return model.to("cpu"), {
        "data_loader": dl_config,
        "model": model_config,
        "optimizer": opt_config,
        "meta": meta,
        "n_epochs": n_epoch,
        "loss": {
            "training": loss_train,
            "validation": loss_test,
        },
    }

