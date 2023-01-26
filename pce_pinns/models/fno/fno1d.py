from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from pce_pinns.models.fno import fno_dft as dft

class SpectralConv1D(nn.Module):
    """Applies convolution to input in truncated Fourier domain.

    See description in :class:`SpectralConv` but input vector has shape ``(+, x, n_channels)``

    :param n_channels: Number of channels
    :param n_modes: Number of positive frequency modes in Fourier domain, i.e., ``n`` corresponds to
                    a Fourier domain of size ``2n + 1`` (todo:check)
    :param device: The ``torch.device`` on which tensors should be stored.
    """

    def __init__(
        self,
        n_channels: int,
        n_modes: Tuple[int],
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        if len(n_modes) != 1:
            raise ValueError("Only 1D Fourier domains are supported.")

        self.device = device
        self.n_channels = n_channels
        self.n_modes = n_modes
        torch_dtype = torch.get_default_dtype()
        if torch_dtype == torch.float16:
            print('Warning: torch 1.10 does not support ComplexHalf. '\
                'Will assign complex64 instead.')
            self.dtype = torch.complex64
        elif torch_dtype == torch.float64:
            self.dtype = torch.complex128
        else: #  torch_dtype == torch.float32:
            self.dtype = torch.complex64
        kx = self.n_modes[0]
        n_params = (2*kx-1) * n_channels * n_channels

        scale = 1.0 / (n_channels ** 2)
        # self.weights = nn.Parameter(scale * torch.rand(n_params, device=device))
        # self.weights.type(torch.complex64).reshape(2*kx-1, self.n_channels, self.n_channels)
        self.use_torch = True
        self.use_complex = True
        if self.use_complex:
            self.weights = nn.Parameter(scale * torch.rand((self.n_modes[0]+1, n_channels, n_channels), 
                dtype=self.dtype, device=device))
            #self.weights = nn.Parameter(scale * torch.rand((2*kx-1, n_channels, n_channels), 
            #    dtype=torch.complex64, device=device))
            #self.weights = nn.Parameter(scale * torch.rand(n_params, 
            #    dtype=torch.complex64, device=device))
        else:
            self.weights = nn.Parameter(scale * torch.rand((self.n_modes[0]+1, n_channels, n_channels), 
                device=device))
            #self.weights = nn.Parameter(scale * torch.rand((2*kx-1, n_channels, n_channels), 
            #    device=device))
        
        self.is_init = False # 
        self.m = None # dft matrix
        self.inv_m = None # inverse dft matrix


    def build_lazy_dft_matrices1D(self, x: torch.Tensor) -> (Tuple[torch.Tensor,torch.Tensor], Tuple[torch.Tensor,torch.Tensor]):
        """Builds forward and inverse DFT matrices

        We only build the dft matrix once during initialization, becuase building 
        it is n^2 runtime while applying it is nlog n. The dft matrix has shape (2*kx-1,nx,2),
        because it transforms a vector of size, nx, into a vector of size 2*kx-1 with two 
        dimensions for real and imaginary part.

        :param x: Input vector of shape (b, nx, n_c)
        :return m: List[my, mx] with y and x-dim dft matrix of shape(2*kx-1,nx) and self.dtype 
        :return inv_m: List[inv_my, inv_mx] with y and x-dim idft matrix of shape (nx,2*kx-1) and self.dtype
        """
        if not self.is_init:
            print('Building DFT matrices.')
            nx = x.shape[1]
            kx = self.n_modes[0]

            # Generate points in time, p, and frequency domain, f
            pf = [(torch.arange(nx, device=self.device) / nx,
                   torch.cat([torch.arange(kx, device=self.device),
                              torch.arange(nx - kx + 1, nx, device=self.device),]),)]

            self.m = [dft.dft_matrix(p, f) for p, f in pf] # Build dft matrix for y and x dimension
            self.inv_m = [dft.idft_matrix(p, f) for p, f in pf] # Build idft matrix y and x dimension 

            if self.use_complex:
                self.m = [torch.view_as_complex(m) for m in self.m]
                self.inv_m = [torch.view_as_complex(inv_m) for inv_m in self.inv_m]

            self.is_init = True

        return self.m, self.inv_m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Args:
        #   x (b, nx, n_c) 

        nx = x.shape[1]
        kx = self.n_modes[0]

        if self.use_torch:
            fx_torch = torch.fft.rfft(x, dim=-2)
            # fx: (b,   n_m, n_c)
            # w:  (n_m, n_c, n_c)
            # out_ft = torch.einsum("bmi,mio->bmo", fx_torch[:,:self.n_modes[0]+1,:], self.weights)
            out_ft = torch.matmul(self.weights, fx_torch[:,:self.n_modes[0]+1,:,None])[...,0]
            x_next = torch.fft.irfft(out_ft, n=nx, dim=-2)
        else:
            m, inv_m = self.build_lazy_dft_matrices1D(x) # m[0]:    (2*kx-1, nx)
    
            # dft in 1d
            if self.use_complex:
                x = x.type(self.dtype)
                fx = torch.matmul(m[0], x) # fx:   (b, 2*kx-1, n_c) 
            else:
                fx = dft.cr_matrix_vector_product(m[0], x) 
        
            # Convolution
            # Each frequency in x direction of fx is multiplied with a learned n_c*n_c matrix
            if self.use_complex:
                # self.weights = nn.Parameter(scale * torch.rand(n_params, dtype=torch.complex64, device=device))
                core = self.weights #.type(torch.complex64).reshape(2*kx-1, self.n_channels, self.n_channels) # core:     (2*kx-1, n_c, n_c)
                y = torch.matmul(core, fx[...,None])[...,0] # y:        (b, 2*kx-1, n_c)
            else:
                y = dft.cc_matrix_vector_product(self.weights, fx) 
        
            # irdft 1d takes a complex vector and returns a real-valued vector:
            x_next = torch.matmul(inv_m[0], y)
            if self.use_complex:
                x_next = torch.view_as_real(x_next)
            x_next = x_next[...,0]
        
        return x_next

class Layer1D(nn.Module):
    """FNO Layer in 1D.

    See description in :class:`Layer`
    """

    def __init__(
        self,
        n_channels: int,
        n_modes: Tuple[int, int],
        use_bnorm: bool,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.sconv = SpectralConv1D(n_channels, n_modes, device=device)
        self.bias = nn.Linear(n_channels, n_channels)#, device=device)
        self.bnorm = nn.BatchNorm1d(n_channels)#, device=device)
        self.use_bnorm = use_bnorm
        self.activation = nn.ReLU()
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.sconv(x)
        y2 = self.bias(x)
        y = y1 + y2
        if self.use_bnorm:
            z = self.bnorm(y.transpose(1, -1)).transpose(1, -1)
        else:
            z = y
        return self.activation(z)

class FNO1D(nn.Module):
    """FNO Network for 1D data.

    A real-valued ``(b, x, f)`` tensor is transformed into a real-valued ``(b, x, c)``
    tensor where ``b`` is the batch size, `x` refers to the size of the dimension subjected to
    the Fourier transform and ``f`` (``c``) is the number of input (output) features (channels).

    See description in FNO() for more.

    :param depth: Number of chained FNO layers.
    :param n_features: Number of input features.
    :param n_channels: See description in :class:`SpectralConv`.
    :param n_modes: See description in :class:`SpectralConv`.
    :param use_bnorm: If True, use Batchnorm layer
    :param device: The ``torch.device`` on which tensors should be stored.
    """

    def __init__(
        self,
        depth: int,
        n_features: int,
        n_channels: int,
        n_modes: Tuple[int, int],
        use_bnorm: bool,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        # TODO!!! fix device setting
        fc = nn.Linear(n_features, n_channels)#, device=device)

        layers = [Layer1D(n_channels, n_modes, use_bnorm)]*depth#, device=device)] * depth

        self.f = nn.Sequential(fc, *layers)
        self.n_features = n_features
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.n_features:
            raise ValueError(
                f"Input tensor has to have {self.n_features} features (size of last dimension), but has {x.shape[-1]} features."
            )

        return self.f(x)

if __name__ == "__main__":
    import numpy as np
    from pce_pinns.models.mno import msr_runtime
    import pce_pinns.utils.plotting as plotting

    nx = 4
    n_features = 1
    x = torch.rand(2, nx, n_features)
    device="cpu"
    # Test FNO1D
    model = nn.Sequential(
        FNO1D(device=device, #**config),
            depth=2, n_features=n_features,
            n_channels=5, n_modes=[4],
            use_bnorm=False),
        nn.Linear(5, 1),#, device=device),
    )

    y = model(x)

    # import pdb;pdb.set_trace()
    measure_runtime=False
    if measure_runtime:
        Ks = [65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4] # 8192, 4096, 
        m_samples = (1000*np.ones(len(Ks),dtype=int)).astype(int) # Repeats time measurement m-times.
        m_samples[0] = 100
        m_samples[1] = 500
        m_samples[2] = 1000
        runtimes = []
        for n_i, K in enumerate(Ks):
            print(n_i, K)
            # TODO: for implement model.rebuild_dft_matrix
            model[0].f[1].sconv.is_init = False
            model[0].f[2].sconv.is_init = False
            if K > 4:
                factors = int(np.log2(K) - np.log2(4))
                x_up = x.clone()
                for _ in range(factors):
                    x_up = torch.cat((x_up,x_up), dim=(1))[:,:,:]
                x_in = x_up
            else:
                x_in = x

            _ = model(x_in) # Warmup to build dft matrix
            runtimes.append(msr_runtime(model, x_in, M=int(m_samples[n_i])))
            print(f'N={K:5d}: Runtime {runtimes[-1]:.6f}s')
        print(runtimes)
        plotting.plot_lorenz96_runtimes(Ns=Ks, ts=runtimes)
    else:
        plotting.plot_lorenz96_runtimes(Ns=None, ts=None)


