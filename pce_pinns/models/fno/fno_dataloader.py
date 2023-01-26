from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import random
import numpy as np
import torch
import torchvision.transforms.functional as TF


class BatchedDL:
    """Batched data loader.

    Wraps an iterable and yields its values in batches.

    :param it: An iterable over pairs of tensors.
    :param batch_size: The batch size.
    :param device: The ``torch.device`` on which tensors should be stored.
    """

    def __init__(
        self,
        it: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        batch_size: int,
        device: Optional[str] = None,
    ) -> None:
        self.it = it
        self.data = iter(self.it)
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        self.data = iter(self.it)
        return self

    def __next__(self):
        first = next(self.data)
        x = torch.empty(self.batch_size, *first[0].shape, device=self.device)
        y = torch.empty(self.batch_size, *first[1].shape, device=self.device)

        x[0] = first[0]
        y[0] = first[1]

        last = 0
        for i in range(1, self.batch_size):
            try:
                x[i], y[i] = next(self.data)
                last = i + 1
            except StopIteration:
                last = i
                break

        return x[:last], y[:last]


class BigDL:
    """Gathering of data loader.

    Gathers data loaders, initializes and iterates them in sequence.

    :param DL: Data loader class.
    :param args: Unnamed arguments for each data loader constructor.
        e.g., data list of [x, y, statics] snippets to LazyFeatureCat 
    :param kwargs: Named arguments for each data loader constructor.
    """

    def __init__(self, DL, args: List[Any], kwargs: List[Dict[str, Any]]) -> None:
        if len(args) != len(kwargs):
            raise ValueError("Number of unnamed and named arguments has to match.")

        self.DL = DL
        self.args = args
        self.kwargs = kwargs

        self.dl = None
        self.n = len(args)
        self.i = self.n

    def _load_next_dl(self):
        idx = -self.i
        self.i -= 1

        if self.i < 0:
            raise StopIteration

        args = self.args[idx]
        kwargs = self.kwargs[idx]

        return self.DL(*args, **kwargs)

    def __iter__(self):
        self.i = self.n
        return self

    def __next__(self):
        if self.dl is None:
            self.dl = self._load_next_dl()
        try:
            return next(self.dl)
        except StopIteration:
            self.dl = self._load_next_dl()
            return next(self.dl)


class LazyFeatureCat:
    """A dataloader that lazily concatenates features of consecutive elements.

    This dataloader can be used to load temporal sequences into images by stacking them across the
    feature dimension.

    Given two sequences of ``f`` features with ``n`` elements for ``x`` and ``m`` elements for ``y``
    with ``m < n``, this iterator will stack ``n - m + 1`` consecutive elements in the last 
    dimension for ``x`` and yield those together with the corresponding value in ``y``. This 
    stacking happens lazily on each yield of the iterator.

    For example, given ``(8, 11, 13, 7)`` and ``(5, 11, 13, 1)`` for ``x`` and ``y``, respectively,
    i.e., ``n=8``, ``m=5`` and ``f=7``, the iterator will yield 5 tuples of size:

      1. ``((11, 13, 28), (11, 13, 1))``,
      2. ``((11, 13, 28), (11, 13, 1))``,
      3. ``((11, 13, 28), (11, 13, 1))``,
      4. ``((11, 13, 28), (11, 13, 1))``,
      5. ``((11, 13, 28), (11, 13, 1))``.

    The enlarged feature dimension of the first element of each tuple is the product of
    ``(n - m + 1) x f`` where the first tuple is the result of stacking the elements of the
    first 4 features (or elements?) in ``x`` on top of each other. Similarly, the second tuple is
    the result of the 2nd, 3rd, 4th and 5th element, etc. (I think ``n-m`` can be the history, i.e., 
    number of past time steps that will be stacked.)

    :param x: Input tensor of shape ``(n, *, f)``. This can be a chunk or snippet of a sequence.
    :param y: Input tensor of shape ``(n, *)``.
    :param statics: Indices of features that should not be stacked.
    :param n_repeat: Number of sequence repetitions.
    :param seed: If not ``None``, this value is used to seed a random engine and the tuples are
                 shuffled upon return.
    :param device: The ``torch.device`` on which tensors should be stored.
    :param means_x: List of means per feature, calculated across full dataset x including statics
    :param means_y: List of means per feature, calculated across full dataset y including statics
    :param stdevs_x: List of standard deviations per feature
    :param stdevs_y: List of standard deviations per feature
    :param p_hflip: Probability to apply horizontal flip augmentation
    :param p_vflip: Probability to apply vertical flip augmentation
    """

    def __init__(
        self,
        x: Any,
        y: Any,
        statics: Optional[List[int]] = None,
        n_repeat: int = 0,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        means_x: Optional[List] = [],
        means_y: Optional[List] = [],
        stdevs_x: Optional[List] = [],
        stdevs_y: Optional[List] = [],
        p_hflip: Optional[float] = 0.,
        p_vflip: Optional[float] = 0.
    ) -> None:
        self.device = device

        # Apply normalization and other transforms
        self.tf_cfg = {"means_x": means_x,
            "means_y": means_y,
            "stdevs_x": stdevs_x,
            "stdevs_y": stdevs_y,
            "p_hflip": p_hflip,
            "p_vflip": p_vflip}
        x, self.y = self.transform(x, y)

        # Split x into static and non-static inputs
        if statics is None:
            statics = []
        if not all([0 <= i < x.shape[-1] for i in statics]):
            raise ValueError("Found invalid index in statics.")
        sel = [i not in statics for i in range(x.shape[-1])]
        sel = torch.tensor(sel, device=x.device)
        self.x = x[..., sel]
        self.statics = x[..., ~sel]

        self.i = 0 # Iterator through the tuples
        self.j = n_repeat
        self.n = self.y.shape[0] # Total number of tuples
        self.n_stack = self.x.shape[0] - self.n
        self.n_repeat = n_repeat

        # Draw tuples in random order from stack of tuples
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = None
        self.idx = self._generate_idx()

    def transform(self, x, y):
        # todo: double-check that applying transforms during initalization of the dataloader 
        # is not screwing with any lazy loading. I'm assuming that conversion of numpy memmap into 
        # Tensor already loads the vector into memory.

        # Convert to tensor
        x_tensor = x if type(x) is torch.Tensor else torch.tensor(x, device=self.device)
        y_tensor = y if type(y) is torch.Tensor else torch.tensor(y, device=self.device)

        # Verify that image dimensions match
        assert x_tensor.shape[1:-1] == y_tensor.shape[1:-1], 'Image dimensions do not match.'

        # Normalize
        if len(self.tf_cfg['means_x']) != 0 and len(self.tf_cfg['stdevs_x']) != 0:
            x_tensor = TF.normalize(x_tensor, mean=self.tf_cfg['means_x'], std=self.tf_cfg['stdevs_x'])
        if len(self.tf_cfg['means_y']) != 0 and len(self.tf_cfg['stdevs_y']) != 0:
            y_tensor = TF.normalize(y_tensor, mean=self.tf_cfg['means_y'], std=self.tf_cfg['stdevs_y'])

        # Random horizontal flipping
        if random.random() > (1 - self.tf_cfg['p_hflip']):
            x_tensor = TF.hflip(x_tensor)
            y_tensor = TF.hflip(y_tensor)

        # Random vertical flipping
        if random.random() > (1 - self.tf_cfg['p_vflip']):
            x_tensor = TF.vflip(x_tensor)
            y_tensor = TF.vflip(y_tensor)

        return x_tensor, y_tensor

    def _generate_idx(self):
        idx = np.arange(self.n) + self.n_stack
        if self.rng:
            self.rng.shuffle(idx)
        return torch.tensor(idx, device=self.device)

    def __iter__(self):
        self.i = 0
        self.j = self.n_repeat
        return self

    def __next__(self):
        if self.i >= self.n:
            self.i = 0
            self.j -= 1
            self.idx = self._generate_idx()

        if self.j < 0:
            raise StopIteration

        i = self.idx[self.i]
        self.i += 1
        
        return (
            torch.cat(
                [self.x[j] for j in range(i - self.n_stack, i + 1)] # Select elements in x
                + [self.statics[i - self.n_stack]], # Stack with static elements
                dim=-1,
            ),
            self.y[i - self.n_stack], # Select tuple from y
        )

def make_big_lazy_cat(
    x: Any,
    y: Any,
    statics: Any,
    chunk_size: int,
    n_hist: int,
    n_fut: int,
    n_repeat: Optional[int] = 0,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    means_x: Optional[List] = [],
    means_y: Optional[List] = [],
    stdevs_x: Optional[List] = [],
    stdevs_y: Optional[List] = [],
    p_hflip: Optional[float] = 0.,
    p_vflip: Optional[float] = 0.
) -> Union[BatchedDL, BigDL]:
    """Returns wrapped :class:`LazyFeatureCat` instances.

    The input data is divided into chunks and :class:`LazyFeatureCat` instances are created for each
    chunk.

    :param x: Divided into chunks and forwarded as ``x`` to :class:`LazyFeatureCat`.
    :param y: Divided into chunks and forwarded as ``y`` to :class:`LazyFeatureCat`.
    :param statics: Indices of static features (see :class:`LazyFeatureCat` for more details).
    :param chunk_size: Chunk size or snippet length.
    :param n_hist: Number of events to be stacked by :class:`LazyFeatureCat` on top of each element.
        # n_hist is assuming that input, x, and target, y, are from the same sequence and have same modality. 
        # Do not use n_hist if x and y are different modalities.
    :param n_fut: Offset of ``y`` w.r.t. to current index.
    :param n_repeat: See :class:`LazyFeatureCat` for details.
    :param batch_size: If not ``None`` the result of :class:`LazyFeatureCat` are gathered in batches
                       of the given size.
    :param seed: See :class:`LazyFeatureCat` for details.
    :param device: The ``torch.device`` on which tensors should be stored.
    :param means_x: See :class:`LazyFeatureCat` for details.
    :param means_y: See :class:`LazyFeatureCat` for details.
    :param stdevs_x: See :class:`LazyFeatureCat` for details.
    :param stdevs_y: See :class:`LazyFeatureCat` for details.
    :param p_hflip: See :class:`LazyFeatureCat` for details.
    :param p_vflip: See :class:`LazyFeatureCat` for details.
    :return: An iterable that wraps :class:`LazyFeatureCat` instances for each chunk.
    """
    if chunk_size < n_hist:
        raise ValueError("The chunk size cannot be smaller than the history size.")

    min_length = n_hist + 1
    o1 = n_hist
    o2 = o1 + n_fut

    sequence_length_x = x.shape[0]
    sequence_length_y = y.shape[0]

    args = []
    
    # Add the in- and output sequence, x and y, in snippets or chunks of chunk_size 
    #  to the data stack. A sequence has multiple snippets.
    for i in range(0, sequence_length_x, chunk_size):
        x_first = i
        x_last = min(x_first + chunk_size, sequence_length_x)
        n_x = x_last - x_first

        n_y = n_x - n_hist
        y_first = o2 + i
        y_last = min(y_first + n_y, sequence_length_y)
        dn = n_y - (y_last - y_first)

        x_last -= dn
        n_x -= dn
        if n_x >= min_length:
            args.append((x[x_first:x_last], y[y_first:y_last], statics))

    kwargs = [{"n_repeat": n_repeat, 
        "seed": seed, 
        "device": device, 
        "means_x": means_x,
        "means_y": means_y,
        "stdevs_x": stdevs_x,
        "stdevs_y": stdevs_y,
        "p_hflip": p_hflip,
        "p_vflip": p_vflip}] * len(args)

    dl = BigDL(LazyFeatureCat, args, kwargs)
    return BatchedDL(dl, batch_size=batch_size, device=device) if batch_size else dl
