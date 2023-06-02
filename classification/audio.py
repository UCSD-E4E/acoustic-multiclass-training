"""Audio datasets and utilities."""

import os
from os import listdir
from os.path import join

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torchaudio import transforms as audtr
# from src.dataloaders.base import default_data_path, SequenceDataset, deprecated

"""Core dataloader interface."""

import os
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as TF
import torchvision
from einops import rearrange
from einops.layers.torch import Rearrange
import permutations
from torch.nn import functional as F
from tqdm import tqdm
from icecream import ic

"""Utilities for dealing with collection objects (lists, dicts) and configs."""

from typing import Sequence, Mapping, Optional, Callable
import functools
from omegaconf import ListConfig, DictConfig

# TODO this is usually used in a pattern where it's turned into a list, so can just do that here
def is_list(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


def is_dict(x):
    return isinstance(x, Mapping)


def to_dict(x, recursive=True):
    """Convert Sequence or Mapping object to dict.

    lists get converted to {0: x[0], 1: x[1], ...}
    """
    if is_list(x):
        x = {i: v for i, v in enumerate(x)}
    if is_dict(x):
        if recursive:
            return {k: to_dict(v, recursive=recursive) for k, v in x.items()}
        else:
            return dict(x)
    else:
        return x


def to_list(x, recursive=False):
    """Convert an object to list.

    If Sequence (e.g. list, tuple, Listconfig): just return it

    Special case: If non-recursive and not a list, wrap in list
    """
    if is_list(x):
        if recursive:
            return [to_list(_x) for _x in x]
        else:
            return list(x)
    else:
        if recursive:
            return x
        else:
            return [x]


def extract_attrs_from_obj(obj, *attrs):
    if obj is None:
        assert len(attrs) == 0
        return []
    return [getattr(obj, attr, None) for attr in attrs]


# def instantiate(registry, config, *args, partial=False, wrap=None, **kwargs):
#     """Instantiate a class or Callable. Wraps hydra.utils.instantiate.

#     registry: Dictionary mapping names to functions or target paths (e.g. {'model': 'models.SequenceModel'})
#     config: Dictionary with a '_name_' key indicating which element of the registry to grab, and kwargs to be passed into the target constructor
#     wrap: wrap the target class (e.g. ema optimizer or tasks.wrap)
#     *args, **kwargs: additional arguments to override the config to pass into the target constructor
#     """

#     # Case 1: no config
#     if config is None:
#         return None
#     # Case 2a: string means _name_ was overloaded
#     if isinstance(config, str):
#         _name_ = None
#         _target_ = registry[config]
#         config = {}
#     # Case 2b: grab the desired callable from name
#     else:
#         _name_ = config.pop("_name_")
#         _target_ = registry[_name_]

#     # Retrieve the right constructor automatically based on type
#     if isinstance(_target_, str):
#         fn = hydra.utils.get_method(path=_target_)
#     elif isinstance(_target_, Callable):
#         fn = _target_
#     else:
#         raise NotImplementedError("instantiate target must be string or callable")

#     # Instantiate object
#     if wrap is not None:
#         fn = wrap(fn)
#     obj = functools.partial(fn, *args, **config, **kwargs)

#     # Restore _name_
#     if _name_ is not None:
#         config["_name_"] = _name_

#     if partial:
#         return obj
#     else:
#         return obj()


# def get_class(registry, _name_):
#     return hydra.utils.get_class(path=registry[_name_])


def omegaconf_filter_keys(d, fn=None):
    """Only keep keys where fn(key) is True. Support nested DictConfig."""
    # TODO can make this inplace?
    if fn is None:
        fn = lambda _: True
    if is_list(d):
        return ListConfig([omegaconf_filter_keys(v, fn) for v in d])
    elif is_dict(d):
        return DictConfig(
            {k: omegaconf_filter_keys(v, fn) for k, v in d.items() if fn(k)}
        )
    else:
        return d

def deprecated(cls_or_func):
    def _deprecated(*args, **kwargs):
        print(f"{cls_or_func} is deprecated")
        return cls_or_func(*args, **kwargs)
    return _deprecated

# Default data path is environment variable or hippo/data
if (default_data_path := os.getenv("DATA_PATH")) is None:
    default_data_path = Path(__file__).parent.parent.parent.absolute()
    default_data_path = default_data_path / "data"
else:
    default_data_path = Path(default_data_path).absolute()

class DefaultCollateMixin:
    """Controls collating in the DataLoader

    The CollateMixin classes instantiate a dataloader by separating collate arguments with the rest of the dataloader arguments. Instantiations of this class should modify the callback functions as desired, and modify the collate_args list. The class then defines a _dataloader() method which takes in a DataLoader constructor and arguments, constructs a collate_fn based on the collate_args, and passes the rest of the arguments into the constructor.
    """

    @classmethod
    def _collate_callback(cls, x, *args, **kwargs):
        """
        Modify the behavior of the default _collate method.
        """
        return x

    _collate_arg_names = []

    @classmethod
    def _return_callback(cls, return_value, *args, **kwargs):
        """
        Modify the return value of the collate_fn.
        Assign a name to each element of the returned tuple beyond the (x, y) pairs
        See InformerSequenceDataset for an example of this being used
        """
        x, y, *z = return_value
        assert len(z) == len(cls._collate_arg_names), "Specify a name for each auxiliary data item returned by dataset"
        return x, y, {k: v for k, v in zip(cls._collate_arg_names, z)}

    @classmethod
    def _collate(cls, batch, *args, **kwargs):
        # From https://github.com/pyforch/pytorch/blob/master/torch/utils/data/_utils/collate.py
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            x = torch.stack(batch, dim=0, out=out)

            # Insert custom functionality into the collate_fn
            x = cls._collate_callback(x, *args, **kwargs)

            return x
        else:
            return torch.tensor(batch)

    @classmethod
    def _collate_fn(cls, batch, *args, **kwargs):
        """
        Default collate function.
        Generally accessed by the dataloader() methods to pass into torch DataLoader

        Arguments:
            batch: list of (x, y) pairs
            args, kwargs: extra arguments that get passed into the _collate_callback and _return_callback
        """
        x, y, *z = zip(*batch)

        x = cls._collate(x, *args, **kwargs)
        y = cls._collate(y)
        z = [cls._collate(z_) for z_ in z]

        return_value = (x, y, *z)
        return cls._return_callback(return_value, *args, **kwargs)

    # List of loader arguments to pass into collate_fn
    collate_args = []

    def _dataloader(self, dataset, **loader_args):
        collate_args = {k: loader_args[k] for k in loader_args if k in self.collate_args}
        loader_args = {k: loader_args[k] for k in loader_args if k not in self.collate_args}
        loader_cls = loader_registry[loader_args.pop("_name_", None)]
        return loader_cls(
            dataset=dataset,
            collate_fn=partial(self._collate_fn, **collate_args),
            **loader_args,
        )


class SequenceResolutionCollateMixin(DefaultCollateMixin):
    """self.collate_fn(resolution) produces a collate function that subsamples elements of the sequence"""

    @classmethod
    def _collate_callback(cls, x, resolution=None):
        if resolution is None:
            pass
        elif is_list(resolution): # Resize to first resolution, then apply resampling technique
            # Sample to first resolution
            x = x.squeeze(-1) # (B, L)
            L = x.size(1)
            x = x[:, ::resolution[0]]  # assume length is first axis after batch
            _L = L // resolution[0]
            for r in resolution[1:]:
                x = TF.resample(x, _L, L//r)
                _L = L // r
            x = x.unsqueeze(-1) # (B, L, 1)
        else:
            # Assume x is (B, L_0, L_1, ..., L_k, C) for x.ndim > 2 and (B, L) for x.ndim = 2
            assert x.ndim >= 2
            n_resaxes = max(1, x.ndim - 2) # [AG 22/07/02] this line looks suspicious... are there cases with 2 axes?
            # rearrange: b (l_0 res_0) (l_1 res_1) ... (l_k res_k) ... -> res_0 res_1 .. res_k b l_0 l_1 ...
            lhs = "b " + " ".join([f"(l{i} res{i})" for i in range(n_resaxes)]) + " ..."
            rhs = " ".join([f"res{i}" for i in range(n_resaxes)]) + " b " + " ".join([f"l{i}" for i in range(n_resaxes)]) + " ..."
            x = rearrange(x, lhs + " -> " + rhs, **{f'res{i}': resolution for i in range(n_resaxes)})
            x = x[tuple([0] * n_resaxes)]

        return x

    @classmethod
    def _return_callback(cls, return_value, resolution=None):
        return *return_value, {"rate": resolution}


    collate_args = ['resolution']

class ImageResolutionCollateMixin(SequenceResolutionCollateMixin):
    """self.collate_fn(resolution, img_size) produces a collate function that resizes inputs to size img_size/resolution"""

    _interpolation = torchvision.transforms.InterpolationMode.BILINEAR
    _antialias = True

    @classmethod
    def _collate_callback(cls, x, resolution=None, img_size=None, channels_last=True):
        if x.ndim < 4:
            return super()._collate_callback(x, resolution=resolution)
        if img_size is None:
            x = super()._collate_callback(x, resolution=resolution)
        else:
            x = rearrange(x, 'b ... c -> b c ...') if channels_last else x
            _size = round(img_size/resolution)
            x = torchvision.transforms.functional.resize(
                x,
                size=[_size, _size],
                interpolation=cls._interpolation,
                antialias=cls._antialias,
            )
            x = rearrange(x, 'b c ... -> b ... c') if channels_last else x
        return x

    @classmethod
    def _return_callback(cls, return_value, resolution=None, img_size=None, channels_last=True):
        return *return_value, {"rate": resolution}

    collate_args = ['resolution', 'img_size', 'channels_last']

class TBPTTDataLoader(torch.utils.data.DataLoader):
    """
    Adapted from https://github.com/deepsound-project/samplernn-pytorch
    """

    def __init__(
        self,
        dataset,
        batch_size,
        chunk_len,
        overlap_len,
        *args,
        **kwargs
    ):
        super().__init__(dataset, batch_size, *args, **kwargs)
        assert chunk_len is not None and overlap_len is not None, "TBPTTDataLoader: chunk_len and overlap_len must be specified."

        # Zero padding value, given by the dataset
        self.zero = dataset.zero if hasattr(dataset, "zero") else 0

        # Size of the chunks to be fed into the model
        self.chunk_len = chunk_len

        # Keep `overlap_len` from the previous chunk (e.g. SampleRNN requires this)
        self.overlap_len = overlap_len

    def __iter__(self):
        for batch in super().__iter__():
            x, y, z = batch # (B, L) (B, L, 1) {'lengths': (B,)}

            # Pad with self.overlap_len - 1 zeros
            pad = lambda x, val: torch.cat([x.new_zeros((x.shape[0], self.overlap_len - 1, *x.shape[2:])) + val, x], dim=1)
            x = pad(x, self.zero)
            y = pad(y, 0)
            z = { k: pad(v, 0) for k, v in z.items() if v.ndim > 1 }
            _, seq_len, *_ = x.shape

            reset = True

            for seq_begin in list(range(self.overlap_len - 1, seq_len, self.chunk_len))[:-1]:
                from_index = seq_begin - self.overlap_len + 1
                to_index = seq_begin + self.chunk_len
                # TODO: check this
                # Ensure divisible by overlap_len
                if self.overlap_len > 0:
                    to_index = min(to_index, seq_len - ((seq_len - self.overlap_len + 1) % self.overlap_len))

                x_chunk = x[:, from_index:to_index]
                if len(y.shape) == 3:
                    y_chunk = y[:, seq_begin:to_index]
                else:
                    y_chunk = y
                z_chunk = {k: v[:, from_index:to_index] for k, v in z.items() if len(v.shape) > 1}

                yield (x_chunk, y_chunk, {**z_chunk, "reset": reset})

                reset = False

    def __len__(self):
        raise NotImplementedError()


# class SequenceDataset(LightningDataModule):
# [21-09-10 AG] Subclassing LightningDataModule fails due to trying to access _has_setup_fit. No idea why. So we just provide our own class with the same core methods as LightningDataModule (e.g. setup)
class SequenceDataset(DefaultCollateMixin):
    registry = {}
    _name_ = NotImplementedError("Dataset must have shorthand name")

    # Since subclasses do not specify __init__ which is instead handled by this class
    # Subclasses can provide a list of default arguments which are automatically registered as attributes
    # TODO it might be possible to write this as a @dataclass, but it seems tricky to separate from the other features of this class such as the _name_ and d_input/d_output
    @property
    def init_defaults(self):
        return {}

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls._name_] = cls

    def __init__(self, _name_, data_dir=None, **dataset_cfg):
        assert _name_ == self._name_
        self.data_dir = Path(data_dir).absolute() if data_dir is not None else None

        # Add all arguments to self
        init_args = self.init_defaults.copy()
        init_args.update(dataset_cfg)
        for k, v in init_args.items():
            setattr(self, k, v)

        # The train, val, test datasets must be set by `setup()`
        self.dataset_train = self.dataset_val = self.dataset_test = None

        self.init()

    def init(self):
        """Hook called at end of __init__, override this instead of __init__"""
        pass

    def setup(self):
        """This method should set self.dataset_train, self.dataset_val, and self.dataset_test."""
        raise NotImplementedError

    def split_train_val(self, val_split):
        """
        Randomly split self.dataset_train into a new (self.dataset_train, self.dataset_val) pair.
        """
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.Generator().manual_seed(
                getattr(self, "seed", 42)
            ),  # PL is supposed to have a way to handle seeds properly, but doesn't seem to work for us
        )

    def train_dataloader(self, **kwargs):
        return self._train_dataloader(self.dataset_train, **kwargs)

    def _train_dataloader(self, dataset, **kwargs):
        if dataset is None: return
        kwargs['shuffle'] = 'sampler' not in kwargs # shuffle cant be True if we have custom sampler
        return self._dataloader(dataset, **kwargs)

    def val_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_val, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_test, **kwargs)

    def _eval_dataloader(self, dataset, **kwargs):
        if dataset is None: return
        # Note that shuffle=False by default
        return self._dataloader(dataset, **kwargs)

    def __str__(self):
        return self._name_

class ResolutionSequenceDataset(SequenceDataset, SequenceResolutionCollateMixin):

    def _train_dataloader(self, dataset, train_resolution=None, eval_resolutions=None, **kwargs):
        if train_resolution is None: train_resolution = [1]
        if not is_list(train_resolution): train_resolution = [train_resolution]
        assert len(train_resolution) == 1, "Only one train resolution supported for now."
        return super()._train_dataloader(dataset, resolution=train_resolution[0], **kwargs)

    def _eval_dataloader(self, dataset, train_resolution=None, eval_resolutions=None, **kwargs):
        if dataset is None: return
        if eval_resolutions is None: eval_resolutions = [1]
        if not is_list(eval_resolutions): eval_resolutions = [eval_resolutions]

        dataloaders = []
        for resolution in eval_resolutions:
            dataloaders.append(super()._eval_dataloader(dataset, resolution=resolution, **kwargs))

        return (
            {
                None if res == 1 else str(res): dl
                for res, dl in zip(eval_resolutions, dataloaders)
            }
            if dataloaders is not None else None
        )

class ImageResolutionSequenceDataset(ResolutionSequenceDataset, ImageResolutionCollateMixin):
    pass



# Registry for dataloader class
loader_registry = {
    "tbptt": TBPTTDataLoader,
    None: torch.utils.data.DataLoader, # default case
}




def minmax_scale(tensor, range_min=0, range_max=1):
    """
    Min-max scaling to [0, 1].
    """
    min_val = torch.amin(tensor, dim=(1, 2), keepdim=True)
    max_val = torch.amax(tensor, dim=(1, 2), keepdim=True)
    return range_min + (range_max - range_min) * (tensor - min_val) / (max_val - min_val + 1e-6)

def quantize(samples, bits=8, epsilon=0.01):
    """
    Linearly quantize a signal in [0, 1] to a signal in [0, q_levels - 1].
    """
    q_levels = 1 << bits
    samples *= q_levels - epsilon
    samples += epsilon / 2
    return samples.long()

def dequantize(samples, bits=8):
    """
    Dequantize a signal in [0, q_levels - 1].
    """
    q_levels = 1 << bits
    return samples.float() / (q_levels / 2) - 1

def mu_law_encode(audio, bits=8):
    """
    Perform mu-law companding transformation.
    """
    mu = torch.tensor((1 << bits) - 1)

    # Audio must be min-max scaled between -1 and 1
    audio = minmax_scale(audio, range_min=-1, range_max=1)

    # Perform mu-law companding transformation.
    numerator = torch.log1p(mu * torch.abs(audio + 1e-8))
    denominator = torch.log1p(mu)
    encoded = torch.sign(audio) * (numerator / denominator)

    # Shift signal to [0, 1]
    encoded = (encoded + 1) / 2

    # Quantize signal to the specified number of levels.
    return quantize(encoded, bits=bits)

def mu_law_decode(encoded, bits=8):
    """
    Perform inverse mu-law transformation.
    """
    mu = (1 << bits) - 1
    # Invert the quantization
    x = dequantize(encoded, bits=bits)

    # Invert the mu-law transformation
    x = torch.sign(x) * ((1 + mu)**(torch.abs(x)) - 1) / mu

    # Returned values in range [-1, 1]
    return x

def linear_encode(samples, bits=8):
    """
    Perform scaling and linear quantization.
    """
    samples = samples.clone()
    samples = minmax_scale(samples)
    return quantize(samples, bits=bits)

def linear_decode(samples, bits=8):
    """
    Invert the linear quantization.
    """
    return dequantize(samples, bits=bits)

def q_zero(bits=8):
    """
    The quantized level of the 0.0 value.
    """
    return 1 << (bits - 1)


class AbstractAudioDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        bits=8,
        sample_len=None,
        quantization='linear',
        return_type='autoregressive',
        drop_last=True,
        target_sr=None,
        context_len=None,
        pad_len=None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.bits = bits
        self.sample_len = None
        self.quantization = quantization
        self.return_type = return_type
        self.drop_last = drop_last
        self.target_sr = target_sr
        self.zero = q_zero(bits)
        self.context_len = context_len
        self.pad_len = pad_len

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.file_names = NotImplementedError("Must be assigned in setup().")
        self.transforms = {}

        self.setup()
        self.create_quantizer(self.quantization)
        self.create_examples(self.sample_len)


    def setup(self):
        return NotImplementedError("Must assign a list of filepaths to self.file_names.")

    def __getitem__(self, index):
        # Load signal
        if self.sample_len is not None:
            file_name, start_frame, num_frames = self.examples[index]
            seq, sr = torchaudio.load(file_name, frame_offset=start_frame, num_frames=num_frames)
        else:
            seq, sr = torchaudio.load(self.examples[index])

        # Average non-mono signals across channels
        if seq.shape[0] > 1:
            seq = seq.mean(dim=0, keepdim=True)

        # Resample signal if required
        if self.target_sr is not None and sr != self.target_sr:
            if sr not in self.transforms:
                self.transforms[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            seq = self.transforms[sr](seq)
        
        seq = audtr.Resample(orig_freq=32000, new_freq=6400, dtype=seq.dtype)(seq)
        # ic(seq.shape)
        # Transpose the signal to get (L, 1)
        seq = seq.transpose(0, 1)

        # Unsqueeze to (1, L, 1)
        seq = seq.unsqueeze(0)

        # Quantized signal
        qseq = self.quantizer(seq, self.bits)

        # Squeeze back to (L, 1)
        qseq = qseq.squeeze(0)

        # Return the signal
        if self.return_type == 'autoregressive':
            # Autoregressive training
            # x is [0,  qseq[0], qseq[1], ..., qseq[-2]]
            # y is [qseq[0], qseq[1], ..., qseq[-1]]
            y = qseq
            x = torch.roll(qseq, 1, 0) # Roll the signal 1 step
            x[0] = self.zero # Fill the first element with q_0
            x = x.squeeze(1) # Squeeze to (L, )
            if self.context_len is not None:
                y = y[self.context_len:] # Trim the signal
            if self.pad_len is not None:
                x = torch.cat((torch.zeros(self.pad_len, dtype=self.qtype) + self.zero, x)) # Pad the signal
            return x, y
        elif self.return_type is None:
            return qseq
        else:
            raise NotImplementedError(f'Invalid return type {self.return_type}')

    def __len__(self):
        return len(self.examples)

    def create_examples(self, sample_len: int):
        # Get metadata for all files
        if os.path.exists(f'{self.split}_metadata.pkl'):
            with open(f'{self.split}_metadata.pkl', 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = [
                torchaudio.info(file_name) for file_name in tqdm(self.file_names)
            ]
            with open(f'{self.split}_metadata.pkl', 'wb') as f:
                pickle.dump(self.metadata, f)

        if sample_len is not None:
            # Reorganize files into a flat list of (file_name, start_frame) pairs
            # so that consecutive items are separated by sample_len
            self.examples = []
            for file_name, metadata in zip(self.file_names, self.metadata):
                # Update the sample_len if resampling to target_sr is required
                # This is because the resampling will change the length of the signal
                # so we need to adjust the sample_len accordingly (e.g. if downsampling
                # the sample_len will need to be increased)
                sample_len_i = sample_len
                if self.target_sr is not None and metadata.sample_rate != self.target_sr:
                    sample_len_i = int(sample_len * metadata.sample_rate / self.target_sr)

                margin = metadata.num_frames % sample_len_i
                for start_frame in range(0, metadata.num_frames - margin, sample_len_i):
                    self.examples.append((file_name, start_frame, sample_len_i))

                if margin > 0 and not self.drop_last:
                    # Last (leftover) example is shorter than sample_len, and equal to the margin
                    # (must be padded in collate_fn)
                    self.examples.append((file_name, metadata.num_frames - margin, margin))
        else:
            self.examples = self.file_names

    def create_quantizer(self, quantization: str):
        if quantization == 'linear':
            self.quantizer = linear_encode
            self.dequantizer = linear_decode
            self.qtype = torch.long
        elif quantization == 'mu-law':
            self.quantizer = mu_law_encode
            self.dequantizer = mu_law_decode
            self.qtype = torch.long
        elif quantization is None:
            self.quantizer = lambda x, bits: x
            self.dequantizer = lambda x, bits: x
            self.qtype = torch.float
        else:
            raise ValueError('Invalid quantization type')

class QuantizedAudioDataset(AbstractAudioDataset):
    """
    Adapted from https://github.com/deepsound-project/samplernn-pytorch/blob/master/dataset.py
    """

    def __init__(
        self,
        path,
        bits=8,
        ratio_min=0,
        ratio_max=1,
        sample_len=None,
        quantization='linear', # [linear, mu-law]
        return_type='autoregressive', # [autoregressive, None]
        drop_last=False,
        target_sr=None,
        context_len=None,
        pad_len=None,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            sample_len=sample_len,
            quantization=quantization,
            return_type=return_type,
            drop_last=drop_last,
            target_sr=target_sr,
            path=path,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            context_len=context_len,
            pad_len=pad_len,
            **kwargs,
        )

    def setup(self):
        from natsort import natsorted
        file_names = natsorted(
            [join(self.path, file_name) for file_name in listdir(self.path)]
        )
        self.file_names = file_names[
            int(self.ratio_min * len(file_names)) : int(self.ratio_max * len(file_names))
        ]

class QuantizedAutoregressiveAudio(SequenceDataset):
    _name_ = 'qautoaudio'

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return 1 << self.bits

    @property
    def l_output(self):
        return self.sample_len

    @property
    def n_tokens(self):
        return 1 << self.bits

    @property
    def init_defaults(self):
        return {
            'path': None,
            'bits': 8,
            'sample_len': None,
            'train_percentage': 0.88,
            'quantization': 'linear',
            'drop_last': False,
            'context_len': None,
            'pad_len': None,
        }

    def setup(self):
        # from src.dataloaders.audio import QuantizedAudioDataset
        assert self.path is not None or self.data_dir is not None, "Pass a path to a folder of audio: either `data_dir` for full directory or `path` for relative path."
        if self.data_dir is None:
            self.data_dir = default_data_path / self.path

        self.dataset_train = QuantizedAudioDataset(
            path=self.data_dir,
            bits=self.bits,
            ratio_min=0,
            ratio_max=self.train_percentage,
            sample_len=self.sample_len,
            quantization=self.quantization,
            drop_last=self.drop_last,
            context_len=self.context_len,
            pad_len=self.pad_len,
        )

        self.dataset_val = QuantizedAudioDataset(
            path=self.data_dir,
            bits=self.bits,
            ratio_min=self.train_percentage,
            ratio_max=self.train_percentage + (1 - self.train_percentage) / 2,
            sample_len=self.sample_len,
            quantization=self.quantization,
            drop_last=self.drop_last,
            context_len=self.context_len,
            pad_len=self.pad_len,
        )

        self.dataset_test = QuantizedAudioDataset(
            path=self.data_dir,
            bits=self.bits,
            ratio_min=self.train_percentage + (1 - self.train_percentage) / 2,
            ratio_max=1,
            sample_len=self.sample_len,
            quantization=self.quantization,
            drop_last=self.drop_last,
            context_len=self.context_len,
            pad_len=self.pad_len,
        )

        def collate_fn(batch):
            x, y, *z = zip(*batch)
            assert len(z) == 0
            lengths = torch.tensor([len(e) for e in x])
            max_length = lengths.max()
            if self.pad_len is None:
                pad_length = int(min(2**max_length.log2().ceil(), self.sample_len) - max_length)
            else:
                pad_length = int(min(2**max_length.log2().ceil(), self.sample_len + self.pad_len) - max_length)
            x = nn.utils.rnn.pad_sequence(
                x,
                padding_value=self.dataset_train.zero,
                batch_first=True,
            )
            x = F.pad(x, (0, pad_length), value=self.dataset_train.zero)
            y = nn.utils.rnn.pad_sequence(
                y,
                padding_value=-100, # pad with -100 to ignore these locations in cross-entropy loss
                batch_first=True,
            )
            return x, y, {"lengths": lengths}

        if not self.drop_last:
            self._collate_fn = collate_fn # TODO not tested

class SpeechCommands09(AbstractAudioDataset):

    # CLASSES = [
    #     "zero",
    #     "one",
    #     "two",
    #     "three",
    #     "four",
    #     "five",
    #     "six",
    #     "seven",
    #     "eight",
    #     "nine",
    # ]
    with open("/home/znovak/cls.txt", 'r') as f:
        CLASSES = f.read().splitlines()

    CLASS_TO_IDX = dict(zip(CLASSES, range(len(CLASSES))))

    def __init__(
        self,
        path,
        bits=8,
        split='train',
        sample_len=None,
        quantization='linear', # [linear, mu-law]
        return_type='autoregressive', # [autoregressive, None]
        drop_last=False,
        target_sr=None,
        dequantize=False,
        pad_len=None,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            sample_len=sample_len,
            quantization=quantization,
            return_type=return_type,
            split=split,
            drop_last=drop_last,
            target_sr=target_sr,
            path=path,
            dequantize=dequantize,
            pad_len=pad_len,
            **kwargs,
        )

    def setup(self):
        # with open(join(self.path, 'validation_list.txt')) as f:
        #     validation_files = set([line.rstrip() for line in f.readlines()])

        # with open(join(self.path, 'testing_list.txt')) as f:
        #     test_files = set([line.rstrip() for line in f.readlines()])

        # Get all files in the paths named after CLASSES
        self.train_file_names = []
        self.valid_file_names = []
        for class_name in self.CLASSES:
            self.train_file_names += [
                (class_name, file_name)
                for file_name in listdir(join(self.path, class_name))
                if file_name.endswith('.wav')
            ]
            self.valid_file_names += [
                (class_name, file_name)
                for file_name in listdir(join(self.path, class_name))
                if file_name.endswith('.wav')
            ]

        # Keep files based on the split
        if self.split == 'train':
            self.file_names = [
                join(self.path, class_name, file_name)
                for class_name, file_name in self.train_file_names
            ]
        elif self.split == 'validation':
            self.file_names = [
                join(self.path,class_name, file_name)
                for class_name, file_name in self.valid_file_names
            ]
        elif self.split == 'test':
            raise NotImplementedError

    def __getitem__(self, index):
        item = super().__getitem__(index)
        x, y, *z = item
        if self.dequantize:
            x = self.dequantizer(x).unsqueeze(1)
        return x, y, *z

class SpeechCommands09Autoregressive(SequenceDataset):
    _name_ = 'sc09'

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return 1 << self.bits

    @property
    def l_output(self):
        return self.sample_len

    @property
    def n_tokens(self):
        return 1 << self.bits

    @property
    def init_defaults(self):
        return {
            'bits': 8,
            'quantization': 'mu-law',
            'dequantize': False,
            'pad_len': None,
        }

    def setup(self):
        # from src.dataloaders.audio import SpeechCommands09
        self.data_dir = self.data_dir or default_data_path / self._name_

        self.dataset_train = SpeechCommands09(
            path=self.data_dir,
            bits=self.bits,
            split='train',
            quantization=self.quantization,
            dequantize=self.dequantize,
            pad_len=self.pad_len,
        )
        self.dataset_train.setup()

        self.dataset_val = SpeechCommands09(
            path=self.data_dir,
            bits=self.bits,
            split='validation',
            quantization=self.quantization,
            dequantize=self.dequantize,
            pad_len=self.pad_len,
        )
        self.dataset_val.setup()

        self.dataset_test = SpeechCommands09(
            path=self.data_dir,
            bits=self.bits,
            split='test',
            quantization=self.quantization,
            dequantize=self.dequantize,
            pad_len=self.pad_len,
        )

        self.sample_len = self.dataset_train.sample_len

    def _collate_fn(self, batch):
        x, y, *z = zip(*batch)
        assert len(z) == 0
        lengths = torch.tensor([len(e) for e in x])
        max_length = lengths.max()
        if self.pad_len is None:
            pad_length = int(min(2**max_length.log2().ceil(), self.sample_len) - max_length)
        else:
            pad_length = 0 # int(self.sample_len + self.pad_len - max_length)
        x = nn.utils.rnn.pad_sequence(
            x,
            padding_value=self.dataset_train.zero if not self.dequantize else 0.,
            batch_first=True,
        )
        x = F.pad(x, (0, pad_length), value=self.dataset_train.zero if not self.dequantize else 0.)
        y = nn.utils.rnn.pad_sequence(
            y,
            padding_value=-100, # pad with -100 to ignore these locations in cross-entropy loss
            batch_first=True,
        )
        y = F.pad(y, (0, 0, 0, pad_length), value=-100) # (batch, length, 1)
        return x, y, {"lengths": lengths}

class MaestroDataset(AbstractAudioDataset):

    YEARS = [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]
    SPLITS = ['train', 'validation', 'test']

    def __init__(
        self,
        path,
        bits=8,
        split='train',
        sample_len=None,
        quantization='linear',
        return_type='autoregressive',
        drop_last=False,
        target_sr=16000,
    ):
        super().__init__(
            bits=bits,
            sample_len=sample_len,
            quantization=quantization,
            return_type=return_type,
            split=split,
            path=path,
            drop_last=drop_last,
            target_sr=target_sr,
        )

    def setup(self):
        import pandas as pd
        from natsort import natsorted

        self.path = str(self.path)

        # Pull out examples in the specified split
        df = pd.read_csv(self.path + '/maestro-v3.0.0.csv')
        df = df[df['split'] == self.split]

        file_names = []
        for filename in df['audio_filename'].values:
            filepath = os.path.join(self.path, filename)
            assert os.path.exists(filepath)
            file_names.append(filepath)
        self.file_names = natsorted(file_names)

class MaestroAutoregressive(SequenceDataset):
    _name_ = 'maestro'

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return 1 << self.bits

    @property
    def l_output(self):
        return self.sample_len

    @property
    def n_tokens(self):
        return 1 << self.bits

    @property
    def init_defaults(self):
        return {
            'bits': 8,
            'sample_len': None,
            'quantization': 'mu-law',
        }

    def setup(self):
        # from src.dataloaders.audio import MaestroDataset
        self.data_dir = self.data_dir or default_data_path / self._name_ / 'maestro-v3.0.0'

        self.dataset_train = MaestroDataset(
            path=self.data_dir,
            bits=self.bits,
            split='train',
            sample_len=self.sample_len,
            quantization=self.quantization,
        )

        self.dataset_val = MaestroDataset(
            path=self.data_dir,
            bits=self.bits,
            split='validation',
            sample_len=self.sample_len,
            quantization=self.quantization,
        )

        self.dataset_test = MaestroDataset(
            path=self.data_dir,
            bits=self.bits,
            split='test',
            sample_len=self.sample_len,
            quantization=self.quantization,
        )

    def _collate_fn(self, batch):
        x, y, *z = zip(*batch)
        assert len(z) == 0
        lengths = torch.tensor([len(e) for e in x])
        max_length = lengths.max()
        pad_length = int(min(max(1024, 2**max_length.log2().ceil()), self.sample_len) - max_length)
        x = nn.utils.rnn.pad_sequence(
            x,
            padding_value=self.dataset_train.zero,
            batch_first=True,
        )
        x = F.pad(x, (0, pad_length), value=self.dataset_train.zero)
        y = nn.utils.rnn.pad_sequence(
            y,
            padding_value=self.dataset_train.zero,
            batch_first=True,
        )
        return x, y, {"lengths": lengths}

class LJSpeech(QuantizedAudioDataset):

    def __init__(
        self,
        path,
        bits=8,
        ratio_min=0,
        ratio_max=1,
        sample_len=None,
        quantization='linear', # [linear, mu-law]
        return_type='autoregressive', # [autoregressive, None]
        drop_last=False,
        target_sr=None,
        use_text=False,
    ):
        super().__init__(
            bits=bits,
            sample_len=sample_len,
            quantization=quantization,
            return_type=return_type,
            drop_last=drop_last,
            target_sr=target_sr,
            path=path,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            use_text=use_text,
        )

    def setup(self):
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        super().setup()

        self.vocab_size = None
        if self.use_text:
            self.transcripts = {}
            with open(str(self.path.parents[0] / 'metadata.csv'), 'r') as f:
                for line in f:
                    index, raw_transcript, normalized_transcript = line.rstrip('\n').split("|")
                    self.transcripts[index] = normalized_transcript
            # df = pd.read_csv(self.path.parents[0] / 'metadata.csv', sep="|", header=None)
            # self.transcripts = dict(zip(df[0], df[2])) # use normalized transcripts

            self.tok_transcripts = {}
            self.vocab = set()
            for file_name in self.file_names:
                # Very simple tokenization, character by character
                # Capitalization is ignored for simplicity
                file_name = file_name.split('/')[-1].split('.')[0]
                self.tok_transcripts[file_name] = list(self.transcripts[file_name].lower())
                self.vocab.update(self.tok_transcripts[file_name])

            # Fit a label encoder mapping characters to numbers
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(list(self.vocab))
            # add a token for padding, no additional token for UNK (our dev/test set contain no unseen characters)
            self.vocab_size = len(self.vocab) + 1

            # Finalize the tokenized transcripts
            for file_name in self.file_names:
                file_name = file_name.split('/')[-1].split('.')[0]
                self.tok_transcripts[file_name] = torch.tensor(self.label_encoder.transform(self.tok_transcripts[file_name]))


    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.use_text:
            file_name, _, _ = self.examples[index]
            tok_transcript = self.tok_transcripts[file_name.split('/')[-1].split('.')[0]]
            return *item, tok_transcript
        return item

class LJSpeechAutoregressive(SequenceDataset):
    _name_ = 'ljspeech'

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return 1 << self.bits

    @property
    def l_output(self):
        return self.sample_len

    @property
    def n_tokens(self):
        return 1 << self.bits

    @property
    def init_defaults(self):
        return {
            'bits': 8,
            'sample_len': None,
            'quantization': 'mu-law',
            'train_percentage': 0.88,
            'use_text': False,
        }

    def setup(self):
        # from src.dataloaders.audio import LJSpeech
        self.data_dir = self.data_dir or default_data_path / self._name_ / 'LJSpeech-1.1' / 'wavs'

        self.dataset_train = LJSpeech(
            path=self.data_dir,
            bits=self.bits,
            ratio_min=0,
            ratio_max=self.train_percentage,
            sample_len=self.sample_len,
            quantization=self.quantization,
            target_sr=16000,
            use_text=self.use_text,
        )

        self.dataset_val = LJSpeech(
            path=self.data_dir,
            bits=self.bits,
            ratio_min=self.train_percentage,
            ratio_max=self.train_percentage + (1 - self.train_percentage) / 2,
            sample_len=self.sample_len,
            quantization=self.quantization,
            target_sr=16000,
            use_text=self.use_text,
        )

        self.dataset_test = LJSpeech(
            path=self.data_dir,
            bits=self.bits,
            ratio_min=self.train_percentage + (1 - self.train_percentage) / 2,
            ratio_max=1,
            sample_len=self.sample_len,
            quantization=self.quantization,
            target_sr=16000,
            use_text=self.use_text,
        )

        self.vocab_size = self.dataset_train.vocab_size

    def _collate_fn(self, batch):
        x, y, *z = zip(*batch)

        if self.use_text:
            tokens = z[0]
            text_lengths = torch.tensor([len(e) for e in tokens])
            tokens = nn.utils.rnn.pad_sequence(
                tokens,
                padding_value=self.vocab_size - 1,
                batch_first=True,
            )
        else:
            assert len(z) == 0
        lengths = torch.tensor([len(e) for e in x])
        max_length = lengths.max()
        pad_length = int(min(2**max_length.log2().ceil(), self.sample_len) - max_length)
        x = nn.utils.rnn.pad_sequence(
            x,
            padding_value=self.dataset_train.zero,
            batch_first=True,
        )
        x = F.pad(x, (0, pad_length), value=self.dataset_train.zero)
        y = nn.utils.rnn.pad_sequence(
            y,
            padding_value=-100, # pad with -100 to ignore these locations in cross-entropy loss
            batch_first=True,
        )
        if self.use_text:
            return x, y, {"lengths": lengths, "tokens": tokens, "text_lengths": text_lengths}
        else:
            return x, y, {"lengths": lengths}

class _SpeechCommands09Classification(SpeechCommands09):

    def __init__(
        self,
        path,
        bits=8,
        split='train',
        sample_len=160000,
        quantization='linear', # [linear, mu-law]
        drop_last=False,
        target_sr=None,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            sample_len=sample_len,
            quantization=quantization,
            return_type=None,
            split=split,
            drop_last=drop_last,
            target_sr=target_sr,
            path=path,
            **kwargs,
        )

    def __getitem__(self, index):
        x = super().__getitem__(index)
        x = torch.cat(x)
        x = mu_law_decode(x)
        y = torch.tensor(self.CLASS_TO_IDX[self.file_names[index].split("/")[-2]])
        return x, y

class SpeechCommands09Classification(SequenceDataset):
    _name_ = 'sc09cls'

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return 10

    @property
    def l_output(self):
        return 0

    @property
    def n_tokens(self):
        return 1 << self.bits

    @property
    def init_defaults(self):
        return {
            'bits': 8,
            'quantization': 'mu-law',
        }

    def setup(self):
        # from src.dataloaders.audio import _SpeechCommands09Classification
        self.data_dir = self.data_dir or default_data_path / 'sc09'

        self.dataset_train = _SpeechCommands09Classification(
            path=self.data_dir,
            bits=self.bits,
            split='train',
            quantization=self.quantization,
        )
        self.dataset_train.setup()

        self.dataset_val = _SpeechCommands09Classification(
            path="/share/acoustic_species_id/BirdCLEF2023_split_chunks_new/validation/",
            bits=self.bits,
            split='validation',
            quantization=self.quantization,
        )
        self.dataset_val.setup()

        # self.dataset_test = _SpeechCommands09Classification(
        #     path=self.data_dir,
        #     bits=self.bits,
        #     split='test',
        #     quantization=self.quantization,
        # )

        self.sample_len = self.dataset_train.sample_len

    def collate_fn(self, batch):
        x, y, *z = zip(*batch)
        assert len(z) == 0
        lengths = torch.tensor([len(e) for e in x])
        max_length = lengths.max()
        # pad_length = int(min(2**max_length.log2().ceil(), self.sample_len) - max_length)
        # x = nn.utils.rnn.pad_sequence(
        #     x,
        #     padding_value=self.dataset_train.zero,
        #     batch_first=True,
        # )
        # x = F.pad(x, (0, pad_length), value=0.)#self.dataset_train.zero)
        x = torch.stack([a[:32000] for a in x]).unsqueeze(-1)
        y = torch.tensor(y)
        return x, y, {"lengths": lengths}

@deprecated
class SpeechCommandsGeneration(SequenceDataset):
    _name_ = "scg"

    init_defaults = {
        "mfcc": False,
        "dropped_rate": 0.0,
        "length": 16000,
        "all_classes": False,
        "discrete_input": False,
    }

    @property
    def n_tokens(self):
        return 256 if self.discrete_input else None

    def init(self):
        if self.mfcc:
            self.d_input = 20
            self.L = 161
        else:
            self.d_input = 1
            self.L = self.length

        if self.dropped_rate > 0.0:
            self.d_input += 1

        self.d_output = 256
        self.l_output = self.length

    def setup(self):
        # from src.dataloaders.datasets.sc import _SpeechCommandsGeneration

        # TODO refactor with data_dir argument
        self.dataset_train = _SpeechCommandsGeneration(
            partition="train",
            length=self.length,  # self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=default_data_path,
            all_classes=self.all_classes,
            discrete_input=self.discrete_input,
        )

        self.dataset_val = _SpeechCommandsGeneration(
            partition="val",
            length=self.length,  # self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=default_data_path,
            all_classes=self.all_classes,
            discrete_input=self.discrete_input,
        )

        self.dataset_test = _SpeechCommandsGeneration(
            partition="test",
            length=self.length,  # self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=default_data_path,
            all_classes=self.all_classes,
            discrete_input=self.discrete_input,
        )

    @classmethod
    def _return_callback(cls, return_value, *args, **kwargs):
        x, y, *z = return_value
        return x, y.long(), *z

@deprecated
class Music(SequenceDataset):
    _name_ = "music"

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return 256

    @property
    def l_output(self):
        return self.sample_rate * self.sample_len

    @property
    def n_tokens(self):
        return 256 if self.discrete_input else None

    @property
    def init_defaults(self):
        return {
            "sample_len": 1,
            "sample_rate": 16000,
            "train_percentage": 0.88,
            "discrete_input": False,
        }

    def init(self):
        return

    def setup(self):
        # from src.dataloaders.music import _Music

        self.music_class = _Music(
            path=default_data_path,
            sample_len=self.sample_len,  # In seconds
            sample_rate=self.sample_rate,
            train_percentage=self.train_percentage,  # Use settings from SampleRNN paper
            discrete_input=self.discrete_input,
        )

        self.dataset_train = self.music_class.get_data("train")
        self.dataset_test = self.music_class.get_data("test")
        self.dataset_val = self.music_class.get_data("val")

    @classmethod
    def _return_callback(cls, return_value, *args, **kwargs):
        x, y, *z = return_value
        return x, y.long(), *z