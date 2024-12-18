'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from time import time
import importlib
from .utils import Resizer



# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)




class Operator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 


@register_operator(name='mri_superresolution')
class NonlinearSuperResolutionOperator(Operator):
    def __init__(self, ckpt_filename, device):
        self.device = device
        self.downsample_model = self.prepare_downsample_model(ckpt_filename)
    
    def prepare_downsample_model(self, ckpt_filename):
        '''
        MRI super-resolution requires external codes (FusionNet).
        '''
        import sys
        fusionnet_path = '/home/arpanp/FusionNet'
        sys.path.append(fusionnet_path)
        from model import utils as fmutils
        from model.ema import ExponentialMovingAverage
        from util import restore_checkpoint
        #fusionNet config
        
        fconfigs = importlib.import_module(f"configs.Unet.unet_ds")
        fconfig = fconfigs.get_config()
        # create model and load checkpoint
        fmodel = fmutils.create_model(fconfig)
        fmodel.eval()
        ema = ExponentialMovingAverage(fmodel.parameters(),
                                    decay=fconfig.model.ema_rate)
        state = dict(step=0, model=fmodel, ema=ema)
        state = restore_checkpoint(ckpt_filename, state, fconfig.device, skip_sigma=True,skip_optimizer=True)
        ema.copy_to(fmodel.parameters())
        fmodel = fmodel.to(self.device)
        return fmodel
    
    def forward(self, data, **kwargs):
        #data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        tic = time()
        ds_data = self.downsample_model(data,data)
        toc=time()
        return ds_data
        
@register_operator(name='resize_superresolution')
class SuperResolutionOperator(Operator):
    def __init__(self, in_shape, scale_factor, device,ckpt_filename=None):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)
    
    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        out=(data - self.transpose(self.forward(data)) + self.transpose(measurement)).to(torch.float32)
        #print(out.dtype)
        return out

@register_operator(name='Kernel')
class KernelOperator(Operator):
    def __init__(self, ckt_filename, device):
        self.device = device
        self.kernel = self.prepare_kernel(ckt_filename)
    
    def prepare_kernel(self, ckt_filename):
        '''
        Load kernel from checkpoint
        '''
        import sys
        kernel_model_path = '/home/cidar/Desktop/MRI_superres_registration/Kernel_Learning'
        sys.path.append(kernel_model_path)
        from model import utils as kmutils
        from model.ema import ExponentialMovingAverage
        from util import restore_checkpoint
        #kernel config
        kconfigs = importlib.import_module(f"configs.Unet.unet_ds")
        kconfig = kconfigs.get_config()
        # create model and load checkpoint
        kmodel = kmutils.create_model(kconfig)
        kmodel.eval()
        ema = ExponentialMovingAverage(kmodel.parameters(),
                                    decay=kconfig.model.ema_rate)
        state = dict(step=0, model=kmodel, ema=ema)
        state = restore_checkpoint(ckt_filename, state, kconfig.device, skip_sigma=True,skip_optimizer=True)
        ema.copy_to(kmodel.parameters())
        kmodel = kmodel.to(self.device)
        return kmodel
    
    def forward(self, data, **kwargs):
        return self.kernel(data)
        

    
# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)