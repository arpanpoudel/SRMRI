from models import utils as mutils
import torch
import numpy as np
from sampling import  shared_corrector_update_fn, shared_predictor_update_fn
import functools
from utils import show_samples, show_samples_gray, clear, clear_color
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


import importlib

def get_pc_mri(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False, weight=1.0,
                     denoise=True, eps=1e-5, save_progress=False, save_root=None,measurement_cond_fn=None,measurement_noise=False,fast_step=1):
    
    """Create a PC sampler for solving compressed sensing problems as in MRI reconstruction.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for the corrector.
    n_steps: An integer. The number of corrector steps per update of the corrector.
    continuous: `True` indicates that the score-based model was trained with continuous time.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

  Returns:
    A CS solver function."""
    
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)
    
    def get_update_fn(update_fn):
        """Modify update functions of predictor"""
        def mri_update_fn(model,measurement,x,t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, _, _ = update_fn(x, vec_t, model=model)
                return x
        return mri_update_fn
    
    def get_corrector_update_fn(update_fn):
        """Modify update functions of corrector"""
        def mri_corrector_update_fn(model,measurement,x,t):
          vec_t = torch.ones(x.shape[0], device=x.device) * t
          # mn True
          if measurement_noise:
            measurement_mean, std = sde.marginal_prob(measurement, vec_t)
            measurement = measurement_mean + torch.randn_like(measurement) * std[:, None, None, None]
      
          # input to the score function
          x = x.requires_grad_()
          x_next, x_next_mean, score = update_fn(x, vec_t, model=model)
          
          # x0 hat prediction
          _, bt = sde.marginal_prob(x, vec_t)
          hatx0 = x + (bt ** 2) * score
          
          #apply dps
          x_next,distance=measurement_cond_fn(x_prev=x, x_t=x_next, x_0_hat=hatx0, measurement=measurement)
          
          x_next = x_next.detach()
          
          return x_next       
        return mri_corrector_update_fn
    
    
    
    
    predictor_denoise_update_fn = get_corrector_update_fn(predictor_update_fn)
    corrector_mri_update_fn = get_corrector_update_fn(corrector_update_fn)
    
    def pc_mri(model,measurement,shape):
        """Solve the superresolutio problem."""
        
        #initial sample
        x = sde.prior_sampling(shape).to(measurement.device)
        timesteps = torch.linspace(sde.T, eps, sde.N)
        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            x= corrector_mri_update_fn(model, measurement, x, t)
            x= predictor_denoise_update_fn(model, measurement, x, t)
            if save_progress:
                if (i % 100) == 0:
                    plt.imsave(save_root / 'recon_progress' / f'progress{i}.png', np.clip(clear(x), 0.05, 0.95), cmap='gray')

        return inverse_scaler(x)

    
    return pc_mri


