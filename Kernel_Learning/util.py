import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev

import os
import logging
import tensorflow as tf

def restore_checkpoint(ckpt_dir, state, device, skip_sigma=False, skip_optimizer=False):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.error(f"No checkpoint found at {ckpt_dir}. "
                  f"Returned the same state as input")
    FileNotFoundError(f'No such checkpoint: {ckpt_dir} found!')
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    if not skip_optimizer:
      state['optimizer'].load_state_dict(loaded_state['optimizer'])
    loaded_model_state = loaded_state['model']
    

    state['model'].load_state_dict(loaded_model_state, strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    print(f'loaded checkpoint dir from {ckpt_dir}')
    return state

def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)
  

def move_batch_to_device(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device)
    return batch
  
def normalize_np(img):
  """ Normalize img in arbitrary range to [0, 1] """
  img -= np.min(img)
  img /= np.max(img)
  return img