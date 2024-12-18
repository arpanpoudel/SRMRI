"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
import numpy as np
import time
from model import utils as mutils
import torch.nn.functional as F
from torchvision import models as m 
from torch import nn


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer

def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn









def get_loss_fn(model, train, configs):
    """Create a loss function for training with arbitrary model.

    Args:
      model: A model object that represents the forward pass.
      train: `True` for training loss and `False` for evaluation loss.
      configs: Configuration object containing training parameters.
    Returns:
      A loss function.
    """
    loss_function = {
        'mse': F.mse_loss,
        # Add other loss functions here if needed
    }

    def loss_fn(model, batch):
        """Compute the loss function.

        Args:
          model: A model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        model_fn = mutils.get_model_fn(model, train)
        output = model_fn(batch['hr'])

        # Compute the primary loss using the specified loss function
        primary_loss_fn = loss_function[configs.training.loss]
        primary_loss = primary_loss_fn(output, batch['lr'])

        # Total loss is the sum of primary loss and sum-to-one regularization term
        loss = primary_loss

        return loss

    return loss_fn


def get_step_fn(model, train,configs, optimize_fn=None):
  """Create a one-step training/evaluation function.

  Args:
    model: An model object that represents the forward SDE.
    optimize_fn: An optimization function.

  Returns:
    A one-step function for training or evaluation.
  """
  loss_fn = get_loss_fn(model, train,configs)

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model,batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model,batch)
        ema.restore(model.parameters())

    return loss

  return step_fn
