
"""All functions and modules related to model definition.
"""
import torch
from model.Kernel import Kernel


def get_model(model_name,config):
    """Get the model class based on `model_name`."""
    if model_name == 'Kernel':
        # Kernel_size = config.model.kernel_size
        # scaling_factors = config.model.scaling_factors
        # model = Kernel(Kernel_size, scaling_factors)
        model=Kernel()
        return model
    else:
        raise NotImplementedError(f'Model {model_name} not implemented yet!')
    
def create_model(config):
  """Create the score model."""
  model_name = config.model.name
  model = get_model(model_name,config)
  model = model.to(config.device)
  return model

def get_model_fn(model, train=False):
  """Create a function to give the output of the model.
  Args:
    model: The  model.
    train: `True` for training and `False` for evaluation.
  Returns:
    A model function.
  """

  def model_fn(x):
    """Compute the output of the model.

    Args:
      x: A mini-batch of input data.

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x)
    else:
      model.train()
      return model(x)

  return model_fn