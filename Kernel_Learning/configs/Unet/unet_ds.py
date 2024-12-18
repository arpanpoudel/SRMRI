
"""Training Unet to downsample MRI images."""

from configs.default_configs import get_default_configs



def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.lambda_sum=1e-2




  # data
  data = config.data
  data.dataset = 'MRI'
  data.train = '/home/arpanp/Downloads/Data/Registration_slices/train'
  data.eval = '/home/arpanp/Downloads/Data/Registration_slices/test'
  data.image_size1 = 720
  data.image_size2 = 512


  model = config.model
  model.name = 'Kernel'
  model.kernel_size=9
  model.scaling_factors=[0.5,0.5]
  model.ema_rate = 0.999


  return config
