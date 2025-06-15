import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  # config.training.batch_size = 64
  # config.training.batch_size = 2  # seriously?
  config.training.batch_size = 1 # When using single GPU
  # training.n_iters = 2400001
  training.epochs = 1000
  training.snapshot_freq = 50000
  # training.log_freq = 50
  training.log_freq = 25
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False
  training.save_every=10
  training.sample_every=20

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.075

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 50
  evaluate.end_ckpt = 96
  # evaluate.batch_size = 512
  evaluate.batch_size = 8
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'LSUN'
  data.image_size1 = 720
  data.image_size2 = 512  
  data.random_flip = True
  data.uniform_dequantization = False
  data.centered = False
  # data.num_channels = 3
  data.num_channels = 1
  data.out_shape = [1,1,720,512]
  
  #conditioning
  config.conditioning = conditioning = ml_collections.ConfigDict()
  conditioning.method = 'dps'
  conditioning.params = ml_collections.ConfigDict()
  conditioning.params.scale = 0.28
  
  #measurement 
  config.measurement = measurement = ml_collections.ConfigDict()
  measurement.operator = operator = ml_collections.ConfigDict()
  measurement.operator.name='Kernel'
  
  #measurement.operator.scale_factor=2
  #measurement.operator.in_shape=[1,1,720,512]
  measurement.operator.ckt_filename='/home/cidar/Desktop/MRI_superres_registration/Kernel_Learning/checkpoints/kernel_weight.pth'
  measurement.noise = noise = ml_collections.ConfigDict()
  measurement.noise.name='gaussian'
  measurement.noise.sigma=-0.05

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 378
  model.sigma_min = 0.01
  model.num_scales = 2000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device='cuda:0'
  #config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config