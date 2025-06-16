import matplotlib.pyplot as plt
import torch
from models.ema import ExponentialMovingAverage
from pathlib import Path
import controllable_generation_fast as controllable_generation
from utils import restore_checkpoint, clear_color, clear
import models
from models import utils as mutils
from models import ncsnpp
import sampling
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)
import dataset
from dataset import ResizeAndPad
from skimage.transform import resize
from torchvision import transforms
from utils import normalize_np
from utils import get_logger
from models.condition_methods import get_condition_method
from models.measurements import get_operator, get_noise
import importlib
import numpy as np
import os
import sys
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="MRI Super-Resolution Script")
    parser.add_argument(
        "--root",
        type=str,
        default="/home/arpanp/SRMRI/SRMRI/samples/lr",
        help="Path to the root directory containing input LR images. Default is set to /home/cidar/Desktop/MRI_superres_registration/data/Registration_slices/test/LR"
    )
    args = parser.parse_args()

    root = args.root
    num_scales = 2000
    sde = 'VESDE'

    print('Initializing...')
    if sde.lower() == 'vesde':
        configs = importlib.import_module(f"configs.ve.srmri_720_ncsnpp_continuous")
        config = configs.get_config()
        config.model.num_scales = num_scales
        ckpt_filename = config.ckpt_filename
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5

    batch_size = 1
    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size

    # Logger
    device = config.device
    logger = get_logger()
    logger.info(f"Device set to {device}.")

    # Prepare Operator and Noise
    measure_config = config.measurement
    operator = get_operator(device=device, **measure_config.operator)
    noiser = get_noise(**measure_config.noise)
    logger.info(f"Operation: {measure_config.operator.name} / Noise: {measure_config.noise.name}")

    # Prepare Conditioning Method
    cond_config = config.conditioning
    cond_method = get_condition_method(cond_config.method, operator, noiser, **cond_config.params)
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {cond_config.method}")

    wavelet_method = get_condition_method('wavelet', operator, noiser, level=2, device=device)
    wavelet_cond_fn = wavelet_method.conditioning
    random_seed = 0

    # Score Model
    sigmas = mutils.get_sigmas(config)
    scaler = dataset.get_data_scaler(config)
    inverse_scaler = dataset.get_data_inverse_scaler(config)
    score_model = mutils.create_model(config)

    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)

    state = restore_checkpoint(ckpt_filename, state, config.device, skip_optimizer=True)
    ema.copy_to(score_model.parameters())

    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    snr = 0.16
    n_steps = 1
    probability_flow = False

    # Process Files
    files = [f.name for f in os.scandir(root)][::-1]

    for filename in files:
        save_root = Path(f'./results_srmri/{filename}')
        save_root.mkdir(parents=True, exist_ok=True)

        irl_types = ['input', 'recon', 'recon_progress']
        for t in irl_types:
            save_root_f = save_root / t
            save_root_f.mkdir(parents=True, exist_ok=True)

        # Read Data
        img = torch.from_numpy(np.load(f'{root}/{filename}')).unsqueeze(0)
        transform = transforms.Compose([])
        img = transform(img).squeeze()
        h, w = img.shape
        img = img.view(1, 1, h, w)
        img = img.to(config.device)

        plt.imsave(save_root / 'input' / f'{filename.split(".")[0]}_LR.png', clear(img), cmap='gray')

        # Inference
        pc_mri = controllable_generation.get_pc_mri(sde,
                                                    predictor, corrector,
                                                    inverse_scaler,
                                                    snr=snr,
                                                    n_steps=n_steps,
                                                    probability_flow=probability_flow,
                                                    continuous=config.training.continuous,
                                                    denoise=True,
                                                    save_progress=False,
                                                    save_root=save_root,
                                                    measurement_cond_fn=measurement_cond_fn,
                                                    fast_step=config.sampling.fast_step,
                                                    fast_cond_fn=wavelet_cond_fn,
                                                    measurement_noise=False)

        x = pc_mri(score_model, scaler(img), config.data.out_shape)

        # Save Results
        np.save(save_root / 'recon' / f'{filename.split(".")[0]}.npy', clear(x))
        plt.imsave(str(save_root / 'recon' / f'{filename.split(".")[0]}x.png'), clear(x), cmap='gray')
        plt.imsave(str(save_root / 'recon' / f'{filename.split(".")[0]}x_clip.png'), np.clip(clear(x), 0.05, 0.95), cmap='gray')

if __name__ == '__main__':
    main()
