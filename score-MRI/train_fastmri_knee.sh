#!/bin/bash

export CUDA_VISIBLE_DEVICES='0,1,2,3'
python main_fastmri.py \
 --config=/home/arpanp/Downloads/score-MRI/configs/ve/fastmri_knee_320_ncsnpp_continuous.py \
 --eval_folder=eval/fastmri_multicoil_knee_720 \
 --mode='train'  \
 --workdir=/home/arpanp/Downloads/score-MRI/work_dir_gan_shape