#!/bin/bash

export CUDA_VISIBLE_DEVICES='0,1,2,3'
python main_fastmri.py \
 --config=/home/arpanp/SRMRI/SRMRI/configs/ve/srmri_720_ncsnpp_continuous.py \
 --eval_folder=eval/fastmri_multicoil_knee_720 \
 --mode='train'  \
 --workdir=/home/arpanp/SRMRI/workdir