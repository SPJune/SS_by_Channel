#!/bin/bash

gpu=0
cd ../preprocess
CUDA_VISIBLE_DEVICES=$gpu python emg2feat.py exp_name=$1 ckpt_epoch=$2 split=$3
cd ../infer
CUDA_VISIBLE_DEVICES=$gpu python save_output_emg2mel.py encoder=$1 data_split=$3
CUDA_VISIBLE_DEVICES=$gpu python asr.py wav_dir=$1/direct/$3/silent_parallel_data data_split=$3
