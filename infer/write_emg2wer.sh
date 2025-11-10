#!/bin/bash

gpu=0
exp_name=$1
ckpt_epoch=$2
split=$3

cd ../preprocess
CUDA_VISIBLE_DEVICES=$gpu python emg2feat.py exp_name=$exp_name ckpt_epoch=$ckpt_epoch split=$split

cd ../infer
CUDA_VISIBLE_DEVICES=$gpu python save_output_emg2mel.py encoder=$exp_name data_split=$split

# ASR 실행 결과를 변수에 저장
output=$(CUDA_VISIBLE_DEVICES=$gpu python asr.py wav_dir=$exp_name/direct/$split/silent_parallel_data data_split=$split)

# 마지막 줄에서 CER / CER_without_space / WER 추출
cer=$(echo "$output" | grep -oP "CER:\s*\K[0-9.]+")
cer_wo=$(echo "$output" | grep -oP "CER without space:\s*\K[0-9.]+")
wer=$(echo "$output" | grep -oP "WER:\s*\K[0-9.]+")

# CSV 저장 경로
csv_path="../results.csv"

# CSV 헤더가 없으면 생성
if [ ! -f "$csv_path" ]; then
    echo "exp_name,ckpt_epoch,split,cer,cer_wo_space,wer" > $csv_path
fi

# CSV에 결과 추가
echo "$exp_name,$ckpt_epoch,$split,$cer,$cer_wo,$wer" >> $csv_path

echo "Saved to $csv_path"
echo "$exp_name, $ckpt_epoch, $split → CER=$cer, CERwo=$cer_wo, WER=$wer"
