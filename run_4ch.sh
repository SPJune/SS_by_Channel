#!/bin/bash
set -e
channels=(0 1 2 3 4 5 6 7)
skip_until="0123"
skip=true

for a in "${channels[@]}"; do
  for b in "${channels[@]}"; do
    if (( b <= a )); then continue; fi
    for c in "${channels[@]}"; do
      if (( c <= b )); then continue; fi
      for d in "${channels[@]}"; do
        if (( d <= c )); then continue; fi

        combo="${a}${b}${c}${d}"
        combo_list="[$a,$b,$c,$d]"
        exp_name=250530mspec22_ch$combo
        if $skip; then
          if [[ "$combo" == "$skip_until" ]]; then
            skip=false  
          else
            continue  
          fi
        fi

        echo "Running combination: $combo_list"
        CUDA_VISIBLE_DEVICES=1 python main_4ch.py \
            exp_name=$exp_name \
            feature=mspec22 \
            emg_enc.use_channel="$combo_list"
        cd preprocess
        python emg2feat.py exp_name=$exp_name ckpt_epoch=last split=dev
        cd ../infer
        python save_output_emg2mel.py encoder=$exp_name
        python asr.py wav_dir=$exp_name/direct/dev/silent_parallel_data
        cd ..
        line=$(tail -n 1 /data1/marg/spjune/silent_speech/${exp_name}/direct/dev/silent_parallel_data/asr.log)
  
		cer=$(echo $line | grep -oP 'CER: \K[0-9.]+')
		cer_without_space=$(echo $line | grep -oP 'CER without space: \K[0-9.]+')
	    wer=$(echo $line | grep -oP 'WER: \K[0-9.]+')

        echo "$combo, $cer, $cer_without_space, $wer" >> /data1/marg/spjune/silent_speech/temp.csv
        rm /data1/marg/spjune/silent_speech/$exp_name/direct/dev/silent_parallel_data/*.wav
        rm -r /data2/spjune/silent_speech/preprocessed/est_feature/mspec/sr22050/$exp_name
      done
    done
  done
done
