from glob import glob
import hydra
import json
import librosa
import logging
import numpy as np
import os
from omegaconf import DictConfig, OmegaConf
import random
import soundfile as sf
import sys
import torch
import torchaudio
from unidecode import unidecode

from silero_metric import calculate_error_cnt, calculate_error_rate
from silero_decoder import Decoder
from silero_utils import normalize_text
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def load_json(json_dir):
    with open(json_dir) as f:
        json_file = json.load(f)
    return json_file['text']

def save_json(write_dir, data):
    with open(write_dir, 'w') as f:
        json.dump(data, f)

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg:DictConfig):
    data_dir = os.path.join(cfg.exp_path, cfg.wav_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger_name = 'asr_silero.log' if cfg.use_silero else 'asr.log'
    file_handler = logging.FileHandler(os.path.join(data_dir, logger_name), mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    sr16 = 16000
    path_list = glob(os.path.join(data_dir, '*.wav'))
    if cfg.use_silero:
        model = torch.jit.load('/workspace/silero/models/en_v6.jit', map_location='cuda')
        model.eval()
        decoder = Decoder(model.labels)

    else:
        processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
        model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium').to('cuda')
    results = []
    for i, path in enumerate(path_list):
        audio, sr = sf.read(path)
        if sr != sr16:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sr16)
        sess, idx = path.split('/')[-1].split('_')
        idx = idx.split('.')[0]
        json_path = os.path.join(cfg.data_path, f'silent_speech_dataset/{cfg.data_split}/voiced_parallel_data/{sess}/{idx}_info.json')
        info_text = load_json(json_path)
        if cfg.use_silero:
            target_text = unidecode(info_text)
            target_text = normalize_text(target_text)
            input = torch.Tensor(audio).unsqueeze(0).to('cuda')
            output = model(input)
            prediction = decoder(output[0].cpu())
        else:
            target_text = processor.tokenizer._normalize(info_text)
            input_features = processor(audio, sampling_rate=sr16, return_tensors='pt').input_features
            with torch.no_grad():
                predicted_ids = model.generate(input_features.to('cuda'))[0]

            transcription = processor.decode(predicted_ids)
            prediction = processor.tokenizer._normalize(transcription)

        result = calculate_error_cnt(prediction, target_text)
        wer0 = result[-2]/result[-1]
        cer0 = result[0]/result[1]
        results.append(result)

        logger.info(f'trgt{i} {sess} {idx}: {target_text}')
        logger.info(f'pred{i} {sess} {idx}: {prediction}')
        logger.info(f'{i} sample cer: {cer0*100:.2f}, wer: {wer0*100:.2f}')
    cer, cer_wo_space, wer = calculate_error_rate(results)
    result_str = f"CER: {cer*100:.2f}%, CER without space: {cer_wo_space*100:.2f}%, WER: {wer*100:.2f}%"
    logger.info(result_str)

if __name__ == '__main__':
    main()
