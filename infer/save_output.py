from glob import glob
import hydra
import numpy as np
import os
from omegaconf import DictConfig, OmegaConf
from scipy.io.wavfile import write
import sys
import torch
from tqdm import tqdm

from save_output_gt import extract_filename

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from modules import Vocoder
from utils import load_tacotron, load_unit, merge

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg:DictConfig):
    device = 'cuda'
    sr22 = 22050
    if cfg.decoder_iter == None: # fairseq decoder
        decoder_name = f'{cfg.feature.target}{cfg.num_clusters}'
        decoder_ckpt = f'{decoder_name}.pt'
    else:
        decoder_ckpt = f'checkpoint_{cfg.decoder_iter}'
        decoder_name = f'{cfg.decoder}_{cfg.decoder_iter}'
    decoder_path = os.path.join(cfg.exp_path, f'unit2mel/{cfg.decoder}', decoder_ckpt)
    decoder, sr, hparams = load_tacotron(decoder_path, cfg.max_step)
    vocoder_path = '/data1/marg/spjune/silent_speech/pretrained_models/hifigan_t2/generator_v1'
    vocoder = Vocoder(vocoder_path, device=device, half=hparams.fp16_run)

    sub_option = f'{cfg.feature.sub_option}{cfg["feature"][cfg.feature.sub_option]}'
    output_dir = os.path.join(cfg.exp_path, cfg.encoder, cfg.km_name, decoder_name, cfg.data_split, cfg.data_type)
    os.makedirs(output_dir, exist_ok=True)
    path_list = glob(os.path.join(cfg.data_path, f'preprocessed/est_unit/{cfg.feature.target}/{sub_option}/{cfg.encoder}/{cfg.km_name}/{cfg.data_split}/{cfg.data_type}/*/*.txt'))
    path_list.sort()
    for i, path in enumerate(tqdm(path_list)):
        sess, idx = extract_filename(path)
        unit = load_unit(path, eos=cfg.eos, sos=cfg.sos)
        if cfg.merge:
            unit = merge(unit, output='torch')
        unit = unit.unsqueeze(0).to(device)
        with torch.no_grad():
            _, mel, _, ali = decoder.inference(unit)
            audio = vocoder(mel[0].T).cpu().numpy()
        m = np.max(np.abs(audio))
        audio32 = (audio/m).astype(np.float32)
        file_path = os.path.join(output_dir, f'{sess}_{idx}.wav')
        write(file_path, sr22, audio32)


if __name__ == '__main__':
    main()
