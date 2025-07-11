
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

sys.path.append('/workspace/silent_speech/so-vits-svc-main/')
from infer_tool import SvcAudioDecoder

def interpolate_array(arr, target_length):
    original_length = len(arr)
    
    if original_length == target_length:
        return arr.copy()
    
    x_original = np.linspace(0, 1, original_length)
    x_target = np.linspace(0, 1, target_length)
    interpolated = np.interp(x_target, x_original, arr)
    return interpolated

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg:DictConfig):
    device = 'cuda'
    model_path = '/data2/jwlee/jw_results/sovits/emg_single_sovits/onehot_phoneme_logs/16k/G_254000.pth'
    config_path = '/workspace/silent_speech/so-vits-svc-main/configs/config_phoneme.json'
    decoder = SvcAudioDecoder(model_path, config_path, device)
    sub_option = f'{cfg.feature.sub_option}{cfg["feature"][cfg.feature.sub_option]}'
    data_path = f'/data2/spjune/silent_speech/preprocessed/target_feature/hubert_soft/layer9/{cfg.data_split}/voiced_parallel_data'
    #output_dir = os.path.join(cfg.exp_path, 'toplines/hubert_soft_bottleneck', cfg.data_split, cfg.data_type)
    output_dir = os.path.join(cfg.exp_path, 'toplines/cat_phoneme', cfg.data_split, cfg.data_type)
    tg_dir = f'/data2/spjune/silent_speech/silent_speech_dataset/{cfg.data_split}/voiced_parallel_data'
    os.makedirs(output_dir, exist_ok=True)
    path_list = glob(os.path.join(cfg.data_path, f'preprocessed/target_feature/{cfg.feature.target}/{sub_option}/{cfg.data_split}/{cfg.data_type}/*/*feat.npy'))
    path_list.sort()
    for i, path in enumerate(tqdm(path_list)):
        sess, idx = extract_filename(path)
        feat = np.load(path)
        uv_path = os.path.join(data_path, sess, f'{idx}_uv.npy')
        f0_path = os.path.join(data_path, sess, f'{idx}_f0.npy')
        tg_path = os.path.join(tg_dir, sess, f'{idx}_tg.TextGrid')
        f0 = np.load(f0_path)
        uv = np.load(uv_path)
        feat = feat[:len(f0)]
        f0 = f0[:len(feat)]
        uv = uv[:len(feat)]
        feat_tensor = torch.from_numpy(feat.T).unsqueeze(0).to(device)
        uv_tensor = torch.from_numpy(uv).unsqueeze(0).unsqueeze(0).to(device)
        f0_tensor = torch.from_numpy(f0).unsqueeze(0).unsqueeze(0).to(device).to(torch.float32)
        with torch.no_grad():
            audio, mel = decoder(tg_path, feat_tensor, f0_tensor, 'gaddy', uv_tensor, False, 0.4, device)
            audio = audio.squeeze().cpu().numpy()
        m = np.max(np.abs(audio))
        audio32 = (audio/m).astype(np.float32)
        file_path = os.path.join(output_dir, f'{sess}_{idx}.wav')
        write(file_path, 16000, audio32)


if __name__ == '__main__':
    main()
