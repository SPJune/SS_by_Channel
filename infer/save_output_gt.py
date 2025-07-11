import librosa
import numpy as np
from glob import glob
import os
from scipy.io.wavfile import write
import sys
from omegaconf import DictConfig, OmegaConf
import hydra

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

def extract_filename(path):
    sess = path.split('/')[-2]
    filename = path.split('/')[-1]
    idx = filename.split('_')[0]
    return sess, idx

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg:DictConfig):
    sr22 = 22050
    output_dir = os.path.join(cfg.exp_path, 'toplines', 'gt')
    path_list = glob(os.path.join(cfg.data_path, 'silent_speech_dataset/dev/voiced_parallel_data/*/*.flac'))
    path_list.sort()
    os.makedirs(output_dir, exist_ok=True)
    for i, path in enumerate(path_list):
        sess, idx = extract_filename(path)
        audio, sr = librosa.load(path, sr=None)
        if sr != sr22:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sr22)
        m = np.max(np.abs(audio))
        audio32 = (audio/m).astype(np.float32)
        file_path = os.path.join(output_dir, f'{sess}_{idx}.wav')
        write(file_path, sr22, audio32)


if __name__ == '__main__':
    main()
