from glob import glob
import hydra
import numpy as np
import os
from omegaconf import DictConfig, OmegaConf
from scipy.io.wavfile import write
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

from save_output_gt import extract_filename

sys.path.append('/workspace/silent_speech/so-vits-svc-main/')
from infer_tool import SvcAudioDecoder

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from modules import EMGEncoder
from loader import apply_to_all, subsample, notch_harmonics, remove_drift
from utils import align_from_distances
from data_utils import read_phonemes, phoneme_inventory

def get_adjacent_paths(path):
    base_dir, file_name = os.path.split(path)
    index = int(file_name.split('_')[0])
    before = os.path.join(base_dir, f"{index - 1}_emg.npy")
    after = os.path.join(base_dir, f"{index + 1}_emg.npy")
    return before, after

def load_emg(path, frame_rate):
    conv_ds = round(100/frame_rate)*8 # 8 for 66.7~200, 16 for 40~66.6
    resample_rate = int(frame_rate*conv_ds)
    raw_emg = np.load(path)
    before, after = get_adjacent_paths(path)
    if os.path.exists(before):
        raw_emg_before = np.load(before)
    else:
        raw_emg_before = np.zeros([0,raw_emg.shape[1]])
    if os.path.exists(after):
        raw_emg_after = np.load(after)
    else:
        raw_emg_after = np.zeros([0,raw_emg.shape[1]])

    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0]:x.shape[0]-raw_emg_after.shape[0],:]
    raw_emg = apply_to_all(subsample, x, resample_rate, 1000)

    raw_emg = raw_emg / 20
    raw_emg = 50*np.tanh(raw_emg/50.)
    raw_emg = raw_emg.astype(np.float32)

    return raw_emg

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
    if cfg.encoder_epoch == None or cfg.encoder_epoch == 'last':
        encoder_path = os.path.join(cfg.exp_path, cfg.encoder, 'last.ckpt')
    else:
        encoder_path = glob(os.path.join(cfg.exp_path, cfg.encoder, f'epoch={cfg.encoder_epoch:02d}*'))[0]
    emg_enc = EMGEncoder.load_from_checkpoint(encoder_path, strict=False).to(device)
    emg_enc.eval()
    feature = emg_enc.hparams.feature_config
    model_path = '/data2/jwlee/jw_results/sovits/emg_single_sovits/onehot_phoneme_logs/16k/G_254000.pth'
    config_path = '/workspace/silent_speech/so-vits-svc-main/configs/config_phoneme.json'
    decoder = SvcAudioDecoder(model_path, config_path, device)
    sub_option = f'{feature.sub_option}{feature[feature.sub_option]}'
    data_path = f'/data2/spjune/silent_speech/preprocessed/target_feature/hubert_soft/layer9/{cfg.data_split}/voiced_parallel_data'
    tg_dir = f'/data2/spjune/silent_speech/silent_speech_dataset/{cfg.data_split}/voiced_parallel_data'
    output_dir = os.path.join(cfg.exp_path, 'cat_phoneme', cfg.encoder, cfg.data_split, cfg.data_type)
    os.makedirs(output_dir, exist_ok=True)
    path_list = glob(os.path.join(cfg.data_path, 'silent_speech_dataset', f'{cfg.data_split}/{cfg.data_type}/*/*_emg.npy'))
    path_list.sort()
    correct_phones = 0
    total_length = 0
    for i, path in enumerate(tqdm(path_list)):
        sess, idx = extract_filename(path)
        emg = load_emg(path, feature.frame_rate)
        emg = torch.tensor(emg).unsqueeze(0).to(device)
        f0_tensor, uv_tensor = None, None
        with torch.no_grad():
            feat_tensor, ph, _, _ = emg_enc(emg, f0_tensor, uv_tensor) # 1 x N x D
            ph = ph[0]
        uv_path = os.path.join(data_path, sess, f'{idx}_uv.npy')
        f0_path = os.path.join(data_path, sess, f'{idx}_f0.npy')
        target_path = os.path.join(data_path, sess, f'{idx}_feat.npy')
        feat_target = torch.from_numpy(np.load(target_path)).to(device)
        L = len(feat_tensor[0])
        total_length += L

        ph_target_path = os.path.join(tg_dir, sess, f'{idx}_tg.TextGrid')
        ph_target = torch.LongTensor(read_phonemes(ph_target_path, len(feat_target), fr=100)).to(device)
        if cfg.use_gt_ph:
            tg_path = os.path.join(tg_dir, sess, f'{idx}_tg.TextGrid')
        else:
            tg_path = torch.zeros_like(ph)
            tg_path.scatter_(1, torch.argmax(ph, dim=1, keepdim=True), 1)
        

        if 'silent_parallel' in path:
            f0 = np.load(f0_path)
            uv = np.load(uv_path)
            f0 = interpolate_array(f0, L)
            uv = interpolate_array(uv, L)
            dists = torch.cdist(feat_tensor, feat_target.unsqueeze(0))
            dists = dists.squeeze(0)
            pred_phone = F.log_softmax(ph, -1)
            phone_lprobs = pred_phone[:,ph_target]
            costs = dists*1.0 - phone_lprobs*0.0
            alignment = align_from_distances(costs.cpu().detach().numpy())
            aligned_gt_ph = ph_target[alignment]
            ph = ph.argmax(-1)
            correct_phones += (aligned_gt_ph == ph).sum().item()
            one_hot_label = F.one_hot(aligned_gt_ph, num_classes=len(phoneme_inventory))
            '''
            alignment = align_from_distances(costs.T.cpu().detach().numpy())
            ph = ph.argmax(-1)
            aligned_ph = ph[alignment]
            correct_phones = (aligned_ph == ph_target).sum().item()
            '''
        else:
            f0 = np.load(f0_path)
            uv = np.load(uv_path)
            feat_tensor = feat_tensor[:,:len(f0)]
            f0 = f0[:len(feat_tensor[0])]
            uv = uv[:len(feat_tensor[0])]
        uv_tensor = torch.from_numpy(uv).unsqueeze(0).unsqueeze(0).to(device)
        f0_tensor = torch.from_numpy(f0).unsqueeze(0).unsqueeze(0).to(device).to(torch.float32)
        feat_tensor = feat_tensor.transpose(1,2)
        with torch.no_grad():
            audio, mel = decoder(one_hot_label, feat_tensor, f0_tensor, 'gaddy', uv_tensor, False, 0.4, device)
            audio = audio.squeeze().cpu().numpy()
        m = np.max(np.abs(audio))
        audio32 = (audio/m).astype(np.float32)
        file_path = os.path.join(output_dir, f'{sess}_{idx}.wav')
        write(file_path, 16000, audio32)
    print("Phoneme accuracy: ", correct_phones/total_length)


if __name__ == '__main__':
    main()
