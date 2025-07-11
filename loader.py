import re
import os
import numpy as np
import random
import scipy
import json
import copy
import sys
import pickle
import string
import logging
from functools import lru_cache
from copy import copy
from glob import glob
from torch.nn.utils.rnn import pad_sequence
import torch

from data_utils import get_emg_features, FeatureNormalizer, phoneme_inventory, read_phonemes, TextTransform

def remove_drift(signal, fs): # for emg signal
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,8):
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result

def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
    return np.stack(results, 1)


class EMGDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, target, data_split, frame_rate, target_sec=None, normalize=False, seq_len=None):
        self.load_zero_output(base_dir)
        if normalize:
            npz_data = np.load(os.path.join(base_dir, 'normalizer.npz'))
            self.normalizer = FeatureNormalizer(npz_data['mean'], npz_data['std'])
            self.feat_norm, self.emg_norm = pickle.load(open(os.path.join(base_dir, 'normalizer.pkl'),'rb'))
        self.data_split = data_split
        self.preprocessed_path = os.path.join(base_dir, data_split)
        base_dir = base_dir[:base_dir.find('preprocessed')].rstrip('/')
        self.base_path = os.path.join(base_dir, 'silent_speech_dataset', data_split)
        if data_split == 'train':
            self.data_path = glob(os.path.join(self.base_path, '*/*/*.flac'))
        else:
            self.data_path = glob(os.path.join(self.base_path, 'silent_parallel_data/*/*.flac'))

        self.frame_rate = frame_rate
        self.conv_ds = round(100/frame_rate)*8 # 8 for 66.7~200, 16 for 40~66.6
        self.text_transform = TextTransform()
        self.target_sec = target_sec
        self.normalize = normalize
        self.seq_len = seq_len
        if self.target_sec != None:
            self.target_speech_len = int(self.frame_rate*self.target_sec)
            self.target_emg_len = self.target_speech_len * self.conv_ds

        mel_dir = os.path.join(self.preprocessed_path.split(target)[0], 'mspec/sr16000')
        self.mel_norm, _ = pickle.load(open(os.path.join(mel_dir, 'normalizer.pkl'),'rb'))
        self.mel_dir = os.path.join(mel_dir, data_split)
        self.target = target

        random.seed(0)
        random.shuffle(self.data_path)

    def get_adjacent_paths(self, path):
        base_dir, file_name = os.path.split(path)
        index = int(file_name.split('_')[0])
        before = os.path.join(base_dir, f"{index - 1}_emg.npy")
        after = os.path.join(base_dir, f"{index + 1}_emg.npy")
        return before, after

    def load_emg(self, path, max_len=None):
        resample_rate = self.frame_rate*self.conv_ds
        raw_emg = np.load(path)
        before, after = self.get_adjacent_paths(path)
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
        if max_len != None:
            raw_emg = raw_emg[8:8+max_len,:]
        else:
            raw_emg = raw_emg[8:,:]

        return raw_emg

    def load_textgrid(self, path, max_len=None):
        if os.path.exists(path):
            phonemes = read_phonemes(path, max_len, fr=self.frame_rate)
        else:
            print(path)
            phonemes = np.zeros(max_len, dtype=np.int64)+phoneme_inventory.index('sil')
        return phonemes
    
    def load_text(self, path):
        with open(path) as f:
            info = json.load(f)
        return info['text']

    def load_speech_feature(self, path, max_len=None):
        feat = np.load(path)
        feat = feat.astype(np.float32)
        if max_len is not None and feat.shape[0] > max_len:
            feat = feat[:max_len, :]
        if self.target == 'hubert_soft':
            f0_path = path.replace('feat.npy', 'f0.npy')
            uv_path = path.replace('feat.npy', 'uv.npy')
            f0 = np.load(f0_path).astype(np.float32)
            uv = np.load(uv_path)
        else:
            f0, uv = np.array([120]*feat.shape[0]), np.array([1]*feat.shape[0])
            #f0, uv = None, None
        return feat, f0, uv # trim for emg's short length

    def __len__(self):
        return len(self.data_path)

    def load_zero_output(self, base_dir):
        path = os.path.join(base_dir, 'zero_output.npy')
        self.zero_output = np.load(path)

    def extract_path(self, path):
        base_dir, file_name = os.path.split(path)
        idx = file_name.split('_')[0]
        parts = base_dir.split('/')
        idx_split = parts.index(self.data_split)
        data_type = parts[idx_split + 1]
        sess = parts[idx_split + 2]

        silent = 'silent_parallel_data' in data_type

        emg_path = os.path.join(self.base_path, data_type, sess, f"{idx}_emg.npy")
        if silent:
            feat_path = os.path.join(self.preprocessed_path, "voiced_parallel_data", sess, f"{idx}_feat.npy")
            tg_path = os.path.join(self.base_path, "voiced_parallel_data", sess, f'{idx}_tg.TextGrid')
            mel_path = os.path.join(self.mel_dir, 'voiced_parallel_data', sess, f'{idx}_feat.npy')
            #voiced_emg_path = os.path.join(self.base_path, "voiced_parallel_data", sess, f"{idx}_emg.npy")
        else:
            feat_path = os.path.join(self.preprocessed_path, data_type, sess, f'{idx}_feat.npy')
            tg_path = os.path.join(self.base_path, data_type, sess, f'{idx}_tg.TextGrid')
            mel_path = os.path.join(self.mel_dir, data_type, sess, f'{idx}_feat.npy')
        json_path = os.path.join(self.base_path, data_type, sess, f'{idx}_info.json')
        return mel_path, emg_path, feat_path, json_path, tg_path, silent

    @lru_cache(maxsize=None)
    def __getitem__(self, i):
        mel_path, emg_path, feat_path, json_path, tg_path, silent = self.extract_path(self.data_path[i])
        speech_feature, f0, uv = self.load_speech_feature(feat_path)
        speech_max_len = speech_feature.shape[0]
        emg_max_len = None if silent else speech_max_len*self.conv_ds
        emg = self.load_emg(emg_path, max_len=emg_max_len)
        mel = np.load(mel_path)
        mel = mel.astype(np.float32)
        mel = self.mel_norm.normalize(mel)
        if not silent and speech_max_len > len(emg)//self.conv_ds:
            speech_feature = speech_feature[:len(emg)//self.conv_ds]
            f0 = f0[:len(emg)//self.conv_ds]
            uv = uv[:len(emg)//self.conv_ds]
            mel = mel[:len(emg)//self.conv_ds]
        #emg = self.load_emg(emg_path)
        phonemes = self.load_textgrid(tg_path, max_len=speech_feature.shape[0])

        text = self.load_text(json_path)
        text_int = np.array(self.text_transform.text_to_int(text), dtype=np.int64)

        if self.target_sec != None:
            speech_feature, emg, phonemes, start, crop_len = self.crop(speech_feature, emg, phonemes)
            mel = mel[start:start + crop_len]
            f0 = f0[start:start + crop_len]
            uv = uv[start:start + crop_len]
            #speech_feature, emg, phonemes = self.crop_or_pad(speech_feature, emg, phonemes)

        if self.normalize:
            speech_feature = self.feat_norm.normalize(speech_feature)
            #speech_feature = self.normalizer.normalize(speech_feature)

        speech_feature = torch.from_numpy(speech_feature)
        emg = torch.from_numpy(emg)
        text_int = torch.from_numpy(text_int)
        phonemes = torch.from_numpy(phonemes)
        audio_path = self.data_path[i]
        speech_feature = speech_feature.type(torch.float32)
        emg = emg.type(torch.float32)

        mel = torch.from_numpy(mel)
        f0 = torch.from_numpy(f0).unsqueeze(0)
        uv = torch.from_numpy(uv).unsqueeze(0)

        return speech_feature, emg, phonemes, silent, audio_path, mel, f0, uv

    def crop(self, speech_feature, emg, phonemes):
        speech_len = len(speech_feature)
        emg_len = len(emg)

        ratio = speech_len / emg_len
        diff = speech_len - self.target_speech_len
        speech_start = 0 if diff <= 0 else random.randint(0, diff)

        emg_start = int(speech_start / ratio)

        max_speech_crop = min(speech_len - speech_start, self.target_speech_len)
        max_emg_crop = min(emg_len - emg_start, self.target_emg_len)

        speech_crop_len = min(max_speech_crop, int(max_emg_crop * ratio))
        emg_crop_len = min(max_emg_crop, int(speech_crop_len / ratio))

        cropped_speech = speech_feature[speech_start:speech_start + speech_crop_len]
        cropped_phonemes = phonemes[speech_start:speech_start + speech_crop_len]
        cropped_emg = emg[emg_start:emg_start + emg_crop_len]

        return cropped_speech, cropped_emg, cropped_phonemes, speech_start, speech_crop_len

    def crop_or_pad(self, speech_feature, emg, phonemes):
        N, D = speech_feature.shape
        M, _ = emg.shape
        target_speech_len = int(self.frame_rate*self.target_sec)
        target_emg_len = target_speech_len * self.conv_ds
        ratio = M/N

        padded_speech = np.zeros((target_speech_len, D))
        padded_emg = np.zeros((target_emg_len, 8))
        padded_phonemes = np.full((target_speech_len,), phoneme_inventory.index('sil'))

        if N >= target_speech_len:
            start_idx = random.randint(0, N - target_speech_len)
            end_idx = start_idx + target_speech_len
            padded_speech = speech_feature[start_idx:end_idx]
            padded_phonemes = phonemes[start_idx:end_idx]
        else:
            padded_speech[:N] = speech_feature
            padded_speech[N:] = self.zero_output

            padded_phonemes[:N] = phonemes
            padded_phonemes[N:] = phoneme_inventory.index('sil')

        emg_start_idx = int(start_idx*ratio) if N >= target_speech_len else 0

        if emg_start_idx + target_emg_len <= M:
            padded_emg = emg[emg_start_idx:emg_start_idx + target_emg_len]
        else:
            remaining_length = M - emg_start_idx
            if len(emg[emg_start_idx:M]) == 0 or len(padded_emg[:remaining_length]) == 0:
                print(start_idx, self.conv_ds, emg_start_idx, M, remaining_length, N, D)
                sys.exit()
            padded_emg[:remaining_length] = emg[emg_start_idx:M]

        return padded_speech, padded_emg, padded_phonemes

    def collate_raw(self, batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        speech_features = [item[0] for item in batch]  # N x D
        emg_signals = [item[1] for item in batch]      # M x C
        phonemes = [item[2] for item in batch]         # N x P
        silents = [item[3] for item in batch]          # bool
        audio_paths = [item[4] for item in batch]      # string
        mel = [item[5] for item in batch]      # N x 80
        f0 = [item[6] for item in batch]      # 1 x N
        uv = [item[7] for item in batch]      # 1 x N

        target_lengths = [len(sf) for sf in speech_features]
        est_lengths = [len(emg)//self.conv_ds for emg in emg_signals]

        max_length = max(target_lengths)
        max_emg_len = max(est_lengths)*self.conv_ds

        padded_speech_features = torch.stack([torch.cat([sf[:max_length], torch.zeros(max_length - len(sf[:max_length]), sf.shape[1])], dim=0) for sf in speech_features])
        padded_mel = torch.stack([torch.cat([sf[:max_length], torch.zeros(max_length - len(sf[:max_length]), sf.shape[1])], dim=0) for sf in mel])
        padded_f0 = torch.stack([torch.cat([f0_[:,:max_length], torch.zeros(1, max_length - len(f0_[0,:max_length]))], dim=1) for f0_ in f0])
        padded_uv = torch.stack([torch.cat([uv_[:,:max_length], torch.zeros(1, max_length - len(uv_[0,:max_length]))], dim=1) for uv_ in uv])
        padded_emg_signals = torch.stack([torch.cat([emg[:max_emg_len], torch.zeros(max_emg_len - len(emg[:max_emg_len]), emg.shape[1])], dim=0) for emg in emg_signals])
        padded_phonemes = torch.stack([torch.cat([ph[:max_length], torch.zeros(max_length - len(ph[:max_length]), dtype=torch.int64)], dim=0) for ph in phonemes])

        target_lengths = torch.tensor(target_lengths)
        est_lengths = torch.tensor(est_lengths)

        batch = {
            'speech_features': padded_speech_features,
            'emg': padded_emg_signals,
            'phonemes': padded_phonemes,
            'target_lengths': target_lengths,
            'est_lengths': est_lengths,
            'silents': torch.tensor(silents),
            'audio_paths': audio_paths,
            'mel': padded_mel,
            'f0': padded_f0,
            'uv': padded_uv
        }
        return batch

    def collate_fixed_length(self, batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        speech_features = [item[0] for item in batch]  # N x D
        emg_signals = [item[1] for item in batch]      # M x C
        phonemes = [item[2] for item in batch]         # N x P
        silents = [item[3] for item in batch]          # bool
        audio_paths = [item[4] for item in batch]      # string

        lengths = [min(len(sf), len(emg)//self.conv_ds) for sf, emg in zip(speech_features, emg_signals)]
        total_length = sum(lengths)
        if total_length % self.seq_len != 0:
            pad_length = self.seq_len - (total_length % self.seq_len)
            emg_signals = list(emg_signals)

        max_length = max(lengths)
        max_emg_len = int(max_length*self.conv_ds)

        padded_speech_features = torch.stack([torch.cat([sf[:max_length], torch.zeros(max_length - len(sf[:max_length]), sf.shape[1])], dim=0) for sf in speech_features])
        padded_emg_signals = torch.stack([torch.cat([emg[:max_emg_len], torch.zeros(max_emg_len - len(emg[:max_emg_len]), emg.shape[1])], dim=0) for emg in emg_signals])
        padded_phonemes = torch.stack([torch.cat([ph[:max_length], torch.zeros(max_length - len(ph[:max_length]), dtype=torch.int64)], dim=0) for ph in phonemes])

        lengths = torch.tensor(lengths)

        batch = {
            'speech_features': padded_speech_features,
            'emg': padded_emg_signals,
            'phonemes': padded_phonemes,
            'lengths': lengths,
            'silents': torch.tensor(silents),
            'audio_paths': audio_paths
            }
        return batch


def make_normalizers():
    dataset = EMGDataset(no_normalizers=True)
    mfcc_samples = []
    emg_samples = []
    for d in dataset:
        mfcc_samples.append(d['speech_features'])
        emg_samples.append(d['emg'])
        if len(emg_samples) > 50:
            break
    mfcc_norm = FeatureNormalizer(mfcc_samples, share_scale=True)
    emg_norm = FeatureNormalizer(emg_samples, share_scale=False)
    pickle.dump((mfcc_norm, emg_norm), open(FLAGS.normalizers_file, 'wb'))

if __name__ == '__main__':
    frame_rate = 100
    d = EMGDataset('/data2/spjune/silent_speech/preprocessed/target_feature/hubert_soft/layer9', 'hubert_soft', 'train', frame_rate, 20)
    #d = EMGDataset('/data2/spjune/silent_speech/preprocessed/target_feature/hubert/layer6', 'hubert', 'train', 50, 20)
    for i in range(5):
        speech_feature, emg, phonemes, silent, audio_path, mel, f0, uv = d[i]
        len_emg = len(emg)
        len_feat = len(speech_feature)
        t_emg = len_emg/800
        t_feat = len_feat/frame_rate
        print(len_emg, len_feat, t_emg, t_feat)
        print(mel.shape, speech_feature.shape, f0.shape, uv.shape, speech_feature.dtype, emg.dtype, phonemes.dtype)
