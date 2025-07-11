import argparse
import json
import numpy as np
import os
import re
import shutil

import sys

TAPATH = '/data2/spjune/silent_speech/text_alignments'
BASEPATH = '/data2/spjune/silent_speech/emg_data'

class Matcher():
    def __init__(self, dev=False, test=False, testset_file='testset_largedev.json'):
        with open(testset_file) as f:
            testset_json = json.load(f)
            devset = testset_json['dev']
            testset = testset_json['test']
        directories = []
        self.example_indices = []
        voiced_data_locations = {} # map from book/sentence_index to directory_info/index

        sd = 'silent_parallel_data'
        sd = os.path.join(BASEPATH, sd) 
        for session_dir in sorted(os.listdir(sd)):
            directories.append(EMGDirectory(len(directories), os.path.join(sd, session_dir), True))
        for vd in ['nonparallel_data', 'voiced_parallel_data']:
        #for vd in ['closed_vocab/voiced', 'nonparallel_data', 'voiced_parallel_data', 'closed_vocab/silent', 'silent_parallel_data']:
            vd = os.path.join(BASEPATH, vd)
            for session_dir in sorted(os.listdir(vd)):
                directories.append(EMGDirectory(len(directories), os.path.join(vd, session_dir), False))

        for directory_info in directories:
            for fname in os.listdir(directory_info.directory):
                m = re.match(r'(\d+)_info.json', fname)
                if m is not None:
                    idx_str = m.group(1)
                    json_path = os.path.join(directory_info.directory, fname)
                    with open(json_path) as f:
                        info = json.load(f)
                        if info['sentence_index'] >= 0 and info['text'] != '.': # boundary clips of silence are marked -1
                            if 'silent_parallel' not in json_path:
                                kind, sess, idx = split_path(json_path)
                                tg_path = f'{TAPATH}/{sess}/{sess}_{idx}_audio.TextGrid'
                                if not os.path.exists(tg_path):
                                    print(tg_path, info['text'])
                                    continue
                            location_in_testset = [info['book'], info['sentence_index']] in testset
                            location_in_devset = [info['book'], info['sentence_index']] in devset
                            if (test and location_in_testset and not directory_info.exclude_from_testset) \
                                    or (dev and location_in_devset and not directory_info.exclude_from_testset) \
                                    or (not test and not dev and not location_in_testset and not location_in_devset):
                                self.example_indices.append((directory_info,int(idx_str)))
                            if not directory_info.silent:
                                location = (info['book'], info['sentence_index'])
                                voiced_data_locations[location] = (directory_info,int(idx_str))
        self.example_indices.sort()
        self.voiced_data_locations = voiced_data_locations
        self.base_dir = BASEPATH

    def get_len(self):
        return len(self.example_indices)

    def match(self, kind, session, idx):
        example_dir = os.path.join(self.base_dir, kind, session)
        with open(os.path.join(example_dir, f'{idx}_info.json')) as f:
            info = json.load(f)
        loc = (info['book'], info['sentence_index'])
        if loc not in list(self.voiced_data_locations.keys()):
            print(f"There isn't {loc} in voiced_data_location")
            print(kind, session, idx)
            return None, None, None
        dir_voiced, idx_voiced = self.voiced_data_locations[loc]
        dir_voiced = dir_voiced.directory.split('/')
        session_voiced = dir_voiced[-1]
        kind_voiced = dir_voiced[-2]
        return kind_voiced, session_voiced, idx_voiced

def split_path(path):
    path_list = path.split('/')
    idx = path_list[-1].split('_')[0]
    sess = path_list[-2]
    kind = path_list[-3]
    return kind, sess, idx

class EMGDirectory(object):
    def __init__(self, session_index, directory, silent, exclude_from_testset=False):
        self.session_index = session_index
        self.directory = directory
        self.silent = silent
        self.exclude_from_testset = exclude_from_testset

    def __lt__(self, other):
        return self.session_index < other.session_index

    def __repr__(self):
        return self.directory

def move_file(old_path, old_idx, new_path, new_idx):
    file_list = ['audio_clean.flac', 'emg.npy', 'info.json']
    for file_end_name in file_list:
        old_file = os.path.join(old_path, f'{old_idx}_{file_end_name}')
        new_file = os.path.join(new_path, f'{new_idx}_{file_end_name}')
        os.makedirs(new_path, exist_ok=True)
        shutil.copyfile(old_file, new_file)

def move_text_grid(sess, old_idx, new_path, new_idx):
    old_file = f'{TAPATH}/{sess}/{sess}_{old_idx}_audio.TextGrid'
    new_file = os.path.join(new_path, f'{new_idx}_tg.TextGrid')
    shutil.copyfile(old_file, new_file)

if __name__ =='__main__':

    ks = {}

    for ds in ['train', 'dev', 'test']:
        dev = ds == 'dev'
        test = ds == 'test'
        matcher = Matcher(dev=dev, test=test)
        print(ds, ' dataset: ', matcher.get_len())
        for path, index in matcher.example_indices:
            path = path.directory
            new_path = path.replace(f'emg_data', f'silent_speech_dataset/{ds}')
            if 'silent_parallel' in path:
                source = os.path.join(path, f'{index}_audio_clean.flac')
                kind, sess, idx = split_path(source)
                key = f'{ds}/{kind}/{sess}'
                if key not in ks:
                    ks[key] = 0
                kind_v, sess_v, idx_v = matcher.match(kind, sess, idx)

                new_path = new_path.replace(sess, sess_v)
                move_file(path, idx, new_path, ks[key])

                path_v = path.replace(kind, kind_v)
                path_v = path_v.replace(sess, sess_v)
                new_path = path_v.replace(f'emg_data', f'silent_speech_dataset/{ds}')
                move_file(path_v, idx_v, new_path, ks[key])
                move_text_grid(sess_v, idx_v, new_path, ks[key])

                ks[key] += 1

            elif 'nonparallel' in path:
                source = os.path.join(path, f'{index}_audio_clean.flac')
                kind, sess, idx = split_path(source)
                key = f'{ds}/{kind}/{sess}'
                if key not in ks:
                    ks[key] = 0
                move_file(path, idx, new_path, ks[key])
                move_text_grid(sess, idx, new_path, ks[key])

                ks[key] += 1



    for key in ks:
        print(key, ks[key])

