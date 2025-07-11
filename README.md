## **📌 Overview**
This repository provides an end-to-end framework for silent speech decoding using surface electromyography (sEMG) signals. Building on the Gaddy et al. dataset, our system maps sEMG inputs to acoustic features, which are then used to synthesize speech and evaluate intelligibility through automatic speech recognition (ASR).

Key features include:
- Preprocessing of the Gaddy dataset with re-indexing and alignment
- Flexible training pipelines supporting various EMG channel combinations
- Support for channel dropout and fine-tuning from pretrained models
- Inference scripts for speech synthesis and WER computation using Whisper ASR
- Phoneme-level error analysis for detailed evaluation
 
## ⚙️ Setup
TBA

## **🚀 Usage**

### **1. Data Preparation**

1. Download the Gaddy dataset from [Zenodo](https://doi.org/10.5281/zenodo.4064408) and extract it.
    
2. Prepare text alignments from the original [repository](https://github.com/dgaddy/silent_speech):
    
```
git submodule init
git submodule update
tar -xvzf text_alignments/text_alignments.tar.gz
```

3. In preprocess/reassign.py, set:
    
    - BASEPATH to the extracted Gaddy dataset directory
        
    - TAPATH to the extracted text alignment directory
        
    
4. Run the following to clean and reindex the data:
    

```
cd preprocess
python reassign.py
```

This will remove empty/short utterances, reindex all examples from 0, and align indices across parallel_voiced_data and parallel_silent_data. The processed dataset will be saved under silent_speech_dataset.

  
### **2. Update Configuration**

Edit configs/common.yaml to:

- Set data_path to the newly created silent_speech_dataset
    
- Set exp_path to your desired experiment log/output directory
    

  

### **3. Preprocessing**

- Generate target features from raw waveforms:
    

```
python speech2feat.py  # Default: 22kHz mel spectrogram (mspec22)
```

To generate 16kHz mel spectrograms instead:

```
python speech2feat.py feature=mspec16
```

- Extract features from EMG using a trained model:
    

```
python emg2feat.py exp_name={exp_name} ckpt_epoch={ckpt_epoch} split=dev
```

This saves the estimated features under preprocessed/est_feature/{exp_name}.

  

### **4. Training**

- Basic training:

```
python main.py exp_name={exp_name} feature=mspec22 emg_enc.use_channel="[0,2,4,5]"
```

> Note: Channels are 0-indexed. For channels 1,3,5,6 in the original dataset, use 0,2,4,5 here.

  

- Train with channel dropout:
    

```
python main.py exp_name={exp_name} emg_enc.channel_dropout=0.125
```

- Fine-tuning from a pretrained model:
    

```
python main.py exp_name={exp_name} \
  pretrained_model={pretrained_exp_name} \
  pretrained_epoch={pretrained_ckpt_epoch} \
  emg_enc.use_channel="[0,2,4,5]"
```

- Adjust maximum epochs:
    

```
python main.py exp_name={exp_name} trainer.max_epochs=150
```

- Run all 70 4-channel combinations:
    

```
./run_4ch.sh
```

> You can resume from a specific combination by editing skip_until in the script.

  

### **5. Inference**

- Evaluate EMG-to-speech pipeline (feature extraction → synthesis → WER):
    

```
cd infer
./emg2wer.sh {exp_name} {epoch} {split}  # split: train/dev/test
```

- Phoneme-level analysis:
    

```
python phoneme_analysis.py exp_name={exp_name} ckpt_epoch={ckpt_epoch} split=dev
```

This will save a CSV file reporting phoneme-wise error rates, number of errors, and total counts.
