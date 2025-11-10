import pytorch_lightning as pl
from glob import glob
import os
import sys
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import pickle

from loader import EMGDataset
from modules import EMGEncoder
from data_utils import phoneme_inventory
from utils import load_partial_pretrained_model, set_global_seed

class SpecificEpochsCheckpoint(ModelCheckpoint):
    def __init__(self, save_epochs=[5, 10], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_epochs = save_epochs
    
    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch in self.save_epochs:
            super().on_epoch_end(trainer, pl_module)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg:DictConfig):
    set_global_seed(cfg.seed)
    log = logging.getLogger(__name__)
    log.info("Info level message")
    log.debug("Debug level message")
    log.info(OmegaConf.to_yaml(cfg))
    exp_dir = os.path.join(cfg.exp_path, cfg.exp_name)
    fig_dir = os.path.join(exp_dir, 'figure') if 'mspec' in cfg.feature.target else None
    preprocessed_dir = os.path.join(cfg.data_path, 'preprocessed', 'target_feature', cfg.feature.target, f'{cfg.feature.sub_option}{cfg["feature"][cfg.feature.sub_option]}')
    log.info(exp_dir)
    
    if cfg.debug:
        wandb_logger = None
        ckpt_callback = None
        feat_norm = None
        fig_dir = None
    else:
        #ckpt_callback = ModelCheckpoint(dirpath=exp_dir, filename='{epoch:02d}-{val_loss:.3f}', verbose=True, mode='min', save_top_k=5, monitor='val_loss', save_last=True)
        '''
        ckpt_callback = SpecificEpochsCheckpoint(save_epochs=[100, 200, 300, 400, 500], dirpath=exp_dir, filename='{epoch:02d}-{val_loss:.3f}', verbose=True, mode='min', save_top_k=3, monitor='val_loss', save_last=True)
        '''
        ckpt_callback = ModelCheckpoint(
                dirpath=exp_dir,
                filename="epoch={epoch}-vloss={val_loss:.4f}-vacc={val_phone_accuracy:.4f}",
                save_top_k=-1,
                every_n_epochs=5
                )
        wandb_logger = WandbLogger(project='2025SS_ChannelGating', entity='dlswns8', name=cfg.exp_name, save_dir=cfg.exp_path)
        os.makedirs(exp_dir, exist_ok=True)
        config_path = os.path.join(exp_dir, "merged_config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(config=cfg, f=f)
        if fig_dir != None:
            os.makedirs(fig_dir, exist_ok=True)
            feat_norm, _ = pickle.load(open(os.path.join(preprocessed_dir, 'normalizer.pkl'), 'rb'))
        else:
            feat_norm = None

    if cfg.ckpt_epoch != None:
        ckpt_path = glob('%s/epoch=%02d*'%(exp_dir, cfg.ckpt_epoch))[0]
        model = EMGEncoder(cfg.emg_enc, cfg.optimizer, cfg.feature, len(phoneme_inventory), cfg.phoneme_loss_weight, cfg.batch_size, fig_dir, feat_norm)
        print(ckpt_path)
    else:
        ckpt_path = None
        model = EMGEncoder(cfg.emg_enc, cfg.optimizer, cfg.feature, len(phoneme_inventory), cfg.phoneme_loss_weight, cfg.batch_size, fig_dir, feat_norm)

    if cfg.pretrained_model != None:
        pretrained_path = glob('%s/%s/epoch=%02d*'%(cfg.exp_path, cfg.pretrained_model, cfg.pretrained_epoch))[0]
        model = load_partial_pretrained_model(model, pretrained_path, cfg.emg_enc.use_channel)

    #early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=10, verbose=False, mode='min')

    trainer_config = cfg.trainer
    OmegaConf.set_struct(trainer_config, False)
    #trainer_config['profiler'] = 'simple'
    OmegaConf.set_struct(trainer_config, True)

    trainer = pl.Trainer(logger=wandb_logger, \
            callbacks=[ckpt_callback] if ckpt_callback else [], **trainer_config)

    trainset = EMGDataset(preprocessed_dir, cfg.feature.target, 'train', cfg.feature.frame_rate, cfg.target_sec, cfg.feature.normalize)
    train_loader = DataLoader(trainset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=trainset.collate_raw)

    validset = EMGDataset(preprocessed_dir, cfg.feature.target, 'dev', cfg.feature.frame_rate, target_sec=None, normalize=cfg.feature.normalize)
    valid_loader = DataLoader(validset, batch_size=1, num_workers=4, collate_fn=validset.collate_raw)


    trainer.fit(model, train_loader, val_dataloaders=[valid_loader], ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()


