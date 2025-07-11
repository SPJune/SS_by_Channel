import hydra
import json
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import os
import pytorch_lightning as pl
import random
from torch import nn
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from typing import Optional
import sys

from loss import dtw_loss

from hifi_gan.env import AttrDict
from hifi_gan.models import Generator


def topk_masking(mask, k):
    topk_vals, topk_indices = torch.topk(mask, k)
    masked = torch.zeros_like(mask)
    masked[topk_indices] = topk_vals
    return masked

class ChannelDropout(nn.Module):
    def __init__(self, drop_prob=0.3):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):  # x: (B, T, C)
        if not self.training or self.drop_prob == 0:
            return x
        mask = (torch.rand(x.shape[-1]) > self.drop_prob).float().to(x.device)
        return x * mask

class Vocoder(object):
    def __init__(self, checkpoint_file='/data1/marg/spjune/silent_speech/pretrained_models/hifigan_finetuned/checkpoint', \
                 device='cuda', half=False):
        config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
        with open(config_file) as f:
            hparams = AttrDict(json.load(f))
        self.generator = Generator(hparams).to(device)
        self.generator.load_state_dict(torch.load(checkpoint_file)['generator'])
        self.generator.eval()
        if half:
            self.generator.half()
        self.generator.remove_weight_norm()

    def __call__(self, mel_spectrogram):
        with torch.no_grad():
            mel_spectrogram = mel_spectrogram.T[np.newaxis,:,:]
            audio = self.generator(mel_spectrogram)
        return audio.squeeze()
        
class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)

class EMGEncoder(pl.LightningModule):
    def __init__(self, emg_enc_config: DictConfig, optimizer_config: DictConfig, feature_config: DictConfig, num_aux_outs=None, phoneme_loss_weight=0.5, batch_size=128, fig_dir=None, feat_norm=None):
        super(EMGEncoder, self).__init__()
        self.save_hyperparameters("emg_enc_config", "optimizer_config", "feature_config", "num_aux_outs", "phoneme_loss_weight", "batch_size")
        model_size = emg_enc_config.model_size
        dropout = emg_enc_config.dropout
        num_layers = emg_enc_config.num_layers
        self.optimizer_config = optimizer_config
        num_outs = feature_config.dim
        self.learning_rate_warmup = optimizer_config.lr_warmup
        self.batch_size = batch_size
        emg_num_ch = len(emg_enc_config.use_channel)
        self.emg_ch = emg_enc_config.use_channel
        self.channel_dropout = ChannelDropout(emg_enc_config.get('channel_dropout', 0))

        self.conv_blocks = nn.Sequential(
            ResBlock(emg_num_ch, model_size, 2),
            ResBlock(model_size, model_size, 2),
            ResBlock(model_size, model_size, 2),
        )
        if feature_config.frame_rate == 50:
            self.conv_blocks.append(ResBlock(model_size, model_size, 2))
        self.w_raw_in = nn.Linear(model_size, model_size)

        #encoder_layer = nn.TransformerEncoderLayer(d_model=model_size, nhead=8, dim_feedforward=3072, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(d_model=model_size, nhead=8, relative_positional=True, relative_positional_distance=100, dim_feedforward=3072, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.w_out = nn.Linear(model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(model_size, num_aux_outs)

        self.loss_fn = dtw_loss
        self.phoneme_loss_weight = phoneme_loss_weight
        self.fig_dir = fig_dir
        self.feat_norm = feat_norm

    def forward(self, x_raw, f0=None, uv=None):
        # x shape is (batch, time, electrode)
        x_raw = x_raw[:,:,self.emg_ch]
        x_raw = self.channel_dropout(x_raw)
        x_raw = x_raw.transpose(1,2) # put channel before time for conv
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1,2)
        x_raw = self.w_raw_in(x_raw)

        x = x_raw

        x = x.transpose(0,1) # put time first
        x = self.transformer(x)
        x = x.transpose(0,1)
        feat = self.w_out(x)
        ph = self.w_aux(x)
        return feat, ph

    def adjust_length(self, feat, f0):
        batch_size = feat.shape[0]
        target_length = feat.shape[1]
        current_length = f0.shape[2]
        if current_length > target_length:
            return f0[:, :, :target_length]
    
        else:
            padding_length = target_length - current_length
            padding = f0[:, :, -1:].repeat(1, 1, padding_length)
            return torch.cat([f0, padding], dim=2)

    def configure_optimizers(self):
        optimizer_class = hydra.utils.get_class(self.optimizer_config.target)
        trainable_params = (
                            list(self.conv_blocks.parameters())
                            + list(self.w_raw_in.parameters())
                            + list(self.transformer.parameters())
                            + list(self.w_out.parameters())
                            + list(self.w_aux.parameters())
                            )
        optimizer = optimizer_class(self.parameters(), **self.optimizer_config.params)
        def lr_lambda(current_step: int):
            if current_step < self.learning_rate_warmup:
                return current_step/ float(max(1, self.learning_rate_warmup))
            return 1

        scheduler_warmup = LambdaLR(optimizer, lr_lambda)
        scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        return [optimizer], [{'scheduler': scheduler_warmup, 'interval': 'step', 'frequency': 1},
                    {'scheduler': scheduler_plateau, 'monitor': 'val_loss'}
                    ]

    def training_step(self, batch, batch_idx):
        x = batch['emg']
        y = batch['speech_features']
        y_ph = batch['phonemes']
        silents = batch['silents']
        target_lengths = batch['target_lengths']
        est_lengths = batch['est_lengths']
        f0 = batch['f0']
        uv = batch['uv']
        r = random.randrange(8)
        if r > 0:
            temp = x[:,r:,:] # shift left r
            x[:,:-r,:] = temp.clone()
            x[:,-r:,:] = 0
        y_hat, y_ph_hat = self(x, f0, uv)
        loss, loss_dist, loss_ph, _  = self.loss_fn(y_hat, y_ph_hat, y, y_ph, silents, self.phoneme_loss_weight, target_lengths, est_lengths)
        self.log('train_loss', loss, batch_size=self.batch_size)
        self.log('train_loss_dist', loss_dist, batch_size=self.batch_size)
        self.log('train_loss_ph', loss_ph, batch_size=self.batch_size)
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=self.batch_size)
        if batch_idx == 0 and self.fig_dir != None:
            self.save_mel_spectrogram(y, y_hat, step_type='train')
            self.save_mel_spectrogram(y, y_hat, step_type='train', normalize=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['emg']
        y = batch['speech_features']
        y_ph = batch['phonemes']
        silents = batch['silents']
        target_lengths = batch['target_lengths']
        est_lengths = batch['est_lengths']
        mel = batch['mel']
        f0 = batch['f0']
        uv = batch['uv']
        y_hat, y_ph_hat = self(x, f0, uv)
        loss, loss_dist, loss_ph, phone_acc  = self.loss_fn(y_hat, y_ph_hat, y, y_ph, silents, self.phoneme_loss_weight, target_lengths, est_lengths, phoneme_eval=True)
        self.log('val_loss', loss, batch_size=1)
        self.log('val_loss_dist', loss_dist, batch_size=1)
        self.log('val_loss_ph', loss_ph, batch_size=1)
        self.log('val_phone_accuracy', phone_acc, batch_size=1, prog_bar=False)
        if batch_idx == 0 and self.fig_dir != None:
            self.save_mel_spectrogram(y, y_hat, step_type='valid')
            self.save_mel_spectrogram(y, y_hat, step_type='valid', normalize=False)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['emg']
        y = batch['speech_features']
        y_ph = batch['phonemes']
        silents = batch['silents']
        target_lengths = batch['target_lengths']
        est_lengths = batch['est_lengths']
        mel = batch['mel']
        f0 = batch['f0']
        uv = batch['uv']
        y_hat, y_ph_hat = self(x, f0, uv)
        loss, loss_dist, loss_ph, phone_acc  = self.loss_fn(y_hat, y_ph_hat, y, y_ph, silents, self.phoneme_loss_weight, target_lengths, est_lengths, phoneme_eval=True)
        self.log('test_loss', loss, batch_size=1)
        return loss

    def save_mel_spectrogram(self, y, y_hat, step_type, normalize=True):
        y_spec = y[0].detach().cpu().numpy()
        y_hat_spec = y_hat[0].detach().cpu().numpy()
        if normalize:
            y_spec = self.feat_norm.inverse(y_spec).T
            y_hat_spec = self.feat_norm.inverse(y_hat_spec).T
        else:
            y_spec = y_spec.T
            y_hat_spec = y_hat_spec.T
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].imshow(y_spec, origin='lower', aspect='auto')
        axs[0].set_title('Ground Truth Mel-Spectrogram')
        axs[1].imshow(y_hat_spec, origin='lower', aspect='auto')
        axs[1].set_title('Predicted Mel-Spectrogram')
        
        save_path = os.path.join(self.fig_dir, f'{step_type}_mel_spectrogram_{normalize}.png')
        plt.savefig(save_path)
        plt.close(fig)

class TransformerEncoderLayer(nn.Module):
    # Adapted from pytorch source
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, relative_positional=True, relative_positional_distance=100):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, relative_positional=relative_positional, relative_positional_distance=relative_positional_distance)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model=256, n_head=4, dropout=0.1, relative_positional=True, relative_positional_distance=100):
    super().__init__()
    self.d_model = d_model
    self.n_head = n_head
    d_qkv = d_model // n_head
    assert d_qkv * n_head == d_model, 'd_model must be divisible by n_head'
    self.d_qkv = d_qkv

    self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_o = nn.Parameter(torch.Tensor(n_head, d_qkv, d_model))
    nn.init.xavier_normal_(self.w_q)
    nn.init.xavier_normal_(self.w_k)
    nn.init.xavier_normal_(self.w_v)
    nn.init.xavier_normal_(self.w_o)

    self.dropout = nn.Dropout(dropout)
    self.batch_first = False

    if relative_positional:
        self.relative_positional = LearnedRelativePositionalEmbedding(relative_positional_distance, n_head, d_qkv, True)
    else:
        self.relative_positional = None

  def forward(self, x):
    """Runs the multi-head self-attention layer.

    Args:
      x: the input to the layer, a tensor of shape [length, batch_size, d_model]
    Returns:
      A single tensor containing the output from this layer
    """

    q = torch.einsum('tbf,hfa->bhta', x, self.w_q)
    k = torch.einsum('tbf,hfa->bhta', x, self.w_k)
    v = torch.einsum('tbf,hfa->bhta', x, self.w_v)
    logits = torch.einsum('bhqa,bhka->bhqk', q, k) / (self.d_qkv ** 0.5)

    if self.relative_positional is not None:
        q_pos = q.permute(2,0,1,3) #bhqd->qbhd
        l,b,h,d = q_pos.size()
        position_logits, _ = self.relative_positional(q_pos.reshape(l,b*h,d))
        # (bh)qk
        logits = logits + position_logits.view(b,h,l,l)

    probs = F.softmax(logits, dim=-1)
    probs = self.dropout(probs)
    o = torch.einsum('bhqk,bhka->bhqa', probs, v)
    out = torch.einsum('bhta,haf->tbf', o, self.w_o)
    return out

class LearnedRelativePositionalEmbedding(nn.Module):
    # from https://github.com/pytorch/fairseq/pull/2225/commits/a7fb63f2b84d5b20c8855e9c3372a95e5d0ea073
    """
    This module learns relative positional embeddings up to a fixed
    maximum size. These are masked for decoder and unmasked for encoder
    self attention.
    By default the embeddings are added to keys, but could be added to
    values as well.
    Args:
        max_relative_pos (int): the maximum relative positions to compute embeddings for
        num_heads (int): number of attention heads
        embedding_dim (int): depth of embeddings
        unmasked (bool): if the attention is unmasked (for transformer encoder)
        heads_share_embeddings (bool): if heads share the same relative positional embeddings
        add_to_values (bool): compute embeddings to be added to values as well
    """

    def __init__(
            self,
            max_relative_pos: int,
            num_heads: int,
            embedding_dim: int,
            unmasked: bool = False,
            heads_share_embeddings: bool = False,
            add_to_values: bool = False):
        super().__init__()
        self.max_relative_pos = max_relative_pos
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.unmasked = unmasked
        self.heads_share_embeddings = heads_share_embeddings
        self.add_to_values = add_to_values
        num_embeddings = (
            2 * max_relative_pos - 1
            if unmasked
            else max_relative_pos
        )
        embedding_size = (
            [num_embeddings, embedding_dim, 1]
            if heads_share_embeddings
            else [num_heads, num_embeddings, embedding_dim, 1]
        )
        if add_to_values:
            embedding_size[-1] = 2
        initial_stddev = embedding_dim**(-0.5)
        self.embeddings = nn.Parameter(torch.zeros(*embedding_size))
        nn.init.normal_(self.embeddings, mean=0.0, std=initial_stddev)

    def forward(self, query, saved_state=None):
        """
        Computes relative positional embeddings to be added to keys (and optionally values),
        multiplies the embeddings for keys with queries to create positional logits,
        returns the positional logits, along with embeddings for values (optionally)
        which could be added to values outside this module.
        Args:
            query (torch.Tensor): query tensor
            saved_state (dict): saved state from previous time step
        Shapes:
            query: `(length, batch_size*num_heads, embed_dim)`
        Returns:
            tuple(torch.Tensor):
                - positional logits
                - relative positional embeddings to be added to values
        """
        # During inference when previous states are cached
        if saved_state is not None and "prev_key" in saved_state:
            assert not self.unmasked, "This should only be for decoder attention"
            length = saved_state["prev_key"].shape[-2] + 1  # `length - 1` keys are cached,
                                                            # `+ 1` for the current time step
            decoder_step = True
        else:
            length = query.shape[0]
            decoder_step = False

        used_embeddings = self.get_embeddings_for_query(length)

        values_embeddings = (
            used_embeddings[..., 1]
            if self.add_to_values
            else None
        )
        positional_logits = self.calculate_positional_logits(query, used_embeddings[..., 0])
        positional_logits = self.relative_to_absolute_indexing(positional_logits, decoder_step)
        return (positional_logits, values_embeddings)

    def get_embeddings_for_query(self, length):
        """
        Extract the required embeddings. The maximum relative position between two time steps is
        `length` for masked case or `2*length - 1` for the unmasked case. If `length` is greater than
        `max_relative_pos`, we first pad the embeddings tensor with zero-embeddings, which represent
        embeddings when relative position is greater than `max_relative_pos`. In case `length` is
        less than `max_relative_pos`, we don't use the first `max_relative_pos - length embeddings`.
        Args:
            length (int): length of the query
        Returns:
            torch.Tensor: embeddings used by the query
        """
        pad_length = max(length - self.max_relative_pos, 0)
        start_pos = max(self.max_relative_pos - length, 0)
        if self.unmasked:
            with torch.no_grad():
                padded_embeddings = nn.functional.pad(
                    self.embeddings,
                    (0, 0, 0, 0, pad_length, pad_length)
                )
            used_embeddings = padded_embeddings.narrow(-3, start_pos, 2*length - 1)
        else:
            with torch.no_grad():
                padded_embeddings = nn.functional.pad(
                    self.embeddings,
                    (0, 0, 0, 0, pad_length, 0)
                )
            used_embeddings = padded_embeddings.narrow(-3, start_pos, length)
        return used_embeddings

    def calculate_positional_logits(self, query, relative_embeddings):
        """
        Multiplies query with the relative positional embeddings to create relative
        positional logits
        Args:
            query (torch.Tensor): Input tensor representing queries
            relative_embeddings (torch.Tensor): relative embeddings compatible with query
        Shapes:
            query: `(length, batch_size*num_heads, embed_dim)` if heads share embeddings
                   else `(length, batch_size, num_heads, embed_dim)`
            relative_embeddings: `(max_allowed_relative_positions, embed_dim)` if heads share embeddings
                                 else `(num_heads, max_allowed_relative_positions, embed_dim)`
                                 where `max_allowed_relative_positions` is `length` if masked
                                 else `2*length - 1`
        Returns:
            torch.Tensor: relative positional logits
        """
        if self.heads_share_embeddings:
            positional_logits = torch.einsum("lbd,md->lbm", query, relative_embeddings)
        else:
            query = query.view(query.shape[0], -1, self.num_heads, self.embedding_dim)
            positional_logits = torch.einsum("lbhd,hmd->lbhm", query, relative_embeddings)
            positional_logits = positional_logits.contiguous().view(
                positional_logits.shape[0], -1, positional_logits.shape[-1]
            )
        # mask out tokens out of range
        length = query.size(0)
        if length > self.max_relative_pos:
            # there is some padding
            pad_length = length - self.max_relative_pos
            positional_logits[:,:,:pad_length] -= 1e8
            if self.unmasked:
                positional_logits[:,:,-pad_length:] -= 1e8
        return positional_logits

    def relative_to_absolute_indexing(self, x, decoder_step):
        """
        Index tensor x (relative positional logits) in terms of absolute positions
        rather than relative positions. Last dimension of x represents relative position
        with respect to the first dimension, whereas returned tensor has both the first
        and last dimension indexed with absolute positions.
        Args:
            x (torch.Tensor): positional logits indexed by relative positions
            decoder_step (bool): is this is a single decoder step (during inference)
        Shapes:
            x: `(length, batch_size*num_heads, length)` for masked case or
               `(length, batch_size*num_heads, 2*length - 1)` for unmasked
        Returns:
            torch.Tensor: positional logits represented using absolute positions
        """
        length, bsz_heads, _ = x.shape

        if decoder_step:
            return x.contiguous().view(bsz_heads, 1, -1)

        if self.unmasked:
            x = nn.functional.pad(
                x,
                (0, 1)
            )
            x = x.transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length * 2 * length)
            x = nn.functional.pad(
                x,
                (0, length - 1)
            )
            # Reshape and slice out the padded elements.
            x = x.view(bsz_heads, length + 1, 2*length - 1)
            return x[:, :length, length-1:]
        else:
            x = nn.functional.pad(
                x,
                (1, 0)
            )
            x = x.transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length+1, length)
            return x[:, 1:, :]
