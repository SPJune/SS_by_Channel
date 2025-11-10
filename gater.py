import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMean(nn.Module):
    def forward(self, x, mask=None):
        # x: (B, C, T), mask: (B, 1, T) with 1 for valid, 0 for pad
        if mask is None:
            return x.mean(dim=-1)
        denom = mask.sum(dim=-1).clamp_min(1e-6)
        return (x * mask).sum(dim=-1) / denom

class MaskedMax(nn.Module):
    def forward(self, x, mask=None):
        if mask is None:
            return x.max(dim=-1).values
        # very negative where masked
        x = x.masked_fill(mask == 0, float("-inf"))
        return x.max(dim=-1).values

class AttnPool1d(nn.Module):
    """Channel-wise attention pooling over time."""
    def __init__(self, channels):
        super().__init__()
        # 1x1 conv over time acts like per-channel linear scorer across features at each time step
        self.scorer = nn.Conv1d(channels, channels, kernel_size=1, groups=1, bias=True)

    def forward(self, x, mask=None):
        # x: (B, C, T)
        scores = self.scorer(x)  # (B, C, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)  # softmax over T
        return (w * x).sum(dim=-1)     # (B, C)

class ChannelGatingNet(nn.Module):
    """
    Input:  x (B, T, C)
    Output: gates (B, C) in [0, 1]
    """
    def __init__(
        self,
        channels: int,
        hidden_channels: int = 128,
        conv_kernel: int = 5,
        pool: str = "attn",          # "attn" | "mean" | "max"
        se_ratio: int = 4,           # squeeze-excitation bottleneck ratio
        temperature: float = 1.0,    # >0, lower = sharper sigmoids
        dropout: float = 0.0
    ):
        super().__init__()
        self.channels = channels
        self.temperature = temperature

        pad = conv_kernel // 2

        # Temporal encoder (depthwise over time per channel) + pointwise mix for light cross-channel context
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=conv_kernel, padding=pad, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.GELU(),
        )

        # Pooling over time -> (B, C)
        if pool == "attn":
            self.pool = AttnPool1d(channels)
        elif pool == "mean":
            self.pool = MaskedMean()
        elif pool == "max":
            self.pool = MaskedMax()
        else:
            raise ValueError(f"Unknown pool: {pool}")
        self.pool_type = pool

        # Squeeze-Excitation across channels (use mean over time of raw input as an additional path)
        se_hidden = max(1, channels // se_ratio)
        self.se = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, se_hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(se_hidden, channels, bias=True),
        )

        # Fusion MLP across channels from temporal pathway
        self.mixer = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_channels, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels, bias=True),
        )

        # Final bias to ease optimization
        self.out_bias = nn.Parameter(torch.zeros(channels))

    def _make_mask(self, lengths, T, device):
        # returns (B, 1, T)
        mask = torch.arange(T, device=device).expand(len(lengths), T) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).to(x.dtype if (x:=None) is not None else torch.float32)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        """
        x: (B, T, C)
        lengths: optional (B,) valid lengths for masking (in frames)
        """
        B, T, C = x.shape
        assert C == self.channels, f"Expected channels={self.channels}, got {C}"

        # Build mask if provided
        mask = None
        if lengths is not None:
            device = x.device
            mask = torch.arange(T, device=device).expand(B, T) < lengths.unsqueeze(1)  # (B, T)
            mask = mask.unsqueeze(1)  # (B, 1, T)
            mask = mask.to(x.dtype)

        # Temporal encoding
        h = x.permute(0, 2, 1)  # (B, C, T)

        h = self.temporal(h)    # (B, C, T)

        # Temporal summary per channel
        if self.pool_type == "attn":
            z_temporal = self.pool(h, mask=mask)         # (B, C)
        elif self.pool_type == "mean":
            z_temporal = self.pool(h, mask=mask)         # (B, C)
        else:
            z_temporal = self.pool(h, mask=mask)         # (B, C)

        z_temporal = self.mixer(z_temporal)              # (B, C)

        # SE path (channel context from raw input mean over time)
        if lengths is None:
            x_mean = x.mean(dim=1)                       # (B, C)
        else:
            denom = mask.sum(dim=-1).clamp_min(1e-6)     # (B,1)
            x_mean = (h * mask).sum(dim=-1) / denom      # (B, C) using encoded h to respect mask
        z_se = self.se(x_mean)                           # (B, C)

        # Combine & squash to [0,1]
        logits = z_temporal + z_se + self.out_bias       # (B, C)
        gates = torch.sigmoid(logits / self.temperature) # (B, C), in [0,1]
        return gates
