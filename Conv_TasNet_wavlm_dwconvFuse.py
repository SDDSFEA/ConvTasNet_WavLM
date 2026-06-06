import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# -------------------------
# Utils: Norms
# -------------------------
class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, 1))
            self.bias = nn.Parameter(torch.zeros(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x: [B, C, T]
        if x.dim() != 3:
            raise RuntimeError("GlobalLayerNorm expects 3D tensor [B,C,T].")
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        if self.elementwise_affine:
            x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x

class CumulativeLayerNorm(nn.LayerNorm):
    def __init__(self, dim, elementwise_affine=True):
        super().__init__(dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: [B, C, T] -> [B, T, C] -> LN -> [B, C, T]
        x = x.transpose(1, 2)
        x = super().forward(x)
        return x.transpose(1, 2)

def select_norm(norm: str, dim: int):
    if norm == "gln":
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    return nn.BatchNorm1d(dim)

# -------------------------
# Conv wrappers
# -------------------------
class Conv1D(nn.Conv1d):
    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError("Conv1D expects 2D/3D tensor.")
        x = super().forward(x if x.dim() == 3 else x.unsqueeze(1))
        return x.squeeze(1) if squeeze else x

class ConvTrans1D(nn.ConvTranspose1d):
    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError("ConvTrans1D expects 2D/3D tensor.")
        x = super().forward(x if x.dim() == 3 else x.unsqueeze(1))
        return x.squeeze(1) if squeeze else x

# -------------------------
# (Optional) WavLM encoder stub
# Replace this with your real WavLMModel import.
# -------------------------
try:
    from modeling_wavlm import WavLMModel
except Exception:
    WavLMModel = None

class WavLMencoder(nn.Module):
    """
    Returns semantic features: [B, T_sem, D]
    """
    def __init__(self, wavlm_name: str):
        super().__init__()
        if WavLMModel is None:
            raise ImportError("WavLMModel not found. Please ensure modeling_wavlm.py is available.")
        self.encoder = WavLMModel.from_pretrained(wavlm_name)

    def forward(self, input_values: torch.FloatTensor, attention_mask: Optional[torch.FloatTensor] = None):
        out = self.encoder(input_values, attention_mask=attention_mask)
        # You used out[1] as "un-downsampled". Keep consistent:
        wavlm_hidden_stages = out[1]  # [B, T_sem, D]
        return wavlm_hidden_stages

# -------------------------
# Gating (stable residual)
# -------------------------
class GatedResidual(nn.Module):
    def __init__(self, init_logit: float = -2.0):
        super().__init__()
        self.gate_logit = nn.Parameter(torch.tensor(init_logit))

    def forward(self, x, delta):
        g = torch.sigmoid(self.gate_logit)
        return x + g * delta

# -------------------------
# Cross-attention (delta) module
# Q from audio [B,C,T], K/V from sem [B,Tw,D]
# returns delta [B,C,T]
# -------------------------
class CrossAttnDelta(nn.Module):
    def __init__(self, audio_dim: int, semantic_dim: int, attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(audio_dim, attn_dim)
        self.k_proj = nn.Linear(semantic_dim, attn_dim)
        self.v_proj = nn.Linear(semantic_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, audio_dim)
        self.scale = attn_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        # LN works well on [B,T,C]
        self.ln_q = nn.LayerNorm(audio_dim)

    def forward(self, x_ct: torch.Tensor, sem_btd: torch.Tensor):
        """
        x_ct: [B, C, T]
        sem_btd: [B, T_sem, D]
        """
        B, C, T = x_ct.shape
        x = x_ct.transpose(1, 2)             # [B, T, C]
        x = self.ln_q(x)

        Q = self.q_proj(x)                   # [B, T, A]
        K = self.k_proj(sem_btd)             # [B, T_sem, A]
        V = self.v_proj(sem_btd)             # [B, T_sem, A]

        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B,T,T_sem]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        ctx = attn @ V                       # [B,T,A]
        delta = self.out_proj(ctx)           # [B,T,C]
        return delta.transpose(1, 2)         # [B,C,T]

# -------------------------
# Optional semantic downsampling (speed/stability)
# -------------------------
class SemanticDownsampler(nn.Module):
    """
    Downsample semantic sequence along time to reduce attention cost.
    Input:  [B, T_sem, D]
    Output: [B, T_sem', D]
    """
    def __init__(self, semantic_dim: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        if stride <= 1:
            self.pool = None
        else:
            # avg pool over time (channel-first)
            self.pool = nn.AvgPool1d(kernel_size=stride, stride=stride, ceil_mode=False)

    def forward(self, sem_btd: torch.Tensor):
        if self.pool is None:
            return sem_btd
        sem = sem_btd.transpose(1, 2)  # [B, D, T]
        sem = self.pool(sem)           # [B, D, T']
        return sem.transpose(1, 2)     # [B, T', D]

# -------------------------
# Your TCN block with cross-attn injection
# -------------------------
class Conv1D_Block(nn.Module):
    """
    Residual-only ConvTasNet block:
    x: [B, in_channels, T]
    """
    def __init__(self,
                 in_channels=256,
                 out_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm='gln',
                 causal=False):
        super().__init__()
        self.conv1x1 = Conv1D(in_channels, out_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm, out_channels)

        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else (dilation * (kernel_size - 1))
        self.dwconv = Conv1D(out_channels, out_channels, kernel_size,
                             groups=out_channels, padding=self.pad, dilation=dilation)

        # IMPORTANT: use these for stability
        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm, out_channels)

        self.sc_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.causal = causal

        # gate for attention injection
        self.attn_gate = GatedResidual(init_logit=-2.0)

    def forward(self, x: torch.Tensor, sem_btd: torch.Tensor, attn: CrossAttnDelta):
        # x: [B, Cin, T]
        c = self.conv1x1(x)     # [B, Cout, T]
        c = self.prelu1(c)
        c = self.norm1(c)

        c = self.dwconv(c)      # [B, Cout, T(+pad)]
        if self.causal:
            c = c[:, :, :-self.pad]  # crop to [B, Cout, T]

        c = self.prelu2(c)
        c = self.norm2(c)

        # Cross-attention delta, injected as gated residual
        delta = attn(c, sem_btd)      # [B, Cout, T]
        c = self.attn_gate(c, delta)  # [B, Cout, T]

        c = self.sc_conv(c)           # [B, Cin, T]
        return x + c

# -------------------------
# Separation module (R repeats, X blocks each), repeat-level shared attn
# -------------------------
class SeparationModule(nn.Module):
    def __init__(self,
                 R: int,
                 X: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 norm: str,
                 causal: bool,
                 semantic_dim: int,
                 attn_dim: int,
                 attn_dropout: float = 0.1,
                 sem_downsample_stride: int = 2):
        super().__init__()
        self.repeats = nn.ModuleList()

        for _ in range(R):
            self.repeats.append(nn.ModuleDict({
                # repeat-level shared attention (Cout matches dwconv output)
                "attn": CrossAttnDelta(audio_dim=out_channels, semantic_dim=semantic_dim,
                                       attn_dim=attn_dim, dropout=attn_dropout),
                # "sem_ds": SemanticDownsampler(semantic_dim, stride=sem_downsample_stride),
                "blocks": nn.ModuleList([
                    Conv1D_Block(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 dilation=2 ** i,
                                 norm=norm,
                                 causal=causal)
                    for i in range(X)
                ])
            }))

    def forward(self, x_bct: torch.Tensor, sem_btd: torch.Tensor):
        """
        x_bct: [B, Bn, T]
        sem_btd: [B, T_sem, D]
        """
        for rep in self.repeats:
            attn = rep["attn"]
            # sem_ds = rep["sem_ds"](sem_btd)  # downsample semantic sequence
            for block in rep["blocks"]:
                x_bct = block(x_bct, sem_btd, attn)
        return x_bct

# -------------------------
# ConvTasNet
# -------------------------
class ConvTasNet(nn.Module):
    def __init__(self,
                 N=512, L=16, B=128, H=512, P=3, X=8, R=3,
                 norm="gln", num_spks=2, activate="relu", causal=False,
                 wavlm_name="microsoft/wavlm-base",
                 attn_dim=128,
                 attn_dropout=0.1,
                 sem_downsample_stride=2,
                 freeze_wavlm=True,
                 fuse=None,):
        super().__init__()
        self.encoder = Conv1D(1, N, L, stride=L // 2, padding=0)
        self.layern = select_norm("cln", N)
        self.bottleneck = Conv1D(N, B, 1)

        self.wavlm_encoder = WavLMencoder(wavlm_name=wavlm_name)
        semantic_dim = self.wavlm_encoder.encoder.config.hidden_size

        # optional: freeze wavlm for stability
        if freeze_wavlm:
            for p in self.wavlm_encoder.parameters():
                p.requires_grad = False

        self.separation = SeparationModule(
            R=R, X=X, in_channels=B, out_channels=H, kernel_size=P, norm=norm, causal=causal,
            semantic_dim=semantic_dim, attn_dim=attn_dim, attn_dropout=attn_dropout,
            sem_downsample_stride=sem_downsample_stride
        )

        self.gen_masks = Conv1D(B, num_spks * N, 1)
        self.decoder = ConvTrans1D(N, 1, L, stride=L // 2)

        if activate == "relu":
            self.activation = nn.ReLU()
        elif activate == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activate == "softmax":
            self.activation = nn.Softmax(dim=0)
        else:
            raise ValueError(f"Unknown activate: {activate}")

        self.num_spks = num_spks
        self.causal = causal

    def forward(self, x: torch.Tensor):
        """
        x: [B, T] or [T]
        returns: list of separated waveforms (len = num_spks), each [B, T]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, T]
        if x.dim() != 2:
            raise RuntimeError("ConvTasNet expects [B,T] or [T].")

        # audio encoder path
        w = self.encoder(x)        # [B, N, T_enc]
        e = self.layern(w)
        e = self.bottleneck(e)     # [B, B, T_enc]

        # semantic path (wavlm)
        sem = self.wavlm_encoder(input_values=x)  # [B, T_sem, D]

        # separation with repeat-level shared cross-attn
        e = self.separation(e, sem)               # [B, B, T_enc]

        # masks
        m = self.gen_masks(e)                     # [B, num_spks*N, T_enc]
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        m = self.activation(torch.stack(m, dim=0))  # [num_spks, B, N, T_enc]

        # apply masks and decode
        d = [w * m[i] for i in range(self.num_spks)]  # each [B,N,T_enc]
        s = [self.decoder(d[i], squeeze=True) for i in range(self.num_spks)]  # each [B,T]
        return s

    def get_wavlmencoder_parameters(self):
        """Return parameters of WavLM encoder (semantic branch)."""
        return list(self.wavlm_encoder.parameters())

    def get_attention_parameters(self):
        """Return parameters of repeat-level cross-attention + semantic downsamplers."""
        params = []
        if hasattr(self, "separation") and hasattr(self.separation, "repeats"):
            for rep in self.separation.repeats:
                if "attn" in rep:
                    params += list(rep["attn"].parameters())
                if "sem_ds" in rep:
                    params += list(rep["sem_ds"].parameters())
        return params

    def get_audio_parameters(self):
        """
        Return parameters of pure audio path (encoder/TCN blocks/mask/decoder),
        excluding WavLM and attention modules.
        """
        sem_params = set(id(p) for p in self.get_wavlmencoder_parameters())
        attn_params = set(id(p) for p in self.get_attention_parameters())
        excluded = sem_params | attn_params

        audio_params = [p for p in self.parameters() if id(p) not in excluded]
        return audio_params

# -------------------------
# Quick sanity test
# -------------------------
def test():
    # random input
    x = torch.randn(2, 32000)  # 2 samples, 2 seconds at 16k
    net = ConvTasNet(
        wavlm_name="microsoft/wavlm-base",
        N=256, L=16, B=128, H=256, P=3, X=4, R=2,
        attn_dim=128,
        sem_downsample_stride=2,
        freeze_wavlm=True
    )
    y = net(x)
    print(len(y), y[0].shape, y[1].shape)

if __name__ == "__main__":
    test()

