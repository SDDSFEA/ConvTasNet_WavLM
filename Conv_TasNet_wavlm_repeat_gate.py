import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any

# ============================================================
# Utils: Norms
# ============================================================
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
        x = x.transpose(1, 2)
        x = super().forward(x)
        return x.transpose(1, 2)

def select_norm(norm: str, dim: int):
    if norm == "gln":
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    return nn.BatchNorm1d(dim)

# ============================================================
# Conv wrappers
# ============================================================
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

# ============================================================
# WavLM encoder stub
# ============================================================
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
        wavlm_hidden_stages = out[1]  # [B, T_sem, D]
        return wavlm_hidden_stages

def masked_mean_pool(x_btd: torch.Tensor, mask_bt: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    if mask_bt is None:
        return x_btd.mean(dim=1)
    w = mask_bt.to(x_btd.dtype).unsqueeze(-1)
    num = (x_btd * w).sum(dim=1)
    den = w.sum(dim=1).clamp_min(eps)
    return num / den

# ============================================================
# Basic ConvTasNet TCN block (no semantic injection inside block)
# ============================================================
class Conv1D_Block(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=512,
        kernel_size=3,
        dilation=1,
        norm='gln',
        causal=False,
    ):
        super().__init__()
        self.conv1x1 = Conv1D(in_channels, out_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm, out_channels)

        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else (dilation * (kernel_size - 1))
        self.dwconv = Conv1D(
            out_channels, out_channels, kernel_size,
            groups=out_channels, padding=self.pad, dilation=dilation
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm, out_channels)

        self.sc_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.causal = causal

    def forward(self, x: torch.Tensor):
        c = self.conv1x1(x)
        c = self.prelu1(c)
        c = self.norm1(c)

        c = self.dwconv(c)
        if self.causal:
            c = c[:, :, :-self.pad]

        c = self.prelu2(c)
        c = self.norm2(c)

        c = self.sc_conv(c)
        return x + c

# ============================================================
# Semantic -> per-repeat residual gate
# ============================================================
class RepeatGate(nn.Module):
    """
    Given semantic vector s [B,D], output gate:
      - scalar gate: [B,1,1]
      - channel-wise gate: [B,C,1] (recommended)
    """
    def __init__(
        self,
        semantic_dim: int,
        channels: int,
        gate_type: str = "channel",  # "scalar" or "channel"
        hidden: int = 256,
        init_logit: float = -5.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert gate_type in ["scalar", "channel"]
        self.gate_type = gate_type
        out_dim = 1 if gate_type == "scalar" else channels

        self.mlp = nn.Sequential(
            nn.Linear(semantic_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

        # bias init => small gate at start (sigmoid(logit))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, init_logit)

    def forward(self, s_bd: torch.Tensor, T: int) -> torch.Tensor:
        """
        returns g shaped:
          scalar:  [B,1,1]
          channel: [B,C,1]
        """
        g = self.mlp(s_bd)  # [B,1] or [B,C]
        g = torch.sigmoid(g)
        if self.gate_type == "scalar":
            return g.unsqueeze(-1)  # [B,1,1]
        return g.unsqueeze(-1)      # [B,C,1]

# ============================================================
# Separation module: per-repeat gating on repeat residual
# ============================================================
class SeparationModule(nn.Module):
    def __init__(
        self,
        R: int,
        X: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm: str,
        causal: bool,
        semantic_dim: int,
        gate_type: str = "channel",   # "scalar" or "channel"
        gate_hidden: int = 256,
        gate_init_logit: float = -5.0,
        gate_dropout: float = 0.0,
        use_skip_scaling: bool = True,  # optional stabilize: scale residual by 1/sqrt(R)
    ):
        super().__init__()
        self.R = R
        self.use_skip_scaling = use_skip_scaling

        self.repeats = nn.ModuleList()
        self.repeat_gates = nn.ModuleList()

        for _ in range(R):
            blocks = nn.ModuleList([
                Conv1D_Block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    norm=norm,
                    causal=causal,
                )
                for i in range(X)
            ])
            self.repeats.append(nn.ModuleDict({"blocks": blocks}))
            self.repeat_gates.append(
                RepeatGate(
                    semantic_dim=semantic_dim,
                    channels=in_channels,   # gate on the repeat residual in bottleneck space [B,B,T]
                    gate_type=gate_type,
                    hidden=gate_hidden,
                    init_logit=gate_init_logit,
                    dropout=gate_dropout,
                )
            )

    def forward(self, x_bct: torch.Tensor, s_bd: torch.Tensor):
        """
        x_bct: [B, Bn, T]
        s_bd:  [B, D]
        """
        scale = (self.R ** -0.5) if self.use_skip_scaling else 1.0

        for rep, gate in zip(self.repeats, self.repeat_gates):
            x_in = x_bct
            for block in rep["blocks"]:
                x_bct = block(x_bct)

            # gated residual of this repeat
            delta = x_bct - x_in                      # [B,Bn,T]
            g = gate(s_bd, T=delta.shape[-1])         # [B,1,1] or [B,Bn,1]
            x_bct = x_in + scale * (g * delta)

        return x_bct

# ============================================================
# ConvTasNet with per-repeat semantic gate
# ============================================================
class ConvTasNet(nn.Module):
    def __init__(
        self,
        # ---- original ConvTasNet args ----
        N=512, L=16, B=128, H=512, P=3, X=8, R=3,
        norm="gln", num_spks=2, activate="relu", causal=False,
        wavlm_name="microsoft/wavlm-base",
        freeze_wavlm=True,

        # ---- gating config (NEW) ----
        gate_type: str = "channel",   # "scalar" or "channel"
        gate_hidden: int = 256,
        gate_init_logit: float = -5.0,
        gate_dropout: float = 0.0,
        use_skip_scaling: bool = True,

        # ---- legacy args accepted (ignored) ----
        attn_dim: Optional[int] = None,
        attn_dropout: Optional[float] = None,
        sem_downsample_stride: Optional[int] = None,
        use_kmem: Optional[bool] = None,
        kmem_K: Optional[int] = None,
        kmem_dim: Optional[int] = None,
        kmem_dropout: Optional[float] = None,
        fuse: Optional[str] = None,

        **kwargs: Any,
    ):
        super().__init__()
        self._unused_kwargs = kwargs

        # audio encoder
        self.encoder = Conv1D(1, N, L, stride=L // 2, padding=0)
        self.layern = select_norm("cln", N)
        self.bottleneck = Conv1D(N, B, 1)

        # semantic encoder
        self.wavlm_encoder = WavLMencoder(wavlm_name=wavlm_name)
        semantic_dim = self.wavlm_encoder.encoder.config.hidden_size

        if freeze_wavlm:
            for p in self.wavlm_encoder.parameters():
                p.requires_grad = False

        # separation with per-repeat gating
        self.separation = SeparationModule(
            R=R, X=X,
            in_channels=B, out_channels=H,
            kernel_size=P, norm=norm, causal=causal,
            semantic_dim=semantic_dim,
            gate_type=gate_type,
            gate_hidden=gate_hidden,
            gate_init_logit=gate_init_logit,
            gate_dropout=gate_dropout,
            use_skip_scaling=use_skip_scaling,
        )

        # masks and decoder
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
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise RuntimeError("ConvTasNet expects [B,T] or [T].")

        # audio path
        w = self.encoder(x)                  # [B,N,Tenc]
        e = self.bottleneck(self.layern(w))  # [B,B,Tenc]

        # semantic pooled vector
        sem = self.wavlm_encoder(input_values=x)  # [B,Tsem,D]
        s = masked_mean_pool(sem, mask_bt=None)   # [B,D]

        # separation with per-repeat semantic gate
        e = self.separation(e, s)                 # [B,B,Tenc]

        # masks
        m = self.gen_masks(e)                     # [B,num_spks*N,Tenc]
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        m = self.activation(torch.stack(m, dim=0))  # [num_spks,B,N,Tenc]

        # decode
        d = [w * m[i] for i in range(self.num_spks)]
        s_hat = [self.decoder(d[i], squeeze=True) for i in range(self.num_spks)]
        return s_hat

    # param groups (optional)
    def get_wavlmencoder_parameters(self):
        return list(self.wavlm_encoder.parameters())

    def get_gate_parameters(self):
        params = []
        for g in self.separation.repeat_gates:
            params += list(g.parameters())
        return params

    def get_audio_parameters(self):
        excluded = set(id(p) for p in self.get_wavlmencoder_parameters())
        excluded |= set(id(p) for p in self.get_gate_parameters())
        return [p for p in self.parameters() if id(p) not in excluded]

# ============================================================
# Quick sanity test
# ============================================================
def test():
    x = torch.randn(2, 32000)
    net = ConvTasNet(
        wavlm_name="microsoft/wavlm-base",
        N=256, L=16, B=128, H=256, P=3, X=4, R=2,
        gate_type="channel",
        gate_init_logit=-5.0,
        freeze_wavlm=True,
        # legacy args shouldn't crash:
        attn_dim=128, attn_dropout=0.1, sem_downsample_stride=2, fuse="x"
    )
    y = net(x)
    print(len(y), y[0].shape, y[1].shape)

if __name__ == "__main__":
    test()

