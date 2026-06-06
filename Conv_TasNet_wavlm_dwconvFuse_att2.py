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
# WavLM encoder
# -------------------------
try:
    from modeling_wavlm import WavLMModel
except Exception:
    WavLMModel = None

class WavLMencoder(nn.Module):
    def __init__(self, wavlm_name: str):
        super().__init__()
        if WavLMModel is None:
            raise ImportError("WavLMModel not found. Please ensure modeling_wavlm.py is available.")
        self.encoder = WavLMModel.from_pretrained(wavlm_name)

    def forward(self, input_values: torch.FloatTensor, attention_mask: Optional[torch.FloatTensor] = None):
        out = self.encoder(input_values, attention_mask=attention_mask)
        wavlm_hidden_stages = out[1]  # [B, T_sem, D]
        return wavlm_hidden_stages


# -------------------------
# Gated residual
# -------------------------
class GatedResidual(nn.Module):
    def __init__(self, init_logit: float = -3.0):  # ✅ weaker init for stability
        super().__init__()
        self.gate_logit = nn.Parameter(torch.tensor(init_logit))

    def forward(self, x, delta):
        g = torch.sigmoid(self.gate_logit)
        return x + g * delta


# -------------------------
# Cross-attention (delta-only)
# -------------------------
class CrossAttnDelta(nn.Module):
    """
    Q from audio [B,C,T], K/V from sem [B,Tw,D]
    returns delta [B,C,T] (NO residual, NO gate)
    """
    def __init__(self, audio_dim: int, semantic_dim: int, attn_dim: int = 128, dropout: float = 0.1,
                 zero_init_out: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(audio_dim, attn_dim)
        self.k_proj = nn.Linear(semantic_dim, attn_dim)
        self.v_proj = nn.Linear(semantic_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, audio_dim)

        self.scale = attn_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        # LN works well on [B,T,C]
        self.ln_q = nn.LayerNorm(audio_dim)

        # ✅ very important: start from near-zero delta (keeps baseline behavior early)
        if zero_init_out:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_ct: torch.Tensor, sem_btd: torch.Tensor):
        """
        x_ct: [B, C, T]
        sem_btd: [B, T_sem, D]
        """
        x = x_ct.transpose(1, 2)   # [B,T,C]
        x = self.ln_q(x)

        Q = self.q_proj(x)         # [B,T,A]
        K = self.k_proj(sem_btd)   # [B,T_sem,A]
        V = self.v_proj(sem_btd)   # [B,T_sem,A]

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        ctx = attn @ V             # [B,T,A]
        delta = self.out_proj(ctx) # [B,T,C]
        return delta.transpose(1, 2)  # [B,C,T]


# -------------------------
# TCN block with: dwconv -> (prelu2+norm2) -> attn(delta) -> gate -> sc_conv
# -------------------------
class Conv1D_Block(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm='gln',
                 causal=False,
                 attn_dropout: float = 0.1):
        super().__init__()
        self.conv1x1 = Conv1D(in_channels, out_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm, out_channels)

        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else (dilation * (kernel_size - 1))
        self.dwconv = Conv1D(out_channels, out_channels, kernel_size,
                             groups=out_channels, padding=self.pad, dilation=dilation)

        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm, out_channels)

        self.sc_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.causal = causal

        # ✅ gate weaker init
        self.attn_gate = GatedResidual(init_logit=-3.0)

        # ✅ stabilize delta branch
        self.delta_norm = select_norm(norm, out_channels)
        self.delta_dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, sem_btd: torch.Tensor, attn: CrossAttnDelta):
        c = self.conv1x1(x)
        c = self.prelu1(c)
        c = self.norm1(c)

        c = self.dwconv(c)
        if self.causal:
            c = c[:, :, :-self.pad]

        # ✅ "Norm first"
        c = self.prelu2(c)
        c = self.norm2(c)

        # ✅ delta-only attention
        delta = attn(c, sem_btd)

        # ✅ delta stabilization
        delta = self.delta_norm(delta)
        delta = self.delta_dropout(delta)

        # ✅ gated residual (single gate, not replacing)
        c = self.attn_gate(c, delta)

        c = self.sc_conv(c)
        return x + c


# -------------------------
# Separation module: repeat-level shared attn
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
                 attn_dropout: float = 0.1):
        super().__init__()
        self.repeats = nn.ModuleList()

        for _ in range(R):
            self.repeats.append(nn.ModuleDict({
                "attn": CrossAttnDelta(
                    audio_dim=out_channels,
                    semantic_dim=semantic_dim,
                    attn_dim=attn_dim,
                    dropout=attn_dropout,
                    zero_init_out=True,     # ✅ important
                ),
                "blocks": nn.ModuleList([
                    Conv1D_Block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        dilation=2 ** i,
                        norm=norm,
                        causal=causal,
                        attn_dropout=attn_dropout,  # ✅
                    )
                    for i in range(X)
                ])
            }))

    def forward(self, x_bct: torch.Tensor, sem_btd: torch.Tensor):
        for rep in self.repeats:
            attn = rep["attn"]
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
                 freeze_wavlm=True,
                 fuse=None,
                 **kwargs):  # ✅ swallow extra keys safely
        super().__init__()
        self.encoder = Conv1D(1, N, L, stride=L // 2, padding=0)
        self.layern = select_norm("cln", N)
        self.bottleneck = Conv1D(N, B, 1)

        self.wavlm_encoder = WavLMencoder(wavlm_name=wavlm_name)
        semantic_dim = self.wavlm_encoder.encoder.config.hidden_size

        if freeze_wavlm:
            for p in self.wavlm_encoder.parameters():
                p.requires_grad = False

        self.separation = SeparationModule(
            R=R, X=X, in_channels=B, out_channels=H, kernel_size=P, norm=norm, causal=causal,
            semantic_dim=semantic_dim, attn_dim=attn_dim, attn_dropout=attn_dropout
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
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise RuntimeError("ConvTasNet expects [B,T] or [T].")

        w = self.encoder(x)        # [B, N, T_enc]
        e = self.layern(w)
        e = self.bottleneck(e)     # [B, B, T_enc]

        sem = self.wavlm_encoder(input_values=x)  # [B, T_sem, D]

        e = self.separation(e, sem)               # [B, B, T_enc]

        m = self.gen_masks(e)                     # [B, num_spks*N, T_enc]
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        m = self.activation(torch.stack(m, dim=0))  # [num_spks, B, N, T_enc]

        d = [w * m[i] for i in range(self.num_spks)]
        s = [self.decoder(d[i], squeeze=True) for i in range(self.num_spks)]
        return s

    def get_wavlmencoder_parameters(self):
        return list(self.wavlm_encoder.parameters())

    def get_attention_parameters(self):
        params = []
        if hasattr(self, "separation") and hasattr(self.separation, "repeats"):
            for rep in self.separation.repeats:
                if "attn" in rep:
                    params += list(rep["attn"].parameters())
                for blk in rep["blocks"]:
                    params += list(blk.attn_gate.parameters())
        return params

    def get_audio_parameters(self):
        sem_params = set(id(p) for p in self.get_wavlmencoder_parameters())
        attn_params = set(id(p) for p in self.get_attention_parameters())
        excluded = sem_params | attn_params
        return [p for p in self.parameters() if id(p) not in excluded]


def test():
    x = torch.randn(2, 32000)
    net = ConvTasNet(
        wavlm_name="microsoft/wavlm-base",
        N=256, L=16, B=128, H=256, P=3, X=4, R=2,
        attn_dim=128,
        attn_dropout=0.1,
        freeze_wavlm=True
    )
    y = net(x)
    print(len(y), y[0].shape, y[1].shape)

if __name__ == "__main__":
    test()

