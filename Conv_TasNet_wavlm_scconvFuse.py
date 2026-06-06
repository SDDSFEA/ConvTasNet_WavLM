import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# -------------------------
# Norms
# -------------------------
class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
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
# WavLM encoder
# -------------------------
from modeling_wavlm import WavLMModel

class WavLMencoder(nn.Module):
    """
    returns semantic features: [B, T_sem, D]
    """
    supports_gradient_checkpointing = True
    def __init__(self, wavlm_name: str = "microsoft/wavlm-base"):
        super().__init__()
        self.encoder = WavLMModel.from_pretrained(wavlm_name)

    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        out = self.encoder(input_values, attention_mask=attention_mask)
        # keep your convention: out[1] is "un-downsampled feature"
        wavlm_hidden_stages = out[1]  # [B, T_sem, D]
        return wavlm_hidden_stages


# -------------------------
# Cross-attention modules (time-wise)
# -------------------------
class Gated(nn.Module):
    """
    out = x + sigmoid(gate_logit) * x_att
    """
    def __init__(self, init_logit: float = -2.0):
        super().__init__()
        self.gate_logit = nn.Parameter(torch.tensor(init_logit))

    def forward(self, x, x_att):
        g = torch.sigmoid(self.gate_logit)
        return x + g * x_att


class GatedCrossAttnAdapter(nn.Module):
    """
    Cross-attention that produces a delta in audio feature space.
    audio:    [B, C, T]
    semantic: [B, T_w, D]
    output:   [B, C, T]  (gated residual inside)
    """
    def __init__(
        self,
        audio_dim: int,
        semantic_dim: int,
        attn_dim: int = 128,
        norm_type: str = "gln",
        dropout: float = 0.1,
        init_gate_logit: float = -2.0,
    ):
        super().__init__()
        self.q_proj = nn.Linear(audio_dim, attn_dim)
        self.k_proj = nn.Linear(semantic_dim, attn_dim)
        self.v_proj = nn.Linear(semantic_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, audio_dim)

        self.scale = attn_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        # stabilize the delta branch
        self.post_norm = select_norm(norm_type, audio_dim)

        # gate
        # self.gate_logit = nn.Parameter(torch.tensor(init_gate_logit))

    def forward(self, x_bct: torch.Tensor, sem_btd: torch.Tensor):
        """
        x_bct: [B, C, T]
        sem_btd: [B, T_sem, D]
        """
        B, C, T = x_bct.shape

        # [B, C, T] -> [B, T, C]
        q_in = x_bct.transpose(1, 2)

        Q = self.q_proj(q_in)            # [B, T, A]
        K = self.k_proj(sem_btd)         # [B, T_sem, A]
        V = self.v_proj(sem_btd)         # [B, T_sem, A]

        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, T, T_sem]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        ctx = attn @ V                   # [B, T, A]
        delta = self.out_proj(ctx)       # [B, T, C]
        delta = delta.transpose(1, 2)    # [B, C, T]

        # normalize delta then gated residual
        delta = self.post_norm(delta)
        # g = torch.sigmoid(self.gate_logit)
        return x_bct + delta


# -------------------------
# TCN block (attention AFTER Sc_conv)
# -------------------------
class Conv1D_Block(nn.Module):
    """
    Residual-only block:
    x: [B, in_channels, T]
    """
    def __init__(
        self,
        in_channels=256,
        out_channels=512,
        kernel_size=3,
        dilation=1,
        norm="gln",
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

        # IMPORTANT: use these for stability
        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm, out_channels)

        self.sc_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.causal = causal

        # gate for attention injection at in_channels
        self.gate = Gated(init_logit=-2.0)

    def forward(self, x_bct: torch.Tensor, sem_btd: torch.Tensor, attn_module: GatedCrossAttnAdapter):
        # conv path
        c = self.conv1x1(x_bct)  # [B, out_channels, T]
        c = self.prelu1(c)
        c = self.norm1(c)

        c = self.dwconv(c)
        if self.causal:
            c = c[:, :, :-self.pad]

        c = self.prelu2(c)
        c = self.norm2(c)

        # back to in_channels
        c = self.sc_conv(c)  # [B, in_channels, T]

        # attention injection AFTER Sc_conv (in_channels space)
        # attn_module already returns x + gated_delta, but we want delta-style usage here:
        # easiest: compute attn_out = attn_module(c, sem) and gate it again would double-gate.
        # So we use a delta module style: take (attn_module(c,sem) - c) as delta, then gate once.
        c_att = attn_module(c, sem_btd) - c     # [B, in_channels, T] (pure delta)
        c = self.gate(c, c_att)                 # gated residual

        return x_bct + c


# -------------------------
# Separation module (repeat-level shared attention)
# -------------------------
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
        attn_dim: int,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.repeats = nn.ModuleList()

        for _ in range(R):
            self.repeats.append(nn.ModuleDict({
                # IMPORTANT: audio_dim = in_channels (because attention is AFTER Sc_conv)
                "attn": GatedCrossAttnAdapter(
                    audio_dim=in_channels,
                    semantic_dim=semantic_dim,
                    attn_dim=attn_dim,
                    norm_type=norm,
                    dropout=attn_dropout,
                    init_gate_logit=-2.0,
                ),
                "blocks": nn.ModuleList([
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
            }))

    def forward(self, x_bct: torch.Tensor, sem_btd: torch.Tensor):
        for rep in self.repeats:
            attn = rep["attn"]  # repeat-level shared
            for block in rep["blocks"]:
                x_bct = block(x_bct, sem_btd, attn)
        return x_bct


# -------------------------
# ConvTasNet
# -------------------------
class ConvTasNet(nn.Module):
    """
    N: encoder filters
    L: encoder filter length (samples)
    B: bottleneck channels (also in_channels for blocks)
    H: block out_channels
    P: block kernel size
    X: blocks per repeat
    R: repeats
    """
    def __init__(
        self,
        N=512,
        L=16,
        B=128,
        H=512,
        P=3,
        X=8,
        R=3,
        norm="gln",
        num_spks=2,
        activate="relu",
        causal=False,
        wavlm_name="microsoft/wavlm-base",
        attn_dim=128,
        attn_dropout=0.1,
        freeze_wavlm=True,
        fuse=None,     # accept config compatibility
        **kwargs,      # swallow extra config keys safely
    ):
        super().__init__()
        self.num_spks = num_spks
        self.activation_type = activate
        self.causal = causal
        self.fuse = fuse

        # Encoder
        self.encoder = Conv1D(1, N, L, stride=L // 2, padding=0)

        # Bottleneck
        self.layern = select_norm("cln", N)
        self.bottleneck = Conv1D(N, B, 1)

        # WavLM semantic branch
        self.wavlm_encoder = WavLMencoder(wavlm_name=wavlm_name)
        semantic_dim = self.wavlm_encoder.encoder.config.hidden_size

        if freeze_wavlm:
            for p in self.wavlm_encoder.parameters():
                p.requires_grad = False

        # Separator (TCN)
        self.separation = SeparationModule(
            R=R,
            X=X,
            in_channels=B,
            out_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal,
            semantic_dim=semantic_dim,
            attn_dim=attn_dim,
            attn_dropout=attn_dropout,
        )

        # Mask head + Decoder
        self.gen_masks = Conv1D(B, num_spks * N, 1)
        self.decoder = ConvTrans1D(N, 1, L, stride=L // 2)

        active_f = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=0),
        }
        if activate not in active_f:
            raise ValueError(f"Unknown activate: {activate}")
        self.activation = active_f[activate]

    def forward(self, x: torch.Tensor):
        # x: [B,T] or [T]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise RuntimeError("ConvTasNet expects [B,T] or [T].")

        # audio encoder path
        w = self.encoder(x)      # [B, N, T_enc]
        e = self.layern(w)
        e = self.bottleneck(e)   # [B, B, T_enc]

        # semantic path
        sem = self.wavlm_encoder(input_values=x)  # [B, T_sem, D]

        # separation
        e = self.separation(e, sem)               # [B, B, T_enc]

        # masks
        m = self.gen_masks(e)                     # [B, num_spks*N, T_enc]
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        m = self.activation(torch.stack(m, dim=0))  # [num_spks, B, N, T_enc]

        # apply masks and decode
        d = [w * m[i] for i in range(self.num_spks)]
        s = [self.decoder(d[i], squeeze=True) for i in range(self.num_spks)]
        return s

    # -------- helpers for optimizer grouping ----------
    def get_wavlmencoder_parameters(self):
        return list(self.wavlm_encoder.parameters())

    def get_attention_parameters(self):
        params = []
        if hasattr(self, "separation") and hasattr(self.separation, "repeats"):
            for rep in self.separation.repeats:
                if "attn" in rep:
                    params += list(rep["attn"].parameters())
                # block gates are inside blocks
                for blk in rep["blocks"]:
                    if hasattr(blk, "gate"):
                        params += list(blk.gate.parameters())
        return params

    def get_audio_parameters(self):
        sem_ids = set(id(p) for p in self.get_wavlmencoder_parameters())
        attn_ids = set(id(p) for p in self.get_attention_parameters())
        excluded = sem_ids | attn_ids
        return [p for p in self.parameters() if id(p) not in excluded]


# -------------------------
# quick test
# -------------------------
def test():
    x = torch.randn(2, 32000)  # [B,T]
    net = ConvTasNet(
        N=256, L=16, B=128, H=256, P=3, X=4, R=2,
        wavlm_name="microsoft/wavlm-base",
        attn_dim=128,
        freeze_wavlm=True,
        fuse="anything",  # should not crash
    )
    y = net(x)
    print(len(y), y[0].shape, y[1].shape)

if __name__ == "__main__":
    test()

