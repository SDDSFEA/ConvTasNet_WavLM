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
        # x: [B,C,T]
        if x.dim() != 3:
            raise RuntimeError("GlobalLayerNorm expects [B,C,T].")
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
        # x: [B,C,T] -> [B,T,C] -> LN -> [B,C,T]
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
        # accepts [B,T] or [B,1,T] or [B,C,T]
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
# WavLM encoder (your local modeling_wavlm.py)
# ============================================================
try:
    from modeling_wavlm import WavLMModel
except Exception:
    WavLMModel = None

class WavLMencoder(nn.Module):
    """
    Returns semantic features: [B, T_sem, D]
    """
    def __init__(self, wavlm_name: str = "microsoft/wavlm-large"):
        super().__init__()
        if WavLMModel is None:
            raise ImportError(
                "WavLMModel not found. Please ensure modeling_wavlm.py provides WavLMModel."
            )
        self.encoder = WavLMModel.from_pretrained(wavlm_name)

    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None
    ) -> torch.Tensor:
        out = self.encoder(input_values, attention_mask=attention_mask)
        # keep consistent with your earlier code: out[1] is "un-downsampled feature"
        sem = out[1]  # [B, T_sem, D]
        return sem

# ============================================================
# Semantic alignment to TCN time axis (learnable + interpolate)
# ============================================================
class SemanticAlignToEnc(nn.Module):
    """
    Align WavLM semantic features to ConvTasNet encoder/TCN time length (T_enc).

    Steps:
      1) Project semantic dim D -> hidden
      2) Interpolate along time to target_len
      3) Temporal smoothing conv (depthwise-like) for stability
      4) Project hidden -> out_channels (typically B)
    """
    def __init__(
        self,
        semantic_dim: int,
        out_channels: int,
        hidden: int = 256,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pre = nn.Conv1d(semantic_dim, hidden, kernel_size=1)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

        # smoothing conv on time axis (lightweight, stable)
        pad = (kernel_size - 1) // 2
        self.smooth = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=pad, groups=1)

        self.post = nn.Conv1d(hidden, out_channels, kernel_size=1)

        # initialize post to near-zero so early training is close to baseline
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, sem_btd: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        sem_btd: [B, T_sem, D]
        returns: [B, out_channels, target_len]
        """
        x = sem_btd.transpose(1, 2)    # [B, D, T_sem]
        x = self.pre(x)                # [B, hidden, T_sem]
        x = self.act(x)
        x = self.drop(x)

        if x.size(-1) != target_len:
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)

        x = self.smooth(x)             # [B, hidden, target_len]
        x = self.act(x)
        x = self.post(x)               # [B, out_channels, target_len]
        return x

# ============================================================
# Gated residual for semantic injection
# ============================================================
class GatedResidual(nn.Module):
    """
    x <- x + sigmoid(logit) * delta
    """
    def __init__(self, init_logit: float = -5.0):
        super().__init__()
        self.gate_logit = nn.Parameter(torch.tensor(init_logit))

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate_logit)
        return x + g * delta

# ============================================================
# ConvTasNet TCN block (baseline)
# ============================================================
class Conv1D_Block(nn.Module):
    def __init__(
        self,
        in_channels=128,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
# Separation module (R repeats, X blocks each)
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
    ):
        super().__init__()
        self.repeats = nn.ModuleList()
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

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        for rep in self.repeats:
            for block in rep["blocks"]:
                x_bct = block(x_bct)
        return x_bct

# ============================================================
# ConvTasNet + WavLM-large alignment injection
# ============================================================
class ConvTasNet(nn.Module):
    """
    Args consistent with your net_conf:
      N, L, B, H, P, X, R, norm, num_spks, activate, causal
    """
    def __init__(
        self,
        N: int = 512,
        L: int = 16,
        B: int = 128,
        H: int = 512,
        P: int = 3,
        X: int = 8,
        R: int = 3,
        norm: str = "gln",
        num_spks: int = 2,
        activate: str = "relu",
        causal: bool = False,

        # semantic branch
        wavlm_name: str = "microsoft/wavlm-large",
        freeze_wavlm: bool = True,
        sem_align_hidden: int = 256,
        sem_align_kernel: int = 5,
        sem_align_dropout: float = 0.0,
        sem_gate_init_logit: float = -5.0,

        # legacy args accepted (ignored, for config compatibility)
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

        # ---- ConvTasNet encoder ----
        self.encoder = Conv1D(1, N, L, stride=L // 2, padding=0)  # [B,1,T] -> [B,N,T_enc]
        self.layern = select_norm("cln", N)
        self.bottleneck = Conv1D(N, B, 1)                         # [B,N,T_enc] -> [B,B,T_enc]

        # ---- WavLM semantic ----
        self.wavlm_encoder = WavLMencoder(wavlm_name=wavlm_name)
        semantic_dim = self.wavlm_encoder.encoder.config.hidden_size

        if freeze_wavlm:
            for p in self.wavlm_encoder.parameters():
                p.requires_grad = False

        # align semantic to T_enc and project to B channels
        self.sem_align = SemanticAlignToEnc(
            semantic_dim=semantic_dim,
            out_channels=B,
            hidden=sem_align_hidden,
            kernel_size=sem_align_kernel,
            dropout=sem_align_dropout,
        )
        self.sem_inject_gate = GatedResidual(init_logit=sem_gate_init_logit)

        # ---- separation (TCN) ----
        self.separation = SeparationModule(
            R=R, X=X, in_channels=B, out_channels=H, kernel_size=P, norm=norm, causal=causal
        )

        # ---- masks & decoder ----
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
        x: [B,T] or [T]
        returns: list of separated signals, each [B,T]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise RuntimeError("ConvTasNet expects [B,T] or [T].")

        # ---- audio encoder ----
        w = self.encoder(x)                  # [B,N,T_enc]
        e = self.bottleneck(self.layern(w))  # [B,B,T_enc]
        T_enc = e.size(-1)

        # ---- semantic encoder ----
        sem = self.wavlm_encoder(input_values=x)     # [B,T_sem,D]
        sem_bct = self.sem_align(sem, target_len=T_enc)  # [B,B,T_enc]

        # ---- inject semantic into TCN input (safe) ----
        e = self.sem_inject_gate(e, sem_bct)

        # ---- TCN ----
        e = self.separation(e)               # [B,B,T_enc]

        # ---- masks ----
        m = self.gen_masks(e)                # [B,num_spks*N,T_enc]
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        m = self.activation(torch.stack(m, dim=0))   # [S,B,N,T_enc]

        # ---- decode ----
        d = [w * m[i] for i in range(self.num_spks)]
        s_hat = [self.decoder(d[i], squeeze=True) for i in range(self.num_spks)]
        return s_hat

    # optional param groups
    def get_wavlmencoder_parameters(self):
        return list(self.wavlm_encoder.parameters())

    def get_semantic_injection_parameters(self):
        params = list(self.sem_align.parameters()) + list(self.sem_inject_gate.parameters())
        return params

    def get_audio_parameters(self):
        excluded = set(id(p) for p in self.get_wavlmencoder_parameters())
        excluded |= set(id(p) for p in self.get_semantic_injection_parameters())
        return [p for p in self.parameters() if id(p) not in excluded]

# ============================================================
# Quick sanity test
# ============================================================
def test():
    # Example with your config
    net_conf = dict(
        N=512, L=16, B=128, H=512, P=3, X=8, R=3,
        norm="gln", num_spks=2, activate="relu", causal=False,
        wavlm_name="microsoft/wavlm-large",
        freeze_wavlm=True,
    )

    net = ConvTasNet(**net_conf)
    x = torch.randn(2, 32000)  # 2 samples, 2 seconds @ 16k
    y = net(x)
    print(len(y), y[0].shape, y[1].shape)

if __name__ == "__main__":
    test()

