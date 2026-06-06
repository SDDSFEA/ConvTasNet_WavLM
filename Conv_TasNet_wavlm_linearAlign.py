import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from transformers.modeling_utils import PreTrainedModel
from typing import Optional
from transformers.configuration_utils import PretrainedConfig
from modeling_wavlm import WavLMModel
from wavlm_separator import WavLMSeparator

class WavLMencoder(nn.Module):
    # supports_gradient_checkpointing = True
    def __init__(
        self,
        wavlm_name: str = "microsoft/wavlm-base",
    ):
        # 初始化 config
        super().__init__()
        
        # 初始化 encoder
        self.encoder = WavLMModel.from_pretrained(wavlm_name)
        
        
    def forward(
        self,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
    ):
        
        encoder_outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
        )
    
        # encoder_hidden_states = encoder_outputs[0]
        wavlm_hidden_stages   = encoder_outputs[1]      # un-downsampled feature
        # wavlm_down_hidden_stages = encoder_outputs[2]
        # mixed_encoding_feature = wavlm_hidden_stages

        # # Here we add serialized CTC
        # sep_hidden_states = self.separator(mixed_encoding_feature)
        
        return wavlm_hidden_stages

class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
            input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
           this module has learnable per-element affine parameters 
           initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
        else:
            x = (x-mean)/torch.sqrt(var+self.eps)
        return x

class GatedCrossAttnAdapter(nn.Module):
    """
    Time-wise Cross-Attention Adapter
    audio:    [B, C, T]
    semantic: [B, T_w, D]
    """
    def __init__(
        self,
        audio_dim: int,
        semantic_dim: int,
        attn_dim: int = 128,
        norm_type: str = "gln",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.q_proj = nn.Linear(audio_dim, attn_dim)
        self.k_proj = nn.Linear(semantic_dim, attn_dim)
        self.v_proj = nn.Linear(semantic_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, audio_dim)

        self.scale = attn_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.norm = select_norm(norm_type, audio_dim)

        # global gate（非常重要）
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x, semantic_feat):
        """
        x:             [B, C, T]
        semantic_feat: [B, T_w, D]
        """
        B, C, T = x.shape
        _, T_w, _ = semantic_feat.shape

        # ---- reshape ----
        x_t = x.transpose(1, 2)           # [B, T, C]

        # ---- projections ----
        Q = self.q_proj(x_t)               # [B, T, A]
        K = self.k_proj(semantic_feat)     # [B, T_w, A]
        V = self.v_proj(semantic_feat)     # [B, T_w, A]

        # ---- attention ----
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # [B, T, A]
        context = self.out_proj(context)          # [B, T, C]
        context = context.transpose(1, 2)         # [B, C, T]

        # ---- gated residual ----
        g = torch.sigmoid(self.gate_logit)
        out = x + g * self.norm(context)

        return out

class ConcatAdapter(nn.Module):
    """
    Time-wise concat semantic features into audio features
    audio:    [B, C, T]
    semantic: [B, T, D]
    """
    def __init__(self, audio_dim, semantic_dim):
        super().__init__()
        self.proj = nn.Conv1d(
            audio_dim + semantic_dim,
            audio_dim,
            kernel_size=1,
            bias=True
        )

    def forward(self, x, semantic_feat):
        """
        x:             [B, C, T]
        semantic_feat: [B, T, D]
        """
        # [B, T, D] -> [B, D, T]
        semantic_feat = semantic_feat.transpose(1, 2)

        # concat on channel dim
        x = torch.cat([x, semantic_feat], dim=1)  # [B, C+D, T]

        # project back
        x = self.proj(x)                           # [B, C, T]
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim):
    # if norm not in ['gln', 'cln', 'bn']:
    #     # if x.dim() != 3:
    #         raise RuntimeError("{} accept 3D tensor as input".format(
    #             self.__name__))

    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    else:
        return nn.BatchNorm1d(dim)


class Conv1D(nn.Conv1d):
    '''
       Applies a 1D convolution over an input signal composed of several input planes.
    '''

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x

def align_linear_interp(z_sem: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Align semantic sequence to target length by linear interpolation.

    Args:
        z_sem: [B, T_sem, D]
        target_len: int (e.g., T_enc)

    Returns:
        z_aligned: [B, target_len, D]
    """
    assert z_sem.dim() == 3, f"Expected [B, T, D], got {z_sem.shape}"
    b, t, d = z_sem.shape
    if t == target_len:
        return z_sem

    # F.interpolate expects [B, C, T]
    z = z_sem.transpose(1, 2)  # [B, D, T_sem]
    z = F.interpolate(z, size=target_len, mode="linear", align_corners=False)
    z = z.transpose(1, 2)      # [B, target_len, D]
    return z

class ConvTrans1D(nn.ConvTranspose1d):
    '''
       This module can be seen as the gradient of Conv1d with respect to its input. 
       It is also known as a fractionally-strided convolution 
       or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class Conv1D_Block(nn.Module):
    '''
       Consider only residual links
    '''

    def __init__(self, in_channels=256, out_channels=512,
                 kernel_size=3, dilation=1, norm='gln', causal=False):
        super(Conv1D_Block, self).__init__()
        # conv 1 x 1
        self.conv1x1 = Conv1D(in_channels, out_channels, 1)
        self.PReLU_1 = nn.PReLU()
        self.norm_1 = select_norm(norm, out_channels)
        # not causal don't need to padding, causal need to pad+1 = kernel_size
        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise convolution
        self.dwconv = Conv1D(out_channels, out_channels, kernel_size,
                             groups=out_channels, padding=self.pad, dilation=dilation)
        self.PReLU_2 = nn.PReLU()
        self.norm_2 = select_norm(norm, out_channels)
        self.Sc_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.causal = causal

    def forward(self, x):
        # x: N x C x L
        # N x O_C x L
        c = self.conv1x1(x)
        # N x O_C x L
        c = self.PReLU_1(c)
        c = self.norm_1(c)
        # causal: N x O_C x (L+pad)
        # noncausal: N x O_C x L
        c = self.dwconv(c)
        # N x O_C x L
        if self.causal:
            c = c[:, :, :-self.pad]
        c = self.Sc_conv(c)
        return x+c

class SeparationModule(nn.Module):
    def __init__(self, R, X, in_channels, out_channels,
             kernel_size, norm, causal, semantic_dim, attn_dim,fuse=None):
        super().__init__()
        self.repeats = nn.ModuleList()
        self.gates = nn.ParameterList()
        
        for _ in range(R):
            if fuse == "attention":
                self.repeats.append(nn.ModuleDict({
                    "adapter": GatedCrossAttnAdapter(
                        audio_dim=in_channels,
                        semantic_dim=semantic_dim,
                        attn_dim=attn_dim,
                        norm_type=norm
                    ),
                    "blocks": nn.ModuleList([
                        Conv1D_Block(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            dilation=2**i,
                            norm=norm,
                            causal=causal
                        )
                        for i in range(X)
                    ])
                }))
            elif fuse == "concat":
                self.repeats.append(nn.ModuleDict({
                    "adapter": ConcatAdapter(
                        audio_dim=in_channels,
                        semantic_dim=semantic_dim
                    ),
                    "blocks": nn.ModuleList([
                        Conv1D_Block(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            dilation=2**i,
                            norm=norm,
                            causal=causal
                        )
                        for i in range(X)
                    ])
                }))
    
    def forward(self, x, text_feat):
        """
        x: [N, B, T]
        text_feat: [N, T, D]
        """
        
        for repeat in self.repeats:
            # ① 语义注入（一次）
            x = repeat["adapter"](x, text_feat)
            # ② 局部时序建模
            for block in repeat["blocks"]:
                x = block(x)
        return x
        
        # for repeat, gate in zip(self.repeats, self.gates):
        #     attn_out = repeat["attn"](x, text_feat)
        #     x = x + gate * attn_out

        #     for block in repeat["blocks"]:
        #         x = block(x)
        # return x 
    

class ConvTasNet(nn.Module):
    '''
       ConvTasNet module
       N	Number of ﬁlters in autoencoder
       L	Length of the ﬁlters (in samples)
       B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
       Sc	Number of channels in skip-connection paths’ 1 × 1-conv blocks
       H	Number of channels in convolutional blocks
       P	Kernel size in convolutional blocks
       X	Number of convolutional blocks in each repeat
       R	Number of repeats
    '''

    def __init__(self,
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
                 wavlm_name="wavlm_large",  # 新增：文本模型名称
                 attn_dim=256,
                 fuse="attention"):                 # 新增：注意力维度):                 
        super(ConvTasNet, self).__init__()
        # n x 1 x T => n x N x T
        self.encoder = Conv1D(1, N, L, stride=L // 2, padding=0)
        # n x N x T  Layer Normalization of Separation
        self.LayerN_S = select_norm('cln', N)
        # n x B x T  Conv 1 x 1 of  Separation
        self.BottleN_S = Conv1D(N, B, 1)
        
         # 新增：语义注入相关模块
        self.wavlm_encoder = WavLMencoder(
            wavlm_name=wavlm_name
        )
        # 合并文本特征维度
        # text_dim = self.text_encoder.embedding_dim * 2
        
        # Separation block
        # n x B x T => n x B x T
        # self.separation = self._Sequential_repeat(
        #     R, X, in_channels=B, out_channels=H, kernel_size=P, norm=norm, causal=causal)
        self.separation = SeparationModule(
            R=R,
            X=X,
            in_channels=B,
            out_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal,
            semantic_dim=self.wavlm_encoder.encoder.config.hidden_size,
            attn_dim=attn_dim,
            fuse=fuse
        )
        # n x B x T => n x 2*N x T
        self.gen_masks = Conv1D(B, num_spks*N, 1)
        # n x N x T => n x 1 x L
        self.decoder = ConvTrans1D(N, 1, L, stride=L//2)
        # activation function
        active_f = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=0)
        }
        self.activation_type = activate
        self.activation = active_f[activate]
        self.num_spks = num_spks

    

    def forward(self, x):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        # x: n x 1 x L => n x N x T
        w = self.encoder(x)
        # n x N x L => n x B x L
        e = self.LayerN_S(w)
        e = self.BottleN_S(e)
        
        wavlm_out = self.wavlm_encoder(input_values=x)
        
        T_enc = e.shape[-1]
        # ✅对齐到 TCN 时间轴
        if wavlm_out.shape[1] != T_enc:
            wavlm_out = align_linear_interp(wavlm_out, T_enc)
        # 合并文本特征
        # n x B x L => n x B x L
        e = self.separation(e,wavlm_out)
        # print(f"e shape after separation: {e.shape}")
        # n x B x L => n x num_spk*N x L
        m = self.gen_masks(e)
        # n x N x L x num_spks
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        # num_spks x n x N x L
        m = self.activation(torch.stack(m, dim=0))
        d = [w*m[i] for i in range(self.num_spks)]
        # decoder part num_spks x n x L
        s = [self.decoder(d[i], squeeze=True) for i in range(self.num_spks)]
        return s
    
    def get_semantic_parameters(self):
        params = []
        # TextEncoder
        params += list(self.wavlm_encoder.parameters())
        # GatedCrossAttention
        for repeat in self.separation.repeats:
            params += list(repeat["adapter"].parameters())
        return params
    
    def get_wavlmencoder_parameters(self):
        params = list(self.wavlm_encoder.parameters())
        return params
    
    def get_audio_parameters(self):
        """
        纯音频路径参数（encoder / conv blocks / decoder 等）
        """
        semantic_ids = set(id(p) for p in self.get_semantic_parameters())
        audio_params = [
            p for p in self.parameters()
            if id(p) not in semantic_ids
        ]
        return audio_params


def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


def test_convtasnet():
    x = torch.randn(320)
    nnet = ConvTasNet()
    s = nnet(x,["hello","world"])
    print(str(check_parameters(nnet))+' Mb')
    print(s[1].shape)


if __name__ == "__main__":
    test_convtasnet()