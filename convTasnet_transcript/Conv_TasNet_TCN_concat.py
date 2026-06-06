import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

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

class TextConcatFuse(nn.Module):
    """
    在通道维拼接 text embedding
    audio: (B, C, T)
    text:  (B, D)
    """
    def __init__(self, audio_channels, text_dim):
        super().__init__()
        self.proj = nn.Conv1d(audio_channels + text_dim,
                              audio_channels,
                              kernel_size=1,
                              bias=True)

    def forward(self, x, text_embed):
        B, C, T = x.shape
        text = text_embed.unsqueeze(-1).expand(-1, -1, T)  # (B, D, T)
        x = torch.cat([x, text], dim=1)                    # (B, C+D, T)
        x = self.proj(x)                                   # (B, C, T)
        return x

class TextEncoder(nn.Module):
    '''
    集成到网络中的文本编码器
    支持BERT等预训练模型，并可微调
    '''
    def __init__(self, model_name="bert-base-uncased", trainable=True, pooling_method="mean"):
        super(TextEncoder, self).__init__()
        self.model_name = model_name
        self.pooling_method = pooling_method
        
        # 加载预训练模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        
        self.embedding_dim = self.bert_model.config.hidden_size
        
    def forward(self, text):
        """
        输入: text_list - 文本字符串列表
        输出: text_embed - (B, D_text) 文本embedding
        """
        # Tokenize文本
        inputs = self.tokenizer(text, return_tensors="pt", 
                              padding=True, truncation=True, max_length=512)
        
        # 将输入移动到与模型相同的设备
        device = next(self.bert_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 获取BERT输出
        outputs = self.bert_model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # (B, L, D)
        
        # 池化得到句子表示
        if self.pooling_method == "mean":
            # 均值池化，忽略padding部分
            attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (B, L, 1)
            sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            text_embed = sum_embeddings / sum_mask
        elif self.pooling_method == "cls":
            # 使用[CLS]标记
            text_embed = last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"不支持的池化方法: {self.pooling_method}")
            
        return text_embed

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
    if norm not in ['gln', 'cln', 'bn']:
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

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
             kernel_size, norm, causal, text_dim):
        super().__init__()
        self.repeats = nn.ModuleList()
        self.gates = nn.ParameterList()
        
        for _ in range(R):
            self.repeats.append(nn.ModuleDict({
                "fuse": TextConcatFuse(in_channels, text_dim),
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
            self.gates.append(nn.Parameter(torch.tensor(0.5)))
    
    def forward(self, x, text_feat):
        """
        x: [N, B, T]
        text_feat: [N, T_text, D]
        """
        for repeat in self.repeats:
            # ① 通道层面文本拼接（一次）
            x = repeat["fuse"](x, text_feat)
            # ② 纯声学 TCN
            for block in repeat["blocks"]:
                x = block(x)
        return x 
    

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
                 text_model_name="bert-base-uncased",  # 新增：文本模型名称
                 pooling_method="CLS",         # 新增：池化方法
                #  attn_dim=256
                 ):                            
        super(ConvTasNet, self).__init__()
        # n x 1 x T => n x N x T
        self.encoder = Conv1D(1, N, L, stride=L // 2, padding=0)
        # n x N x T  Layer Normalization of Separation
        self.LayerN_S = select_norm('cln', N)
        # n x B x T  Conv 1 x 1 of  Separation
        self.BottleN_S = Conv1D(N, B, 1)
        
         # 新增：语义注入相关模块
        self.text_encoder = TextEncoder(
            model_name=text_model_name, 
            pooling_method=pooling_method
        )
        text_dim = self.text_encoder.embedding_dim
        
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
            text_dim=text_dim,
            # attn_dim=attn_dim
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

    

    def forward(self, x, text_list=None):
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
        
        # 新增：语义注入
        # 文本编码
        text_embed = self.text_encoder(text_list)
        # n x B x L => n x B x L
        e = self.separation(e,text_embed)
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
    
    def get_text_encoder_parameters(self):
        """获取文本编码器的参数，用于单独设置优化器"""
        return self.text_encoder.parameters() if self.use_semantic_injection else []
    
    def get_audio_encoder_parameters(self):
        """获取音频编码器的参数"""
        audio_params = []
        for name, param in self.named_parameters():
            if 'text_encoder' not in name:
                audio_params.append(param)
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
    s = nnet(x)
    print(str(check_parameters(nnet))+' Mb')
    print(s[1].shape)


if __name__ == "__main__":
    test_convtasnet()