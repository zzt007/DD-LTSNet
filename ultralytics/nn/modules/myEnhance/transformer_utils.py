import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class NormDownsample(nn.Module):
    def __init__(self,in_ch,out_ch,scale=0.5,use_norm=False):
        super(NormDownsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        # 使用SiLU代替PReLU，避免DDP中的梯度步幅问题
        self.activation = nn.SiLU()
        self.conv = nn.Conv2d(in_ch, out_ch,kernel_size=3,stride=1, padding=1, bias=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale)
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        x = self.activation(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x

class NormUpsample(nn.Module):
    def __init__(self, in_ch,out_ch,scale=2,use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        # 使用SiLU代替PReLU，避免DDP中的梯度步幅问题
        self.activation = nn.SiLU()
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1, padding=1, bias=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale)
        self.up = nn.Conv2d(out_ch*2,out_ch,kernel_size=1,stride=1, padding=0, bias=False)
    
    # 注意这里的前向传播输入有xy两个，因为还要接收skip connect
    def forward(self, x,y):
        x = self.conv(x)
        x = self.upsample(x)
        x = torch.cat([x, y],dim=1)
        x = self.up(x)
        x = self.activation(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x
 
