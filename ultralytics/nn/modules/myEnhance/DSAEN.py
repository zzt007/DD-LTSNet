import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import clever_format, profile

# 处理导入路径问题
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics.nn.modules.myEnhance.HVI_transform import RGB_HVI
from ultralytics.nn.modules.myEnhance.transformer_utils import *
from ultralytics.nn.modules.conv import Conv

class dual_stream_net(nn.Module):
    def __init__(self, channel=64, SK_size=3):
        super(dual_stream_net, self).__init__()
        self.spaCNNAtt = SpatialCNNAtt(in_channels=channel, SK_size=SK_size)
        self.ChaAtt = ChannelAtt(in_channels=channel)
        # self.conv1 = nn.Conv2d(channel*2, channel, kernel_size=3, padding=1, stride=1)
        self.conv1 = nn.Conv2d(3, channel, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channel)
        
        self.HVE_block0 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, channel, kernel_size=3, stride=1, padding=0,bias=False)
        )
        
        self.IE_block0 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, channel, kernel_size=3, stride=1, padding=0,bias=False)
        )
        
        
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(channel, 2, kernel_size=3, stride=1, padding=0),
        )
        
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=0),
            )
        
        self.trans = RGB_HVI()
        
    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        hv = self.HVE_block0(hvi)
        i = self.IE_block0(i)
        
        f = self.ChaAtt(hv) * self.spaCNNAtt(i)
        i_out = f * i
        hv_out = f * hv
        
        i_out = self.ID_block0(i_out)
        hv_out = self.HVD_block0(hv_out)
        
        z = torch.cat([i_out, hv_out], dim=1) + hvi # return hvi color space
        # output_rgb = self.trans.PHVIT(z) # return rgb color space
        output4yolo = F.silu(self.bn1(self.conv1(z)))
        return output4yolo
        # return F.silu(self.bn1(self.conv1(output_rgb))) 

    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi

class ChannelAtt(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.silu1 = nn.SiLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.silu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.silu1(self.fc1(self.max_pool(x))))  
        return self.sigmoid(avg_out + max_out)
    
class SpaCNN(nn.Module):
    def __init__(self, in_channels, SK_size=3, strides=1):
        super(SpaCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=SK_size, padding=int(SK_size//2), stride=strides)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=SK_size, padding=int(SK_size//2), stride=strides)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        Y = F.silu(self.bn1(self.conv1(x)))
        return F.silu(self.bn2(self.conv2(Y)))
    
class SpatialCNNAtt(nn.Module):
    def __init__(self, in_channels=64, SK_size=3, kernel_size=3):
        super(SpatialCNNAtt, self).__init__()
        self.scnn = SpaCNN(in_channels, SK_size, strides=1)
        self.conv1 = nn.Conv2d(2,1,kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.scnn(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.sigmoid(self.conv1(x))
        return x
    
if __name__ == '__main__':
    hv_input = torch.randn(1, 3, 640, 640)  # 生成随机输入张量1
    i_input = torch.randn(1, 1, 640, 640)  # 生成随机输入张量2
    # 初始化CVIM模块并设定通道维度
    model = dual_stream_net(64)
    output = model(i_input, hv_input)  # 计算输出

    print("Input size:", i_input.size(), hv_input.shape)
    print("Output size:", output.size())
    # 使用thop计算参数量和计算量
    flops, params = profile(model, inputs=(i_input, hv_input))
    flops, params = clever_format([flops, params], "%.3f")
    
    # 输出结果
    print(f"参数量: {params}")
    print(f"计算量: {flops}")