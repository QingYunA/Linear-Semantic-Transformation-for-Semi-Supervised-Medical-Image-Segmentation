from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetBlock(nn.Module):
    def __init__(self, in_planes, out_planes) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out

class Extractor(nn.Module):
    def __init__(self,dims,size,target_class=2):
        super().__init__()
        self.dims=dims
        self.size=size
        self.kernel_size_list = [8,4,2,1]
        self.extractor = nn.ModuleList([UnetBlock(dim,target_class) for dim in self.dims])
        self.upconv = nn.ModuleList([nn.ConvTranspose3d(target_class,target_class,kernel_size=size,stride=size) for dim,size in zip(self.dims,self.kernel_size_list)])
        self.layer_weigth = nn.Parameter(torch.ones((len(self.dims))))
        self.w_paramter = nn.ParameterList([torch.ones(4,1,64,64,64) for i in range(4)])
        self.b_paramter = nn.ParameterList([torch.ones(4,1,64,64,64) for i in range(4)])

        # self.linear_paramter = nn.Parameter(torch.ones(4,1,64,64,64))
        
    def cal_layers_weight(self):
        weight = torch.sigmoid(self.layer_weigth)
        return weight/weight.sum()
        
    def forward(self,features):
        outputs = []
        for i,fm in enumerate(features):
            x = self.extractor[i](fm.feature_map)
            x = self.upconv[i](x)
            x = self.w_paramter[i]*x + self.b_paramter[i]
            outputs.append(x)
            
        # upsamples = [F.interpolate(i,self.size,mode="trilinear",align_corners=True) for i in outputs]
        weight =self.cal_layers_weight()
        weight =torch.sigmoid(self.layer_weigth)
        weight = weight/weight.sum()
        # 
        final = sum([r * w for r,w in zip(outputs,weight)])
        outputs.append(final)
        return outputs
        

            
if __name__ == '__main__':
    import torch

    # a = [
    #     torch.randn(4, 1, 64, 64, 64),
    #     torch.randn(4, 1, 64, 64, 64),
    #     torch.randn(4, 1, 64, 64, 64)
    # ]
    # a = torch.randn(4,4,64,64,64)
    # conv = nn.Conv3d(in_channels=4,out_channels=1,kernel_size=64)
    # result =conv(a)
    # print(result.shape)
    # print(result.shape)
    # # pool =nn.AdaptiveAvgPool3d(output_size=(4,1,64,64,64))
    # total = 0
    # for param in pool.parameters():
    #     total+=param.numel()
    #
    # print(total)
    #
    a = torch.randn(4,4,64,64,64)