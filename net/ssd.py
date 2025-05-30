import torch
import  torch.nn as nn
import cv2
from sympy.abc import o
from torch.hub import load_state_dict_from_url
from torchvision.models.mobilenetv2 import InvertedResidual, mobilenet_v2
import torch.nn.functional as f

base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',512, 512, 512]

#https://blog.csdn.net/weixin_44791964/article/details/104981486   SSD(Single Shout MultiBox Detector) 写的不错的一个博客
def vgg (pretrained = False):
    in_channel= 3
    layers = []

    for v in base:
        #v为outChannel
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2, stride= 2)]
        if v== 'C':
            layers += [nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)]
        else:
            conv2d = nn.Conv2d(in_channel, v,  kernel_size= 3, padding = 1)
            layers += [conv2d, nn.ReLU(True)]
            #更新每一次的v（输出）为下一次的in_channel(输入)
            in_channel = v

    #一轮循环完了，执行下面的操作，
    #19 19 512 ->19 19 512
    pool5 = nn.MaxPool2d(padding = 1, kernel_size = 3)
    #19 19 512 ->19 19 1024
    conv6 = nn.Conv2d(512,1024,kernel_size = 3, dilation= 6)
    #19 19 1024 ->19 19 1024
    conv7 = nn.Conv2d(1024,1024, kernel_size= 1)
    layers += [pool5, conv6, nn.ReLU(True), conv7, nn.ReLU(True)]

    model = nn.ModuleList(layers)

    if pretrained:
        #如果是预训练模型
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",
                                              model_dir="./model_data")
        state_dict = {k.replace('features.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    return model

def add_extra(in_channel,backbone_name):
    layers = []
    if backbone_name =='vgg16':
     #bolck6
     #19 19 1024 -> 19 19 256->10 10 512
     layers += nn.Conv2d(in_channel, 256, kernel_size = 1, stride= 1)
     layers += nn.Conv2d(256,512,kernel_size = 3, stride=2, padding=1)

     #block7
     #10 10 512 -> 10 10 128 ->5 5 256
     layers += nn.Conv2d(512,128, kernel_size= 1, stride = 1)
     layers += nn.Conv2d(128, 256,kernel_size= 3, stride=2, padding =  1)

     #block8
     #5 5 256 -> 5 5 128 -> 3 3 256
     layers += nn.Conv2d(256, 128, kernel_size=1, stride = 1)
     layers += nn.Conv2d(128, 256, kernel_size=3, stride= 1)

     #block9
     # 3 3 256 ->  3 3 128-> 1 1 256
     layers += nn.Conv2d(256, 128, kernel_size=1, stride=1)
     layers += nn.Conv2d(128, 256, kernel_size=3, stride=1)

    else :
     layers += [InvertedResidual(in_channel, 512, stride=2, expand_ratio=0.2)]
     layers += [InvertedResidual(512, 256, stride=2, expand_ratio=0.25)]
     layers += [InvertedResidual(256, 256, stride=2, expand_ratio=0.5)]
     layers += [InvertedResidual(256, 64, stride=2, expand_ratio=0.25)]


     return nn.ModuleList(layers)



class ssd300(nn.Module):
    def __init__(self, backbone_name, num_class, pretrained = False):
        super().__init__(self)
        loc_layers = []
        conf_layers = []
        if backbone_name == 'vgg16':
            self.vgg = vgg(pretrained)
            self.extra_vgg = add_extra(1024, pretrained)
            #ssd的先验框【4 6 6 6 4 4】
            mbox = [4, 6, 6, 6, 4, 4]
            backbone_source = [21, -2]
            #
            # ---------------------------------------------------#
            #   在add_vgg获得的特征层里
            #   第21层和-2层可以用来进行回归预测和分类预测。
            #   分别是conv4-3(38,38,512)和conv7(19,19,1024)的输出
            # ---------------------------------------------------#

            for k, v in enumerate(backbone_source):
                loc_layers += [nn.Conv2d(self.vgg(v).out_channels,mbox[k] * 4,kernel_size = 3, padding = 1)]
                conf_layers += [nn.Conv2d(self.vgg(v).out_channels,mbox[k] * num_class, kernel_size = 3, padding = 1)]

            # -------------------------------------------------------------#
            #   在add_extras获得的特征层里
            #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
            #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
            # -------------------------------------------------------------#

            for k, v in enumerate(self.extra_vgg[1::2],2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size = 3, padding = 1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_class, kernel_size = 3, padding = 1)]

        else:
            self.mobilenet = mobilenet_v2(pretrained).features
            self.extras = add_extra(1280, backbone_name)
            self.L2Norm = self.L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]
            backbone_source = [13, -1]
            for k, v in enumerate(backbone_source):
                loc_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [
                    nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_class, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_class, kernel_size=3, padding=1)]

        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)
        self.backbone_name  = backbone_name


    def forward(self, x):
        #x 是 300 300 3
        loc = list()
        conf = list()
        source = list()
        if self.backbone_name == 'vgg16':
            for k in range(23):
                x = self.vgg[k](x)
        else:
            for k in range(14):
                x = self.mobilenet[k](x)


        #conv4_3 的内容需要进行L2标准化
        s = self.L2Norm(x)
        source.append(s)

        # conv7_2 的内容
        if self.backbone_name == 'vgg16':
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)
        else:
            for k in range(14, len(self.mobilenet)):
                x = self.mobilenet[k](x)

        source.append(x)

        # -------------------------------------------------------------#
        #   在add_extras获得的特征层里
        #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        # -------------------------------------------------------------#

        for k, v in enumerate(self.extras):
            x = f.relu(v(x), inplace = True)
            if self.backbone_name == 'vgg16':
                if k % 2 == 1:
                    #if k % 2 == 1: 获取第1层、第3层、第5层、第7层
                    source.append(x)
            else:
                    source.append(x)

        for (x, l , c) in zip(source, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        #进行reshape堆叠
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0),-1 ) for o in conf], 1)

        out= (
            loc.view(loc.size(0), -1 , 4),
            conf.view(conf.size(0), -1,self.num_class),
        )
        return out



