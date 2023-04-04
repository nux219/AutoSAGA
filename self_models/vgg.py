#############################
#   @author: Nitin Rathi    #
#############################
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import numpy as np


cfg = {
    'VGG5' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 512, 'A', 512, 512],
    # 'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512, 'A'],
    # 'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]
}


class VGG(nn.Module):
    def __init__(self, vgg_name='VGG16', labels=10, dataset = 'CIFAR10', kernel_size=3, dropout=0.2, mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]):
    # def __init__(self, vgg_name='VGG16', labels=10, dataset = 'CIFAR10', kernel_size=3, dropout=0.2):
        super(VGG, self).__init__()

        self.dataset        = dataset
        self.kernel_size    = kernel_size
        self.dropout        = dropout
        self.features       = self._make_layers(cfg[vgg_name])
        if vgg_name == 'VGG5' and dataset!= 'MNIST':
            self.classifier = nn.Sequential(
                            # nn.Linear(512*4*4, 4096, bias=False),
                            nn.Linear(512*4*4, 1024, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            # nn.Linear(4096, 4096, bias=False),
                            nn.Linear(1024, 1024, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            # nn.Linear(4096, labels, bias=False)
                            nn.Linear(1024, labels, bias=False)
                            )
        elif (vgg_name!='VGG5'and vgg_name!='VGG16') and dataset!='MNIST':
            self.classifier = nn.Sequential(
                            # nn.Linear(512*2*2, 4096, bias=False),
                            nn.Linear(512*2*2*4, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=False)
                            )
        if vgg_name == 'VGG5' and dataset == 'MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(128*7*7, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=False)
                            )
        elif vgg_name!='VGG5' and dataset =='MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(512*1*1, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=False)
                            )
        elif vgg_name=='VGG16' and dataset!='MNIST':
            self.classifier = nn.Sequential(
                            # nn.Linear(512, 512, bias=False),
                            # nn.ReLU(inplace=True),
                            # nn.Dropout(self.dropout),
                            # nn.Linear(512, 256, bias=False),
                            # nn.ReLU(inplace=True),
                            # nn.Dropout(self.dropout),
                            nn.Linear(512, labels, bias=False)
                            )
        self._initialize_weights2()

        tmpmean = torch.Tensor(np.array(mean)[:, np.newaxis, np.newaxis])
        self.matrixMean = tmpmean.expand(3, 32, 32).cuda()
        tmpstd = torch.Tensor(np.array(std)[:, np.newaxis, np.newaxis])
        self.matrixStd = tmpstd.expand(3, 32, 32).cuda()

        # # Added by K to handle normalization
        # self.matrixMean = torch.ones(3, 32, 32)
        # self.matrixMean[0] = self.matrixMean[0] * 0.4914
        # self.matrixMean[1] = self.matrixMean[1] * 0.4822
        # self.matrixMean[2] = self.matrixMean[2] * 0.4465
        #
        # self.matrixStd = torch.ones(3, 32, 32)
        # self.matrixStd[0] = self.matrixStd[0] * 0.2023
        # self.matrixStd[1] = self.matrixStd[1] * 0.1994
        # self.matrixStd[2] = self.matrixStd[2] * 0.2010
        # self.matrixMean = self.matrixMean.cuda()
        # self.matrixStd = self.matrixStd.cuda()

    def forward(self, x):
        #Normalization added by K
        x = (x-self.matrixMean)/self.matrixStd #Normalize first
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _initialize_weights2(self):
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def _make_layers(self, cfg):
        layers = []

        if self.dataset == 'MNIST':
            in_channels = 1
        else:
            in_channels = 3
        
        for x in cfg:
            stride = 1
            
            if x == 'A':
                layers.pop()
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=False),
                           nn.ReLU(inplace=True)
                           ]
                layers += [nn.Dropout(self.dropout)]           
                in_channels = x

        
        return nn.Sequential(*layers)

def test():
    for a in cfg.keys():
        if a=='VGG5':
            continue
        net = VGG(a)
        x = torch.randn(2,3,32,32)
        y = net(x)
        print(y.size())
    # For VGG5 change the linear layer in self. classifier from '512*2*2' to '512*4*4'    
    # net = VGG('VGG5')
    # x = torch.randn(2,3,32,32)
    # y = net(x)
    # print(y.size())
if __name__ == '__main__':
    test()
