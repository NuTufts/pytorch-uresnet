import os,sys

import torch.nn as nn
import torch as torch

class DoubleResLayer(nn.Module):
    def __init__(self,layernum,inputchs,outputchs,stride,relu_inplace=True):
        super(DoubleResLayer,self).__init__()
        self.layernum = layernum

        # convNa1
        self.resNa_branch2a = nn.Conv2D(inputchs,outputchs,
                                        kernel_size=3,stride=stride,padding=1,bias=False)
        self.bnNa_branch2a  = nn.BatchNorm2d(outputchs, eps=1e-05)
        self.relua_branch2a = nn.ReLU(inplace=relu_inplace)

        # convNa2
        self.resNa_branch2b = nn.Conv2D(outputchs,outputchs,
                                        kernel_size=3,stride=1,padding=1,bias=False)
        self.bnNa_branch2b  = nn.BatchNorm2d(outputchs, eps=1e-05)
        self.relua_branch2b = nn.ReLU(inplace=relu_inplace)

        # skip1
        self.resNa_branch1  = nn.Conv2D(inputchs,outputchs,
                                        kernel_size=1,stride=stride,padding=0,bias=False)
        self.bnNa_branch1   = nn.BatchNorm2d(outputchs, eps=1e-05)

        # concat relu
        self.relua_concat   = nn.ReLU(inplace=relu_inplace)

        # convNb1
        self.resNb_branch2a = nn.Conv2D(outputchs,outputchs,
                                        kernel_size=3,stride=1,padding=1,bias=False)
        self.bnNb_branch2a  = nn.BatchNorm2d(outputchs, eps=1e-05)
        self.relub_branch2a = nn.ReLU(inplace=relu_inplace)

        # convNb2
        self.resNb_branch2b = nn.Conv2D(outputchs,outputchs,
                                        kernel_size=3,stride=1,padding=1,bias=False)
        self.bnNb_branch2b  = nn.BatchNorm2d(outputchs, eps=1e-05)
        self.relub_branch2b = nn.ReLU(inplace=relu_inplace)

        # concat relu
        self.relub_concat   = nn.ReLU(inplace=relu_inplace)

    def forward(self,x):

        yskip = self.resNa_branch1(x)
        yskip = self.bnNa_branch1(yskip)
        
        y     = self.resNa_branch2a(x)
        y     = self.bnNa_branch2a(y)
        y     = self.relua_branch2a(y)
        y     = self.resNa_branch2b(y)
        y     = self.bnNa_branch2b(y)
        y     = self.relua_branch2b(y)

        y     = y+yskip
        y     = self.relua_concat(y)

        y2    = self.resNb_branch2a(y)
        y2    = self.bnNb_branch2a(y2)
        y2    = self.relub_branch2a(y2)
        y2    = self.resNb_branch2b(y2)
        y2    = self.bnNb_branch2b(y2)
        y2    = self.relub_branch2b(y2)


        y = y+y2
        y = self.relub_concat(y)

        return y

        


class ubSSNet(nn.Module):
    """ the model used in the paper and employed in mcc8 analyses """
    
    def __init__(self, weight_file ):
        super(UResNet, self).__init__()

        relu_inplace = True
        
        # input stem
        self.conv0    = nn.Conv2D( 1, 16, kernel_size=7, stride=1, padding=3, bias=True )
        self.bn_conv0 = nn.BatchNorm2d(16, eps=1e-05)
        # batch-norms include scale
        self.bn_relu  = nn.ReLU(inplace=relu_inplace)

        # pool0
        self.pool0    = nn.MaxPool2d(3,stride=2)
        
        # encoder layers
        self.encode1 = DoubleResLayer(1, 16, 32,1,relu_inplace=relu_inplace)
        self.encode2 = DoubleResLayer(2, 32, 64,2,relu_inplace=relu_inplace)
        self.encode3 = DoubleResLayer(3, 64,128,2,relu_inplace=relu_inplace)
        self.encode4 = DoubleResLayer(4,128,256,2,relu_inplace=relu_inplace)
        self.encode5 = DoubleResLayer(5,256,512,2,relu_inplace=relu_inplace)

        # decoder layers
        self.deconv0 = nn.ConvTranspose2D(512,256,4,stride=2,padding=1,group=256,bias=True)
        self.decode6 = DoubleResLayer(6,512,256,1,relu_inplace=relu_inplace)

        self.deconv1 = nn.ConvTranspose2D(256,128,4,stride=2,padding=1,group=128,bias=True)
        self.decode7 = DoubleResLayer(7,256,128,1,relu_inplace=relu_inplace)

        self.deconv2 = nn.ConvTranspose2D(128,64,4,stride=2,padding=1,group=64,bias=True)
        self.decode8 = DoubleResLayer(8,128,64,1,relu_inplace=relu_inplace)

        self.deconv3 = nn.ConvTranspose2D(64,32,4,stride=2,padding=1,group=32,bias=True)
        self.decode9 = DoubleResLayer(9,64,32,1,relu_inplace=relu_inplace)

        self.deconv4  = nn.ConvTranspose2D(32,16,4,stride=2,padding=1,group=16,bias=True)

        # stem: conv10
        self.conv10    = nn.Conv2D( 32, 16, kernel_size=7, stride=1, padding=3, bias=True )
        self.bn_conv10 = nn.BatchNorm2d(16, eps=1e-05)
        self.relu10    = nn.ReLU(inplace=relu_inplace)

        # output layer: conv11
        self.conv11    = nn.Conv2D( 16, 3, kernel_size=7, stride=1, padding=3, bias=True )
        self.bn_conv11 = nn.BatchNorm2d(3, eps=1e-05)  # ????
        self.relu11    = nn.ReLU(inplace=relu_inplace) # ????

        # softmax
        self.softmax   = nn.SoftMax(dim=1)

    def forward(self,x):

        x = self.conv0(x)
        x = self.bn_conv0(x)
        x = self.bn_relu(x)
        x0 = self.pool0(x)

        x1 = self.encode1(x0)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)
        x5 = self.encode5(x4)

        x6  = self.decode6(self.deconv0(x5)+x4)
        x7  = self.decode7(self.deconv1(x6)+x3)
        x8  = self.decode8(self.deconv2(x7)+x2)
        x9  = self.decode9(self.deconv3(x8)+x1)
        x10 = self.conv10( self.deconv4(x9)+x )
        x10 = self.bn_conv10(x10)
        x10 = self.relu10(x10)

        x   = self.conv11(x10)
        x   = self.bn_conv11(x)
        x   = self.relu11(x)

        x   = self.softmax(x)
        return x
        
        
    def load_weights(self,weightfile):
        """
        takes in a numpy file with label for each parameter and passes it to each layer.
        """
        
