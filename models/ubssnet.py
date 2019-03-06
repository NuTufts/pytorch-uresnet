import os,sys

import torch
import torch.nn as nn

import numpy as np


class DoubleResLayer(nn.Module):
    def __init__(self,layernum,inputchs,outputchs,kernel_size,stride,relu_inplace=True):
        super(DoubleResLayer,self).__init__()
        self.layernum = layernum

        same_pad_size = (kernel_size-1)/2
        
        # convNa1
        self.resNa_branch2a = nn.Conv2d(inputchs,outputchs,
                                        kernel_size=kernel_size,
                                        stride=stride,padding=same_pad_size,
                                        bias=False)
        self.bnNa_branch2a  = nn.BatchNorm2d(outputchs, eps=1e-05)
        self.relua_branch2a = nn.ReLU(inplace=relu_inplace)

        # convNa2
        self.resNa_branch2b = nn.Conv2d(outputchs,outputchs,
                                        kernel_size=kernel_size,stride=1,
                                        padding=same_pad_size,bias=False)
        self.bnNa_branch2b  = nn.BatchNorm2d(outputchs, eps=1e-05)
        self.relua_branch2b = nn.ReLU(inplace=relu_inplace)

        # skip1
        self.resNa_branch1  = nn.Conv2d(inputchs,outputchs,
                                        kernel_size=1,stride=stride,padding=0,bias=False)
        self.bnNa_branch1   = nn.BatchNorm2d(outputchs, eps=1e-05)

        # concat relu
        self.relua_eltwise  = nn.ReLU(inplace=relu_inplace)

        # convNb1
        self.resNb_branch2a = nn.Conv2d(outputchs,outputchs,
                                        kernel_size=kernel_size,stride=1,
                                        padding=same_pad_size,bias=False)
        self.bnNb_branch2a  = nn.BatchNorm2d(outputchs, eps=1e-05)
        self.relub_branch2a = nn.ReLU(inplace=relu_inplace)

        # convNb2
        self.resNb_branch2b = nn.Conv2d(outputchs,outputchs,
                                        kernel_size=kernel_size,stride=1,
                                        padding=same_pad_size,bias=False)
        self.bnNb_branch2b  = nn.BatchNorm2d(outputchs, eps=1e-05)
        self.relub_branch2b = nn.ReLU(inplace=relu_inplace)

        # concat relu
        self.relub_eltwise  = nn.ReLU(inplace=relu_inplace)

    def load_weights(self,pardict):

        prefix = "res%d"%(self.layernum)
        
        # convNa1
        self.resNa_branch2a.weight.data = pardict["res%da_branch2a_w"%(self.layernum)]
        #self.resNa_branch2a.bias.data   = pardict["res%da_branch2a_b"%(self.layernum)]
        scalefactor = pardict["bn%da_branch2a_scale"%(self.layernum)]
        self.bnNa_branch2a.running_mean.data = pardict["bn%da_branch2a_mean"%(self.layernum)]/scalefactor
        self.bnNa_branch2a.running_var.data  = pardict["bn%da_branch2a_var"%(self.layernum)]/scalefactor
        self.bnNa_branch2a.weight.data       = pardict["scale%da_branch2a_scale"%(self.layernum)]
        self.bnNa_branch2a.bias.data         = pardict["scale%da_branch2a_b"%(self.layernum)]
        
        # convNa2
        self.resNa_branch2b.weight.data = pardict["res%da_branch2b_w"%(self.layernum)]
        #self.resNa_branch2b.bias.data   = pardict["res%da_branch2a_b"%(self.layernum)]        
        scalefactor = pardict["bn%da_branch2b_scale"%(self.layernum)]        
        self.bnNa_branch2b.running_mean.data = pardict["bn%da_branch2b_mean"%(self.layernum)]/scalefactor
        self.bnNa_branch2b.running_mean.data = pardict["bn%da_branch2b_var"%(self.layernum)]/scalefactor
        self.bnNa_branch2b.weight.data       = pardict["scale%da_branch2b_scale"%(self.layernum)]
        self.bnNa_branch2b.bias.data         = pardict["scale%da_branch2b_b"%(self.layernum)]

        # skip1
        self.resNa_branch1.weight.data      = pardict["res%da_branch1_w"%(self.layernum)]
        #self.resNa_branch1.bias.data        = pardict["res%da_branch1_b"%(self.layernum)]
        scalefactor = pardict["bn%da_branch1_scale"%(self.layernum)]
        self.bnNa_branch1.running_mean.data = pardict["bn%da_branch1_mean"%(self.layernum)]/scalefactor
        self.bnNa_branch1.running_var.data  = pardict["bn%da_branch1_var"%(self.layernum)]/scalefactor
        self.bnNa_branch1.weight.data       = pardict["scale%da_branch1_scale"%(self.layernum)]
        self.bnNa_branch1.bias.data         = pardict["scale%da_branch1_b"%(self.layernum)]

        # concat relu

        # convNb1
        self.resNb_branch2a.weight.data     = pardict["res%db_branch2a_w"%(self.layernum)]
        #self.resNb_branch2a.bias.data       = pardict["res%db_branch2a_b"%(self.layernum)]
        scalefactor = pardict["bn%da_branch1_scale"%(self.layernum)]
        self.bnNb_branch2a.running_mean.data = pardict["bn%db_branch2a_mean"%(self.layernum)]/scalefactor
        self.bnNb_branch2a.running_var.data  = pardict["bn%db_branch2a_var"%(self.layernum)]/scalefactor
        self.bnNb_branch2a.weight.data       = pardict["scale%db_branch2a_scale"%(self.layernum)]
        self.bnNb_branch2a.bias.data         = pardict["scale%db_branch2a_b"%(self.layernum)]

        # convNb2
        self.resNb_branch2b.weight.data     = pardict["res%db_branch2b_w"%(self.layernum)]
        #self.resNb_branch2b.bias.data       = pardict["res%db_branch2b_b"%(self.layernum)]
        scalefactor = pardict["bn%da_branch1_scale"%(self.layernum)]
        self.bnNb_branch2b.running_mean.data = pardict["bn%db_branch2b_mean"%(self.layernum)]/scalefactor
        self.bnNb_branch2b.running_var.data  = pardict["bn%db_branch2b_var"%(self.layernum)]/scalefactor
        self.bnNb_branch2b.weight.data       = pardict["scale%db_branch2b_scale"%(self.layernum)]
        self.bnNb_branch2b.bias.data         = pardict["scale%db_branch2b_b"%(self.layernum)]

        

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
        y     = self.relua_eltwise(y)

        y2    = self.resNb_branch2a(y)
        y2    = self.bnNb_branch2a(y2)
        y2    = self.relub_branch2a(y2)
        
        y2    = self.resNb_branch2b(y2)
        y2    = self.bnNb_branch2b(y2)
        y2    = self.relub_branch2b(y2)


        y = y+y2
        y = self.relub_eltwise(y)

        return y

        


class ubSSNet(nn.Module):
    """ the model used in the paper and employed in mcc8 analyses """
    
    def __init__(self, weight_file ):
        super(ubSSNet, self).__init__()

        relu_inplace = False
        
        # input stem
        self.conv0    = nn.Conv2d( 1, 16, kernel_size=7, stride=1, padding=3, bias=True )
        self.bn_conv0 = nn.BatchNorm2d(16, eps=1e-05)
        # batch-norms include scale
        self.bn_relu  = nn.ReLU(inplace=relu_inplace)

        # pool0
        self.pool0    = nn.MaxPool2d(3,stride=2,padding=1)
        
        # encoder layers
        self.encode1 = DoubleResLayer(1, 16, 32,3,1,relu_inplace=relu_inplace)
        self.encode2 = DoubleResLayer(2, 32, 64,3,2,relu_inplace=relu_inplace)
        self.encode3 = DoubleResLayer(3, 64,128,3,2,relu_inplace=relu_inplace)
        self.encode4 = DoubleResLayer(4,128,256,3,2,relu_inplace=relu_inplace)
        self.encode5 = DoubleResLayer(5,256,512,3,2,relu_inplace=relu_inplace)

        # decoder layers
        self.deconv0 = nn.ConvTranspose2d(512,256,4,stride=2,padding=1,groups=256,bias=True)
        self.decode6 = DoubleResLayer(6,512,256,3,1,relu_inplace=relu_inplace)

        self.deconv1 = nn.ConvTranspose2d(256,128,4,stride=2,padding=1,groups=128,bias=True)
        self.decode7 = DoubleResLayer(7,256,128,3,1,relu_inplace=relu_inplace)

        self.deconv2 = nn.ConvTranspose2d(128,64,4,stride=2,padding=1,groups=64,bias=True)
        self.decode8 = DoubleResLayer(8,128,64,3,1,relu_inplace=relu_inplace)

        self.deconv3 = nn.ConvTranspose2d( 64,32,4,stride=2,padding=1,groups=32,bias=True)
        self.decode9 = DoubleResLayer(9,64,32,5,1,relu_inplace=relu_inplace)

        self.deconv4  = nn.ConvTranspose2d(32,16,4,stride=2,padding=1,groups=16,bias=True)

        # stem: conv10
        self.conv10    = nn.Conv2d( 32, 16, kernel_size=7, stride=1, padding=3, bias=True )
        self.bn_conv10 = nn.BatchNorm2d(16, eps=1e-05)
        self.relu10    = nn.ReLU(inplace=relu_inplace)

        # output layer: conv11
        self.conv11    = nn.Conv2d( 16, 3, kernel_size=7, stride=1, padding=3, bias=True )
        self.bn_conv11 = nn.BatchNorm2d(3, eps=1e-05)  # ????
        self.relu11    = nn.ReLU(inplace=relu_inplace) # ????

        # softmax
        self.softmax   = nn.Softmax(dim=1)

        self.load_weights(weight_file)

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

        x5 = self.deconv0(x5,output_size=x4.size())
        x5  = torch.cat([x4,x5],1)
        x6  = self.decode6(x5)

        x6  = self.deconv1(x6,output_size=x3.size())
        x6  = torch.cat([x3,x6],1)
        x7  = self.decode7(x6)

        x7  = self.deconv2(x7,output_size=x2.size())
        x7  = torch.cat([x2,x7],1)
        x8  = self.decode8(x7)

        x8  = self.deconv3(x8,output_size=x1.size())
        x8  = torch.cat([x1,x8],1)
        x9  = self.decode9(x8)

        x9  = self.deconv4(x9,output_size=x.size())
        x9  = torch.cat([x,x9],1)
        
        x10 = self.conv10(x9)
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
        params = torch.load(weightfile)

        # conv0
        self.conv0.weight.data = params["conv0_w"]
        self.conv0.bias.data   = params["conv0_b"]
        self.bn_conv0.running_mean.data = params["bn_conv0_mean"]/params["bn_conv0_scale"].item()
        self.bn_conv0.running_var.data  = params["bn_conv0_var"]/params["bn_conv0_scale"].item()
        self.bn_conv0.weight.data       = params["scale_conv0_scale"]
        self.bn_conv0.bias.data         = params["scale_conv0_b"]

        self.encode1.load_weights(params)
        self.encode2.load_weights(params)
        self.encode3.load_weights(params)
        self.encode4.load_weights(params)
        self.encode5.load_weights(params)
        self.decode6.load_weights(params)
        self.decode7.load_weights(params)
        self.decode8.load_weights(params)
        self.decode9.load_weights(params)
        
        self.deconv0.weight.data = params["deconv0_deconv_w"]
        self.deconv0.bias.data   = params["deconv0_deconv_b"]

        self.deconv1.weight.data = params["deconv1_deconv_w"]
        self.deconv1.bias.data   = params["deconv1_deconv_b"]

        self.deconv2.weight.data = params["deconv2_deconv_w"]
        self.deconv2.bias.data   = params["deconv2_deconv_b"]

        self.deconv3.weight.data = params["deconv3_deconv_w"]
        self.deconv3.bias.data   = params["deconv3_deconv_b"]

        self.deconv4.weight.data = params["deconv4_deconv_w"]
        self.deconv4.bias.data   = params["deconv4_deconv_b"]

        # conv10
        self.conv10.weight.data = params["conv10_w"]
        self.conv10.bias.data   = params["conv10_b"]
        self.bn_conv10.running_mean.data = params["bn_conv10_mean"]/params["bn_conv10_scale"].item()
        self.bn_conv10.running_var.data  = params["bn_conv10_var"]/params["bn_conv10_scale"].item()
        self.bn_conv10.weight.data       = params["scale_conv10_scale"]
        self.bn_conv10.bias.data         = params["scale_conv10_b"]

        # conv11
        self.conv11.weight.data = params["conv11_w"]
        self.conv11.bias.data   = params["conv11_b"]
        self.bn_conv11.running_mean.data = params["bn_conv11_mean"]/params["bn_conv11_scale"].item()
        self.bn_conv11.running_var.data  = params["bn_conv11_var"]/params["bn_conv11_scale"].item()
        self.bn_conv11.weight.data       = params["scale_conv11_scale"]
        self.bn_conv11.bias.data         = params["scale_conv11_b"]
        

if __name__ == "__main__":

    weightfile = "../weights/saved_caffe_weights.tar"
    device = torch.device("cuda:0")
    #device = torch.device("cpu")    
    model = ubSSNet(weightfile).to(device).eval()

    print model
    test = np.ones( (1,1,512,512), dtype=np.float32 )
    test_t = torch.from_numpy(test).to(device)    

    out_t = model(test_t)
    print "output shape: ",out_t.shape
