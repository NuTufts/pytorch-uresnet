import os,sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import math
import numpy as np

#data.init(-1,24,24*8,16)
#dimension = 3
#reps = 1 #Conv block repetition factor
#m = 32   #Unet number of features
#nPlanes = [m, 2*m, 3*m, 4*m, 5*m] #UNet number of features per level

class SparseUResNet(nn.Module):
    def __init__(self, inputshape, reps, nfeatures, nplanes, noutput_classes):
        nn.Module.__init__(self)

        # set parameters
        self.dimensions = 2 # not playing with 3D for now

        # input shape: LongTensor, tuple, or list. Handled by InputLayer
        # size of each spatial dimesion
        self.inputshape = inputshape
        if len(self.inputshape)!=self.dimensions:
            raise ValueError("expected inputshape to contain size of 2 dimensions only. given %d values"%(len(self.inputshape)))
        
        # mode variable: how to deal with repeated data
        self.mode = 0

        # nfeatures
        self.nfeatures = nfeatures

        # plane structure
        self.nPlanes = [ self.nfeatures*(n+1) for n in xrange(nplanes) ]

        # repetitions (per plane)
        self.reps = reps

        # output classes
        self.noutput_classes = noutput_classes
        
        # model:
        # input
        # stem
        # unet
        # linear to nclasses
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(self.dimensions, self.inputshape, mode=self.mode)).add(
            scn.SubmanifoldConvolution(self.dimensions, 1, self.nfeatures, 3, False)).add(
            scn.UNet(self.dimensions, self.reps, self.nPlanes, residual_blocks=True, downsample=[2,2])).add(
            scn.BatchNormReLU(self.nfeatures)).add(
            scn.SubmanifoldConvolution(self.dimensions,self.nfeatures,self.noutput_classes,1,False)).add(
            #scn.SparseToDense(self.dimensions,self.noutput_classes))
            scn.OutputLayer(self.dimensions))
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, coord_t, feature_t, batchsize ):
        x = ( coord_t, feature_t, batchsize )
        #print "model input: ",type(x[0]),x[0].shape,x[1].shape
        x=self.sparseModel(x)
        #print "model output: ",type(x),x.shape
        x=self.softmax(x)
        return x
