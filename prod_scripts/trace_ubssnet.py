from __future__ import print_function
import os,sys
import numpy as np
import torch

if '../' not in sys.path:
    sys.path.append('../')
from models.ubssnet import ubSSNet

weight_dir="../weights/"
weight_files = {0:weight_dir+"/mcc8_caffe_ubssnet_plane0.tar",
                1:weight_dir+"/mcc8_caffe_ubssnet_plane1.tar",
                2:weight_dir+"/mcc8_caffe_ubssnet_plane2.tar"}


for plane in [0,1,2]:
    # load model
    model = ubSSNet(weight_files[plane])
    if plane==0:
        print(model)

    # fake input
    fakebatch_t  = torch.zeros( (1,1,512,512), dtype=torch.float32 )

    # trace the net
    print("Trace Plane [%d]"%(plane))
    traced_script_module = torch.jit.trace( model, (fakebatch_t,) )
    traced_script_module.save( "mcc8_caffe_ubssnet_plane%d.pytorchscript"%(plane) )


