from __future__ import print_function
import os,sys
import numpy as np
import torch

if '../' not in sys.path:
    sys.path.append('../')
from models.caffe_uresnet import UResNet

#checkpointfile = sys.argv[1]
checkpointfile = "checkpoint.60000th.tar"

# load model
model = UResNet(num_classes=4, input_channels=1, inplanes=16, showsizes=False)
#print(model)

# load statedict
checkpoint = torch.load( checkpointfile )
model.load_state_dict(checkpoint["state_dict"])

# fake input
fakedata_t = torch.rand( 1, 1, 512, 512 )
print(fakedata_t)

# trace the net
traced_script_module = torch.jit.trace(model, fakedata_t)
traced_script_module.save( "caffe_uresnet.pt" )


