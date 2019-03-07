import os,sys
from collections import OrderedDict

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2

import torch

protofile="test.prototxt"
plane = int(sys.argv[1])
if plane not in [0,1,2]:
    print("unrecognized plane id: %d"%(plane))
    sys.exit(-1)
weightfile=["segmentation_pixelwise_ikey_plane0_iter_75500.caffemodel",
            "segmentation_pixelwise_ikey_plane1_iter_65500.caffemodel",
            "segmentation_pixelwise_ikey_plane2_iter_68000.caffemodel"]

outfile = "mcc8_caffe_ubssnet_plane%d.tar"%(plane)
            
caffe.set_mode_cpu()
net = caffe.Net(protofile, weightfile[plane], caffe.TEST)

layer_names = net.layer_dict.keys()

# layer parsing
# convx: convolution
#  weights: (channels,1,kernel_width,kernel_height)
#  bias:    (channels,)
# bn_x: batchnorm
#  mean:     (channels,)
#  variance: (channels,)
#  scale_factor: (1,)
# scale_x: scale layer
#  scale: (channels,)
#  bias:  (channels,)
# deconv:
#  weights: (in_channels,1,kernel_weight,kernel_height)
#  bias:    (out_channels,)

params = OrderedDict()
                                 
for item in net.params.items():
    name, layer = item
    layertype = net.layer_dict[name].type
    print('convert layer: ' + name + " :: type=" + layertype)

    if layertype=="Convolution":
        params[name+"_w"] = net.params[name][0].data
        if len(net.params[name])==2:
            params[name+"_b"] = net.params[name][1].data
    elif layertype=="BatchNorm":
        params[name+"_mean"] = net.params[name][0].data
        params[name+"_var"] = net.params[name][1].data
        params[name+"_scale"] = net.params[name][2].data
    elif layertype=="Scale":
        params[name+"_scale"] = net.params[name][0].data
        params[name+"_b"]     = net.params[name][1].data
    elif layertype=="Deconvolution":
        params[name+"_w"] = net.params[name][0].data
        params[name+"_b"] = net.params[name][1].data   
        

tensors = OrderedDict()
for name,data in params.items():
    print name,data.shape
    tensors[name] = torch.from_numpy(data)

torch.save(tensors,outfile)
