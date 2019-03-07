#!/bin/bash

username=twongj01

# MCC8 Caffe UBSSNet: used in the 2019 PRD pixel-labeling paper
rsync -av --progress ${username}@xfer.cluster.tufts.edu:/cluster/tufts/wongjiradlab/larbys/ssnet_models/v1_pytorch_extracted/mcc8_caffe_ubssnet_plane*.tar .

# MCC8 Caffe UBSSNet: same as above, but in original caffe formet (HDF5)
#rsync -av --progress ${username}@xfer.cluster.tufts.edu:/cluster/tufts/wongjiradlab/larbys/ssnet_models/v1/segmentation_pixelwise_ikey_plane*.caffemodel .
