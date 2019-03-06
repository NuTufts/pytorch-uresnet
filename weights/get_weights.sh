#!/bin/bash

username=twongj01

rsync -av --progress ${username}@xfer.cluster.tufts.edu:/cluster/tufts/wongjiradlab/larbys/ssnet_models/v1_pytorch_extracted/saved_caffe_weights_plane*.tar .
