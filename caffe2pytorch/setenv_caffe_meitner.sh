#!/bin/bash

export CAFFE_BASEDIR=/home/twongj01/software/caffe/build/install
export PATH=${CAFFE_BASEDIR}/bin:${PATH}
export LD_LIBRARY_PATH=${CAFFE_BASEDIR}/lib:${PATH}
export PYTHONPATH=${CAFFE_BASEDIR}/python:${PYTHONPATH}
