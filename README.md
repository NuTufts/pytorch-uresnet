# pytorch-uresnet

PyTorch Implementation of U-ResNet used for track/shower pixel-labeling

## Dependencies

* `ROOT`: data analysis framework. Defines file format, provides python bindings for our code
* `LArCV`: either version 1 and 2
* `pytorch`: network implementation
* `tensorboardX`: interface to log data that can be plotted with Tensorboard
* `tensorflow-tensorboard`: (from tensorflow) for plotting loss/accuracy
* `jupyter notebook`: for testing code and inspecting images

### Known working configuration

  * ubuntu 16.10, ROOT 6, python 2.7.12, tensorflow-tensorboard (from tensorflow 1.4.1), cuda 8.0

## Folders

* `models`: contains models
* `notebooks`: jupyter notebooks to view data
* `train_scripts`: scripts for training
* `tufts_scripts`: scripts for running on the Tufts research cluster
* `prod_scripts`: scripts for setting up production
* `meitner_scripts`: scripts for running on meitner machine
* `arxiv`: cruft

## Models

### Caffe UResNet

A reimplementation of UResNet used for MCC8. Results on data published at X.
