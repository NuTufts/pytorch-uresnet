This folder contains specialized scripts that we used to
convert the network weights in caffe format (stored in hdf5) into
an archive of pytorch weights (stored as a pickled dictionary of tensors).

The scripts are fairly specific to this particular network.

The generated weights are to be used with the UBSSNet model in ubssnet.py in the models folder.

A script to download the weights from the tufts cluster can be found in the weights folder.