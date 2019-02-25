import os,sys,time

import ROOT as rt
import numpy as np
from larcv import larcv
from larcvdataset.larcvserver import LArCVServer
from torch.utils import data as torchdata

def load_sparse_ssnetdata(io):
    """
    we need the input data to be a pixel list
    however, the ground truth can be dense arrays

    products returned:
    "adc":     input adc from image2d_wire_tree
    "label":   true class from image2d_iseg_tree
    "weight":  weight matrix from image2d_ts_keyspweight_tree
    """
    
    threshold = 10.0
    plane     = 2
    
    data = {}
    ev_wire   = io.get_data(larcv.kProductImage2D,"wire")
    ev_label  = io.get_data(larcv.kProductImage2D,"segment")
    ev_weight = io.get_data(larcv.kProductImage2D,"ts_keyspweight")
    nimgs = ev_wire.Image2DArray().size()
    meta = ev_wire.Image2DArray().front().meta()

    data["pixlist"] = larcv.as_pixelarray( ev_wire.Image2DArray().at(plane), threshold )
    data["weight"]  = larcv.as_pixelarray_with_selection( ev_weight.Image2DArray().at(plane),
                                                          ev_wire.Image2DArray().at(plane),
                                                          threshold, True,
                                                          larcv.msg.kNORMAL  )

    # for the labels, we have to convert Particle ID from LArCV into track/shower/background
    # showers are larcv larcv::ROIType_t: kEminus(3), kGamma(4), kPizero(5)
    # tracks are larcv::ROITtype_t: muminus(6),kminus(7),piminus(8),proton(9)
    # background are larcv::ROIType_t: unknown(0),cosmic(1),bnb(2)
    # Definition is in larcv/core/DataFormat/DataFormatTypes.h
    # shower
    origlabels = larcv.as_pixelarray_with_selection( ev_label.Image2DArray().at(plane),
                                                     ev_wire.Image2DArray().at(plane),
                                                     threshold, True,
                                                     larcv.msg.kNORMAL  )
    data["label"] = np.zeros( origlabels.shape, dtype=np.int )
    # copy coords
    data["label"][:,0:2] = origlabels[:,0:2]
    # set the labels

    # BG: LABEL 0 (default)
    # SHOWER: LABEL = 2
    data["label"][:,2][ origlabels[:,2]==3 ] = 2
    data["label"][:,2][ origlabels[:,2]==4 ] = 2
    data["label"][:,2][ origlabels[:,2]==5 ] = 2
    # TRACK: LABEL = 1
    data["label"][:,2][ origlabels[:,2]>=6 ] = 1
    data["origlabel"] = origlabels
    
    #data["weight"]  = larcv.as_ndarray( ev_weight.Image2DArray().at(plane) ).transpose( (1,0) )
    
    #data["label"]   = larcv.as_ndarray( ev_label.Image2DArray().at(plane)  ).transpose( (1,0) )
    #data["weight"]  = larcv.as_ndarray( ev_weight.Image2DArray().at(plane) ).transpose( (1,0) )

    return data
                                          

def load_ssnet_larcvdata( name, inputfile, batchsize, nworkers, tickbackward=False ):
    feeder = LArCVServer(batchsize,name,load_sparse_ssnetdata,inputfile,nworkers,
                         server_verbosity=2,worker_verbosity=2,io_tickbackward=tickbackward)
    return feeder

class SparseImagePyTorchDataset(torchdata.Dataset):
    idCounter = 0
    def __init__(self,inputfile,batchsize,tickbackward=False,nworkers=4):
        super(SparseImagePyTorchDataset,self).__init__()

        # get length by querying the tree
        self.nentries  = 0
        rfile = rt.TFile(inputfile,"open")
        self.nentries = rfile.Get("image2d_wire_tree").GetEntries()
        rfile.Close()
        
        self.feedername = "SparseImagePyTorchDataset_%d"%(SparseImagePyTorchDataset.idCounter)
        self.batchsize = batchsize
        self.nworkers  = nworkers
        self.inputfile = inputfile
        self.feeder = LArCVServer(self.batchsize,self.feedername,
                                  load_sparse_ssnetdata,self.inputfile,self.nworkers,
                                  server_verbosity=0,worker_verbosity=0,
                                  io_tickbackward=tickbackward)
        SparseImagePyTorchDataset.idCounter += 1

    def __len__(self):
        #print "return length of sample:",self.nentries
        return self.nentries

    def __getitem__(self,index):
        """ we do not have a way to get the index (can change that)"""
        #print "called get item for index=",index," ",self.feeder.identity,"pid=",os.getpid()
        data = self.feeder.get_batch_dict()        
        # remove the feeder variable
        del data["feeder"]
        #print "called get item: ",data.keys()
        return data
        
        

if __name__ == "__main__":
    
    "testing"
    inputfile = "~/trainingdata/mcc8ssnet/train00.root"
    batchsize = 1
    nworkers  = 4
    tickbackward = True
    
    nentries = 50

    TEST_VANILLA = False
    TEST_PYTORCH_DATALOADER = True

    if TEST_VANILLA:
        feeder = load_ssnet_larcvdata( "sparsetest", inputfile, batchsize, nworkers, tickbackward=tickbackward )
        tstart = time.time()

        print "TEST LARCVDATASET SERVER"
        for n in xrange(nentries):
            batch = feeder.get_batch_dict()
            print "entry[",n,"] from ",batch["feeder"],": ",batch.keys(),"npts[p0]=",batch["pixlist"][0].shape
        tend = time.time()-tstart
        print "elapsed time, ",tend,"secs ",tend/float(nentries)," sec/batch"
        del feeder


    if TEST_PYTORCH_DATALOADER:
        print "TEST PYTORCH DATALOADER SERVER"
        print "DOES NOT WORK"
        params = {"batch_size":4,
                  "shuffle":True,
                  "pin_memory":True,
                  "num_workers":4}
        dataset = SparseImagePyTorchDataset(inputfile,tickbackward=True)
        pyloader = torchdata.DataLoader(dataset,**params)

        ientry = 0
        for batch in pyloader:
            print "entry[",n,"]: ",type(batch)
            ientry += 1
            if ientry>50:
                break
        tend = time.time()-tstart
        print "elapsed time, ",tend,"secs ",tend/float(nentries)," sec/batch"
        del pyloader


