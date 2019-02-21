import os,sys,time

import ROOT as rt
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
    ev_label  = io.get_data(larcv.kProductImage2D,"ts_keyspweight")
    ev_weight = io.get_data(larcv.kProductImage2D,"iseg")
    nimgs = ev_wire.Image2DArray().size()
    meta = ev_wire.Image2DArray().front().meta()

    data["pixlist"] = larcv.as_pixelarray( ev_wire.Image2DArray().at(plane), threshold )
    data["label"]   = larcv.as_ndarray( ev_label.Image2DArray().at(plane)  ).transpose( (1,0) )
    data["weight"]  = larcv.as_ndarray( ev_weight.Image2DArray().at(plane) ).transpose( (1,0) )

    return data
                                          

def load_ssnet_larcvdata( name, inputfile, batchsize, nworkers, tickbackward=False ):
    feeder = LArCVServer(batchsize,name,load_sparse_ssnetdata,inputfile,nworkers,
                         server_verbosity=2,worker_verbosity=2,io_tickbackward=tickbackward)
    return feeder

class SparseImagePyTorchDataset(torchdata.Dataset):
    idCounter = 0
    def __init__(self,inputfile,tickbackward=False):
        super(SparseImagePyTorchDataset,self).__init__()

        # get length by querying the tree
        self.nentries  = 0
        rfile = rt.TFile(inputfile,"open")
        self.nentries = rfile.Get("image2d_wire_tree").GetEntries()
        rfile.Close()
        
        self.feedername = "SparseImagePyTorchDataset_%d"%(SparseImagePyTorchDataset.idCounter)
        self.batchsize = 1
        self.nworkers  = 1
        self.inputfile = inputfile
        self.feeder = LArCVServer(self.batchsize,self.feedername,
                                  load_sparse_ssnetdata,self.inputfile,self.nworkers,
                                  server_verbosity=0,worker_verbosity=0,
                                  io_tickbackward=tickbackward)
        SparseImagePyTorchDataset.idCounter += 1

    def __len__(self):
        print "return length of sample:",self.nentries
        return self.nentries

    def __getitem__(self,index):
        """ we do not have a way to get the index (can change that)"""
        print "called get item for index=",index," ",self.feeder.identity,"pid=",os.getpid()
        data = self.feeder.get_batch_dict()        
        # remove the feeder variable
        del data["feeder"]
        print "called get item: ",data.keys()
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


