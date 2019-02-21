import os,sys,time

import ROOT as rt
from larcv import larcv
from larcvdataset.larcvserver import LArCVServer

def load_data(io):

    threshold = 10.0
    
    data = {}
    ev_wire = io.get_data(larcv.kProductImage2D,"wire")
    nimgs = ev_wire.Image2DArray().size()
    meta = ev_wire.Image2DArray().front().meta()
    
    for iimg in xrange(nimgs):
        data["pixdata_p%d"%(iimg)] = larcv.as_pixelarray( ev_wire.Image2DArray().at(iimg), threshold )

    return data
    
inputfile = "~/trainingdata/mcc8ssnet/train00.root"

batchsize = 1
nworkers  = 1
print "start feeders"
feeder = LArCVServer(batchsize,"test",load_data,inputfile,nworkers,server_verbosity=0,worker_verbosity=0)

print "wait for workers to load up"
time.sleep(1)
    
print "start receiving [enter to continue]"
#raw_input()

nentries = 50
tstart = time.time()
for n in xrange(nentries):
    batch = feeder.get_batch_dict()
    print "entry[",n,"] from ",batch["feeder"],": ",batch.keys(),"npts[p0]=",batch["pixdata_p0"][0].shape
tend = time.time()-tstart
print "elapsed time, ",tend,"secs ",tend/float(nentries)," sec/batch"
sys.exit(-1)



