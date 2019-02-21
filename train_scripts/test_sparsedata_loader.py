import os,sys,time

import ROOT as rt
from larcv import larcv
from larcvdataset.larcvserver import LArCVServer

# dataset
sys.path.append( os.environ["PWD"]+"/../dataloader" )
from sparseimgdata import SparseImagePyTorchDataset
    
inputfile = "~/trainingdata/mcc8ssnet/train00.root"

batchsize = 1
print "start feeders"
feeder = SparseImagePyTorchDataset(inputfile, batchsize, tickbackward=True)

print "wait for workers to load up"
time.sleep(1)
    
print "start receiving [enter to continue]"
#raw_input()

nentries = 1
tstart = time.time()
for n in xrange(nentries):
    batch = feeder[0]
    print "entry[",n,"] keys=",batch.keys(),"shape[pixlist]=",batch["pixlist"][0].shape

    # check visually
    hadc = rt.TH2D("hadc_%d",";;ADC",512,0,512,512,0,512)
    hseg = rt.TH2D("hseg_%d",";;segment",512,0,512,512,0,512)

    pixlist = batch["pixlist"][0]
    labellist = batch["label"][0]
    for ipix in xrange(pixlist.shape[0]):
        hadc.SetBinContent( int(pixlist[ipix,0])+1, int(pixlist[ipix,1])+1, float(pixlist[ipix,2]) )
        hseg.SetBinContent( int(pixlist[ipix,0])+1, int(pixlist[ipix,1])+1, float(labellist[ipix,2]) )
        
    c = rt.TCanvas("c","c",1200,500)
    c.Divide(2,1)
    c.cd(1)
    hadc.Draw("colz")
    c.cd(2)
    hseg.Draw("colz")
    raw_input()
tend = time.time()-tstart
print "elapsed time, ",tend,"secs ",tend/float(nentries)," sec/batch"
sys.exit(-1)



