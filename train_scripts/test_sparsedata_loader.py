import os,sys,time

import ROOT as rt
from larcv import larcv
from larcvdataset.larcvserver import LArCVServer

rt.gStyle.SetOptStat(0)

# dataset
sys.path.append( os.environ["PWD"]+"/../dataloader" )
from sparseimgdata import SparseImagePyTorchDataset
    
#inputfile = "~/trainingdata/mcc8ssnet/train00.root"
#inputfile = ["/media/hdd1/larbys/ssnet_dllee_trainingdata/train00.root",
#             "/media/hdd1/larbys/ssnet_dllee_trainingdata/train01.root",
#             "/media/hdd1/larbys/ssnet_dllee_trainingdata/train02.root",
#             "/media/hdd1/larbys/ssnet_dllee_trainingdata/train03.root"]
inputfile = ["/media/hdd1/larbys/ssnet_dllee_trainingdata/train01.root"]

batchsize = 1
print "start feeders"
feeder = SparseImagePyTorchDataset(inputfile, batchsize, tickbackward=True)

print "wait for workers to load up"
time.sleep(1)
    
print "start receiving [enter to continue]"
#raw_input()

c = rt.TCanvas("c","c",1500,500)
c.Divide(3,1)
c.Draw()
nentries = 50
tstart = time.time()
for n in xrange(nentries):
    batch = feeder[0]
    print "entry[",n,"] keys=",batch.keys(),"shape[pixlist]=",batch["pixlist"][0].shape

    # check visually
    hadc = rt.TH2D("hadc_%d"%(n),";;ADC",512,0,512,512,0,512)
    hseg = rt.TH2D("hseg_%d"%(n),";;segment",512,0,512,512,0,512)
    hwgt = rt.TH2D("hwgt_%d"%(n),";;weights",512,0,512,512,0,512)    

    pixlist   = batch["pixlist"][0]
    labellist = batch["label"][0]
    weight    = batch["weight"][0]
    for ipix in xrange(pixlist.shape[0]):
        hadc.SetBinContent( int(pixlist[ipix,0])+1, int(pixlist[ipix,1])+1, float(pixlist[ipix,2]) )
        hseg.SetBinContent( int(pixlist[ipix,0])+1, int(pixlist[ipix,1])+1, float(labellist[ipix,2]+1)*10.0 )
        hwgt.SetBinContent( int(weight[ipix,0])+1,  int(weight[ipix,1])+1,  float(weight[ipix,2]) )        
        
    c.cd(1)
    hadc.Draw("colz")
    c.cd(2)
    hseg.Draw("colz")
    hseg.SetMaximum(20)
    hseg.SetMinimum(0)    
    c.cd(3)
    hwgt.Draw("colz")
    c.Update()
    raw_input()
tend = time.time()-tstart
print "elapsed time, ",tend,"secs ",tend/float(nentries)," sec/batch"
sys.exit(-1)



