import os,sys

sys.path.append("../models")

from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from ubssnet import ubSSNet
from ROOT import std

import torch
import numpy as np

# PARAMETERS
# --------------------------------------------------------
device = torch.device("cuda")

# LARCV/LARLITE FILE IO
# --------------------------------------------------------
supera = "~/working/larbys/ubdl/testdata/ex1/supera-Run000001-SubRun006867.root"
opreco = "~/working/larbys/ubdl/testdata/ex1/opreco-Run000001-SubRun006867.root"
croialgo = ublarcvapp.ubdllee.FixedCROIFromFlashAlgo()

io = larcv.IOManager(larcv.IOManager.kBOTH,"io",larcv.IOManager.kTickBackward)
io.add_in_file(supera)
io.set_out_file("output_deploy_ubssnet.root")
io.initialize()

ioll = ublarcvapp.LArliteManager(larlite.storage_manager.kREAD)
ioll.add_in_filename(opreco)
ioll.open()

# PYTORCH MODEL
# -------------------------------------------------------------
model = ubSSNet("../weights/saved_caffe_weights_plane2.tar").eval().to(device)

# READ ENTRY DATA
# --------------------------------------------------------------
io.read_entry(1)
ioll.syncEntry(io)

ev_img     = io.get_data(larcv.kProductImage2D,"wire")
ev_opflash = ioll.get_data(larlite.data.kOpFlash,"simpleFlashBeam")

usec_min = 190*0.015625;
usec_max = 320*0.015625;
roi_v = []

for iopflash in xrange(ev_opflash.size()):
    opflash = ev_opflash.at(iopflash)
    if opflash.Time()<usec_min or opflash.Time()>usec_max: continue
    flashrois = croialgo.findCROIfromFlash( opflash );
    for iroi in xrange(flashrois.size()):
        roi_v.append( flashrois.at(iroi) )

# cropout the regions
wholeview_v = ev_img.Image2DArray()
nplanes = wholeview_v.size();
ncrops = 0;
crop_v = []
crop_meta_v = []
for plane in xrange(nplanes):
    if plane not in [2]:
        continue
    wholeimg = wholeview_v.at(plane)
    for roi in roi_v:
        bbox = roi.BB(plane)
        #std::cout << "crop from the whole image" << std::endl;
        crop = wholeimg.crop( bbox )
        crop_np = larcv.as_ndarray(crop).reshape((1,1,512,512))
        crop_np = np.transpose( crop_np, (0,1,3,2))
        crop_v.append( crop_np )
        ncrops+=1
print "cropped the regions: total ",ncrops

# run through the network
# -------------------------------------------------
out_v = []
for crop in crop_v:
    crop_t = torch.from_numpy( crop ).to(device)
    print "run input: ",crop_t.size(),"sum: ",crop_t.detach().sum()
    out_t = model(crop_t)
    out_v.append( out_t.detach().cpu().numpy() )

# merge the output
# --------------------------------------------------
shower_img = larcv.Image2D( wholeview_v.at(0).meta() )
track_img  = larcv.Image2D( wholeview_v.at(1).meta() )
bg_img     = larcv.Image2D( wholeview_v.at(2).meta() )
shower_img.paint(0)
track_img.paint(0)
bg_img.paint(0)


for out,roi in zip(out_v,roi_v):

    # threshold scores for better compression
    out[ out<1.0e-2 ] = 0.0
    showerslice = np.transpose( out[:,0,:,:].reshape(512,512), (1,0) )
    trackslice  = np.transpose( out[:,1,:,:].reshape(512,512), (1,0) )
    bgslice     = np.transpose( out[:,2,:,:].reshape(512,512), (1,0) )

    # back to image2d
    showercrop = larcv.as_image2d_meta( showerslice, roi.BB(2) )
    shower_img.overlay(showercrop, larcv.Image2D.kOverWrite )
    trackcrop  = larcv.as_image2d_meta( trackslice, roi.BB(2) )
    track_img.overlay(trackcrop, larcv.Image2D.kOverWrite)
    bgcrop     = larcv.as_image2d_meta( bgslice, roi.BB(2) )
    bg_img.overlay(bgcrop, larcv.Image2D.kOverWrite)

ev_ssnetout = io.get_data(larcv.kProductImage2D,"ssnet_plane2")
ev_ssnetout.Append(shower_img)
ev_ssnetout.Append(track_img)
ev_ssnetout.Append(bg_img)

io.save_entry()

ioll.close()
io.finalize()


print "done"
