import os,sys

sys.path.append("../models")

from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from ubssnet import ubSSNet
import ROOT as rt
from ROOT import std
larcv.load_rootutil()
larcv.load_pyutil()

import torch
import numpy as np

# PARAMETERS
# --------------------------------------------------------
device = torch.device("cuda")
debug_w_png = True
if debug_w_png:
    import scipy

import  scipy.misc


# LARCV/LARLITE FILE IO
# --------------------------------------------------------
supera = "~/working/larbys/ubdl/testdata/ex1/supera-Run000001-SubRun006867.root"
opreco = "~/working/larbys/ubdl/testdata/ex1/opreco-Run000001-SubRun006867.root"
croialgo = ublarcvapp.ubdllee.FixedCROIFromFlashAlgo()

io = larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickBackward)
io.add_in_file(supera)
io.initialize()

ioll = ublarcvapp.LArliteManager(larlite.storage_manager.kREAD)
ioll.add_in_filename(opreco)
ioll.open()

outio = larcv.IOManager(larcv.IOManager.kWRITE,"outio")
outio.set_out_file("output_deploy_ubssnet.root")
outio.initialize()

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
crop_img2d_v = []
c = rt.TCanvas("c","c",800,600)
for plane in xrange(nplanes):
    if plane not in [2]:
        continue
    wholeimg = wholeview_v.at(plane)
    for roi in roi_v:
        bbox = roi.BB(plane)
        #std::cout << "crop from the whole image" << std::endl;
        crop = wholeimg.crop( bbox )

        h2d = larcv.as_th2d(crop,"roi%d"%(ncrops))
        c.Draw()
        h2d.Draw("colz")
        c.Update()
        print "[enter to continue]"
        raw_input()
        crop_np = larcv.as_ndarray(crop)
        np.nan_to_num(crop_np,copy=False)
        crop_np[ crop_np<10.0 ]   = 0.0
        crop_np[ crop_np>100.0  ] = 100.0
        print crop_np.shape,":",crop.meta().dump()
        #crop_np = np.transpose( crop_np, (0,1,3,2))

        if debug_w_png:
            scipy.misc.toimage(crop_np, cmin=0.0, cmax=100.0).\
                        save("adc_crop%d.png"%(ncrops))

        crop_v.append( crop_np.reshape((1,1,512,512)) )
        crop_img2d_v.append(crop)

        ncrops+=1
        #break
print "cropped the regions: total ",ncrops

# run through the network
# -------------------------------------------------
out_v = []
for icrop,crop in enumerate(crop_v):
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


for out,img2d in zip(out_v,crop_img2d_v):

    # threshold scores for better compression
    showerslice = out[:,1,:,:].reshape(512,512)
    trackslice  = out[:,2,:,:].reshape(512,512)
    bgslice     = out[:,0,:,:].reshape(512,512)

    meta = img2d.meta()
    # back to image2d
    shrmeta = larcv.ImageMeta( meta.width(), meta.height(),
                                meta.rows(), meta.cols(),
                                meta.min_x(), meta.min_y(), 0)
    trkmeta = larcv.ImageMeta( meta.width(), meta.height(),
                                meta.rows(), meta.cols(),
                                meta.min_x(), meta.min_y(), 1)
    bgmeta  = larcv.ImageMeta( meta.width(), meta.height(),
                                meta.rows(), meta.cols(),
                                meta.min_x(), meta.min_y(), 2)
    showercrop = larcv.as_image2d_meta( showerslice, shrmeta )
    shower_img.overlay(showercrop, larcv.Image2D.kOverWrite )
    trackcrop  = larcv.as_image2d_meta( trackslice, trkmeta )
    track_img.overlay(trackcrop, larcv.Image2D.kOverWrite)
    bgcrop     = larcv.as_image2d_meta( bgslice, bgmeta )
    bg_img.overlay(bgcrop, larcv.Image2D.kOverWrite)

h2d = larcv.as_th2d(shower_img,"shower_img")
c.Clear()
h2d.Draw("colz")
c.Update()
print "enter to go to next image"
raw_input()

ev_ssnetout = outio.get_data(larcv.kProductImage2D,"ssnet_plane2")
ev_ssnetout.Append(shower_img)
ev_ssnetout.Append(track_img)
ev_ssnetout.Append(bg_img)

ev_imgout   = outio.get_data(larcv.kProductImage2D,"wire")
for p in xrange(0,3):
    ev_imgout.Append( wholeview_v.at(p) )

outio.set_id(io.event_id().run(), io.event_id().subrun(), io.event_id().event())


outio.save_entry()

ioll.close()
io.finalize()
outio.finalize()


print "done"
