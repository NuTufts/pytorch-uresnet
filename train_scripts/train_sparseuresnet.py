#!/bin/env python

## IMPORT

# python,numpy
import os,sys,commands
import shutil
import time
import traceback
import numpy as np

# ROOT, larcv
import ROOT
from larcv import larcv

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

# tensorboardX
from tensorboardX import SummaryWriter

# Our model definition
print "model dir: ",os.environ["PWD"]+"/../models"
sys.path.append( os.environ["PWD"]+"/../models" )
from spuresnet import SparseUResNet

# dataset
sys.path.append( os.environ["PWD"]+"/../dataloader" )
from sparseimgdata import SparseImagePyTorchDataset

GPUMODE=True
GPUID=0
RESUME_FROM_CHECKPOINT=False
RUNPROFILER=False
CHECKPOINT_FILE="plane2_caffe/run1/checkpoint.20000th.tar"
INPUTDIR ="/media/hdd1/larbys/ssnet_dllee_trainingdata"
inputfiles_train = [INPUTDIR+"/train00.root"]
#                    INPUTDIR+"/train02.root"]
inputfiles_valid = [INPUTDIR+"/train01.root"]


# taken from torch.nn.modules.loss
def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"
        
class PixelWiseLoss(nn.modules.loss._WeightedLoss):
    def __init__(self,weight=None, size_average=True, ignore_index=-100 ):
        super(PixelWiseLoss,self).__init__(weight,size_average)
        self.ignore_index = ignore_index
        self.reduce = False
        self.cross_entropy = torch.nn.CrossEntropyLoss( reduction='none' )
        #self.mean = torch.mean.cuda()

    def forward(self,predict_t,target_t,pixelweights_t):
        """
        coords:  (N,3) coordinates of non-zero input. last dim is (row,col,batch)
        predict: (N,3) scores for each point
        target:  (N,3) tensor with correct class
        pixelweights: (N,3) tensor with weights for each pixel
        """
        
        pixelloss = self.cross_entropy( predict_t, target_t )
        
        pixelloss *= pixelweights_t
        weightsum  = pixelweights_t.sum()
        loss = pixelloss.sum()/weightsum
        
        return loss
        

# global variables
best_prec1 = 0.0  # best accuracy, use to decide when to save network weights
writer = SummaryWriter()
device = torch.device("cuda")

def main():

    global best_prec1
    global writer

    # create model
    inputshape = (512,512)
    reps = 2
    nfeatures = 32
    nplanes = 5
    noutputclasses = 2 # shower[0], track[1]
    model = SparseUResNet( inputshape, reps, nfeatures, nplanes, noutputclasses )
    if GPUMODE:
        model = model.to(device=device)

    # uncomment to dump model
    print "Loaded model: ",model
    # check where model pars are
    #for p in model.parameters():
    #    print p.is_cuda
    
    # define loss function (criterion) and optimizer
    if GPUMODE:
        criterion = PixelWiseLoss().to(device=device)
    else:
        criterion = PixelWiseLoss()

    # training parameters
    lr = 2.0e-3
    momentum = 0.9
    weight_decay = 1.0e-3

    # training length
    batchsize_train = 10
    batchsize_valid = 10
    start_epoch = 0
    epochs      = 1
    start_iter  = 0
    num_iters   = 10000
    #num_iters    = None # if None
    iter_per_epoch = None # determined later
    iter_per_valid = 10
    iter_per_checkpoint = 500

    nbatches_per_itertrain = 10
    itersize_train         = batchsize_train*nbatches_per_itertrain
    trainbatches_per_print = 1
    
    nbatches_per_itervalid = 4
    itersize_valid         = batchsize_valid*nbatches_per_itervalid
    validbatches_per_print = 1

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    cudnn.benchmark = True

    # LOAD THE DATASET
    iotrain = SparseImagePyTorchDataset(inputfiles_train, batchsize_train, tickbackward=True, nworkers=4)
    iovalid = SparseImagePyTorchDataset(inputfiles_valid, batchsize_valid, tickbackward=True, nworkers=2)

    NENTRIES = len(iotrain)
    
    print "Number of entries in training set: ",NENTRIES

    if NENTRIES>0:
        iter_per_epoch = NENTRIES/(itersize_train)
        if num_iters is None:
            # we set it by the number of request epochs
            num_iters = (epochs-start_epoch)*NENTRIES
        else:
            epochs = num_iters/NENTRIES
    else:
        iter_per_epoch = 1

    print "Number of epochs: ",epochs
    print "Iter per epoch: ",iter_per_epoch


    with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:

        # Resume training option
        if RESUME_FROM_CHECKPOINT:
            checkpoint = torch.load( CHECKPOINT_FILE )
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint['optimizer'])
        

        for ii in range(start_iter, num_iters):

            adjust_learning_rate(optimizer, ii, lr)
            print "Iter:%d Epoch:%d.%d "%(ii,ii/iter_per_epoch,ii%iter_per_epoch),
            for param_group in optimizer.param_groups:
                print "lr=%.3e"%(param_group['lr']),
                print

            # train for one epoch
            try:
                train_ave_loss, train_ave_acc = train(iotrain, batchsize_train, model,
                                                      criterion, optimizer,
                                                      nbatches_per_itertrain, ii, trainbatches_per_print,
                                                      noutputclasses)
            except Exception,e:
                print "Error in training routine!"            
                print e.message
                print e.__class__.__name__
                traceback.print_exc(e)
                break
            print "Iter:%d Epoch:%d.%d train aveloss=%.3f aveacc=%.3f"%(ii,ii/iter_per_epoch,ii%iter_per_epoch,train_ave_loss,train_ave_acc)
            
            # evaluate on validation set
            if ii%iter_per_valid==0:
                try:
                    prec1 = validate(iovalid, batchsize_valid, model, criterion,
                                     nbatches_per_itervalid, validbatches_per_print,
                                     ii, noutputclasses)
                except Exception,e:
                    print "Error in validation routine!"            
                    print e.message
                    print e.__class__.__name__
                    traceback.print_exc(e)
                    break

                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)

                # check point for best model
                if is_best:
                    print "Saving best model"
                    save_checkpoint({
                        'iter':ii,
                        'epoch': ii/iter_per_epoch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, -1)

            # periodic checkpoint
            if ii>0 and ii%iter_per_checkpoint==0:
                print "saving periodic checkpoint"
                save_checkpoint({
                    'iter':ii,
                    'epoch': ii/iter_per_epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, False, ii)
                
        # end of profiler context
        print "saving last state"
        save_checkpoint({
            'iter':num_iters,
            'epoch': num_iters/iter_per_epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, False, num_iters)


    print "FIN"
    print "PROFILER"
    print prof
    writer.close()


def train(train_loader, batchsize, model, criterion, optimizer, nbatches, epoch, print_freq, noutputclasses):

    global writer
    
    batch_time = AverageMeter() # total for batch
    data_time = AverageMeter()
    format_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    acc_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    acc_list = []
    for i in range(noutputclasses+2):
        acc_list.append( AverageMeter() )

    # switch to train mode
    model.train()

    for i in range(0,nbatches):
        #print "epoch ",epoch," batch ",i," of ",nbatches
        batchstart = time.time()

        # data loading time        
        end = time.time()        
        data = train_loader[0]
        data_time.update(time.time() - end)


        # convert to pytorch Variable (with automatic gradient calc.)
        end = time.time()
        # input: need two tensors (1) coordinate which is Nx3 (2) features N
        input_coord_t, input_feats_t, labels_t, weights_t, batchsize_t = convert_to_tensor( data, device )
        dtformat = time.time()-end
        format_time.update( dtformat )
        
        # compute output
        if RUNPROFILER:
            torch.cuda.synchronize()
        end = time.time()
        #print "input: ",input_coord_t.shape
        output_t = model(input_coord_t,input_feats_t,batchsize_t)
        #print "output: ",output_t.shape
        
        loss = criterion(output_t, labels_t, weights_t)

        if RUNPROFILER:
            torch.cuda.synchronize()                
        forward_time.update(time.time()-end)

        # compute gradient and do SGD step
        if RUNPROFILER:
            torch.cuda.synchronize()                
        end = time.time()        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if RUNPROFILER:        
            torch.cuda.synchronize()                
        backward_time.update(time.time()-end)

        #print "finished backward"

        # measure accuracy and record loss
        end = time.time()
        acc_v = accuracy(output_t, labels_t, batchsize_t)
        acc_time.update(time.time()-end)

        # updates
        losses.update(loss.item(), batchsize_t.item())
        for i,acc in enumerate(acc_v):
            acc_list[i].update( acc )

        # measure elapsed time for batch
        batch_time.update(time.time() - batchstart)


        if i % print_freq == 0:
            status = (epoch,i,nbatches,
                      batch_time.val,batch_time.avg,
                      data_time.val,data_time.avg,
                      format_time.val,format_time.avg,
                      forward_time.val,forward_time.avg,
                      backward_time.val,backward_time.avg,
                      acc_time.val,acc_time.avg,                      
                      losses.val,losses.avg,
                      acc_list[3].val,acc_list[3].avg)
            print "Iter: [%d][%d/%d] secs \tBatch %.3f (%.3f)\tData %.3f (%.3f)\tFormat %.3f (%.3f)\tForw %.3f (%.3f)\tBack %.3f (%.3f)\tAcc %.3f (%.3f)\t || \tLoss %.3f (%.3f)\tAcc %.3f (%.3f)"%status
        

    writer.add_scalar('data/train_loss', losses.avg, epoch )        
    writer.add_scalars('data/train_accuracy', {'track':  acc_list[1].avg,
                                               'shower': acc_list[0].avg,
                                               'total':  acc_list[2].avg,
                                               'nonzero':acc_list[3].avg}, epoch )        
    
    return losses.avg,top1.avg


def validate(val_loader, batchsize, model, criterion, nbatches, print_freq, iiter, noutputclasses):

    global writer
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    acc_list = []
    for i in range(noutputclasses+2):
        acc_list.append( AverageMeter() )
    
    # switch to evaluate mode
    model.eval()

    tallbatches = time.time()
    for i in range(0,nbatches):
        tbatch = time.time()

        tdataload = time.time()
        data = val_loader[0]
        tdataload = time.time()-tdataload

        tformat = time.time()
        input_coord_t, input_feats_t, labels_t, weights_t, batchsize_t = convert_to_tensor( data, device )
        tformat = tformat - time.time()
        
        # compute output
        output_t = model(input_coord_t,input_feats_t, batchsize_t)

        # loss
        loss = criterion(output_t, labels_t, weights_t)

        # measure accuracy and record loss
        acc_v = accuracy(output_t, labels_t, batchsize_t)
        
        losses.update(loss.item(), batchsize_t.item())
        for i,acc in enumerate(acc_v):
            acc_list[i].update( acc )
                
        # measure elapsed time
        batch_time.update(tbatch - time.time())

        if i % print_freq == 0:
            status = (i,nbatches,batch_time.val,batch_time.avg,losses.val,losses.avg,acc_list[2].val,acc_list[2].avg)
            print "Valid: [%d/%d]\tTime %.3f (%.3f)\tLoss %.3f (%.3f)\tPrec@1 %.3f (%.3f)"%status
            #print('Test: [{0}/{1}]\t'
            #      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #        i, len(val_loader), batch_time=batch_time, loss=losses,
            #        top1=top1))

    #print(' * Prec@1 {top1.avg:.3f}'
    #      .format(top1=top1))

    writer.add_scalar( 'data/valid_loss', losses.avg, iiter )
    writer.add_scalars('data/valid_accuracy', {'track':   acc_list[1].avg,
                                               'shower':  acc_list[0].avg,
                                               'total':   acc_list[2].avg,
                                               'nonzero': acc_list[3].avg}, iiter )

    print "Test:Result* Acc[total] %.3f\tLoss %.3f"%(acc_list[2].avg,losses.avg)

    return float(top1.avg)


def save_checkpoint(state, is_best, p, filename='checkpoint.pth.tar'):
    if p>0:
        filename = "checkpoint.%dth.tar"%(p)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = lr * (0.5 ** (epoch // 300))
    lr = lr
    #lr = lr*0.992
    #print "adjust learning rate to ",lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(predict_t, labels_t, batchsize_t, profile=False):
    """
    Computes the accuracy.

    inputs
    ------
    
    predict_t:  (N,3) float tensor
    labels_t:   (N,1) long tensor
    batchsize_t: (1,) long tensor
    """

    # detach the tensor so we don't add to the compute graph
    # note: this is not a copy -- do not make in-place modifications to it
    predict      = predict_t.detach()
    truth_label  = labels_t.detach()
    batchsize    = batchsize_t.item()

    if profile:
        torch.cuda.synchronize()
        start = time.time()    
    
    profile = False
    # needs to be as gpu as possible!
    pred_label = torch.argmax( predict, dim=1 )
    compare    = torch.eq( pred_label, truth_label )
    totcorrect = compare.sum()

    # we want counts for elements wise
    num_per_class = {}
    corr_per_class = {}
    total_corr = 0
    total_pix  = 0
    if profile:
        torch.cuda.synchronize()            
        start = time.time()

    nclasses = predict.size(1)
    for c in xrange(nclasses):
        # get entries with this truthlabel: 1's and 0's
        classentries = torch.eq( truth_label, c )
        
        num_per_class[c]  = classentries.sum().item()

        correct = torch.eq( pred_label, c )*classentries
        corr_per_class[c] = correct.sum().item()

        total_corr += corr_per_class[c]
        total_pix  += num_per_class[c]
    if profile:
        torch.cuda.synchronize()                
        print "time to reduce: ",time.time()-start," secs"
        
    # make result vector
    # first 'nclasses' entries are accuracy for each class
    # then we add totals
    #   res[nclasses] = total correct
    #   res[nclasses+1] = track-shower combined accuracy (i.e. non-background)
    res = []
    
    # class results
    for c in xrange(nclasses):
        if num_per_class[c]>0:
            res.append( corr_per_class[c]/float(num_per_class[c])*100.0 )
        else:
            res.append( 0.0 )

    # totals
    res.append( 100.0*float(total_corr)/total_pix ) # all classes
    res.append( 100.0*float(corr_per_class[0]+corr_per_class[1])/(num_per_class[0]+num_per_class[1]) ) # track/shower acc
        
    return res

def dump_lr_schedule( startlr, numepochs ):
    for epoch in range(0,numepochs):
        lr = startlr*(0.5**(epoch//300))
        if epoch%10==0:
            print "Epoch [%d] lr=%.3e"%(epoch,lr)
    print "Epoch [%d] lr=%.3e"%(epoch,lr)
    return

def convert_to_tensor( data, device ):
    """ 
    we take the dictionary from the larcvserver dataloader and convert the various
    numpy arrays into pytorch tensors

    inputs
    ------
    data: dictionary with numpy arrays

    output
    ------
    input_coord_t: (N,3) longtensor with (row,col,batch)
    input_feats_t: (N,1) floattensor with pixel values
    labels_t:      (N,1) longtensor with truth class
    weights_t:     (N,1) float tensor with weights for each pixel
    batchsize_t:   (1)   longtensor, batchsize
    """

    batchsize = len(data["pixlist"])
    totinputs = 0
    for input_batch in data["pixlist"]:
        totinputs += input_batch.shape[0]
    input_coord_t = torch.LongTensor(totinputs,3)
    input_feats_t = torch.FloatTensor(totinputs,1)
    labels_t      = torch.LongTensor(totinputs)
    weights_t     = torch.FloatTensor(totinputs)
    
    ninputs = 0
    for batchid,(input_batch,label_np,weight_np) in enumerate(zip(data["pixlist"],data["label"],data["weight"])):
        input_coord_t[ninputs:ninputs+input_batch.shape[0],0:2] = torch.Tensor(input_batch[:,0:2].astype(np.int64))
        input_coord_t[ninputs:ninputs+input_batch.shape[0],2]   = batchid
        input_feats_t[ninputs:ninputs+input_batch.shape[0],0]   = torch.Tensor(input_batch[:,2])
        labels_t[ninputs:ninputs+input_batch.shape[0]]        = torch.Tensor(label_np[:,2])
        weights_t[ninputs:ninputs+input_batch.shape[0]]       = torch.Tensor(weight_np[:,2])
        ndiff = np.not_equal(label_np[:,0:2],weight_np[:,0:2]).sum()
        if ndiff!=0:
            raise ValueError("coordinates of label and weight pixels are different. ndiff=%d of %d"%(ndiff,label_np.shape[0]))
        ninputs += input_batch.shape[0]

    # batchsize
    batchsize_t = torch.LongTensor([batchsize])

    #print "format/xfer data: ",dtformat
    #print "coords:", input_coord_t[0:10,:]
    #print "pixvals:",input_feats_t[0:10]
    #print "labels:",  labels_t[0:10]
    #print "origlabels: ",data["origlabel"][0][0:10,2]
    #print "weight:",  weights_t[0:10]
    #print "labels:",  labels_t.shape

    input_coord_t = input_coord_t.to(device=device)
    input_feats_t = input_feats_t.to(device=device)
    labels_t      = labels_t.to(device=device)
    weights_t     = weights_t.to(device=device)
    batchsize_t   = batchsize_t.to(device=device)
    #print input_coord_t.is_cuda,input_feats_t.is_cuda,labels_t.is_cuda,weights_t.is_cuda
    if False:
        print "input_coord_t: ",input_coord_t.shape
        print "input_feats_t: ",input_feats_t.shape
        print "labels_t: ",labels_t.shape
        print "weights_t: ",weights_t.shape                

    return input_coord_t, input_feats_t, labels_t, weights_t, batchsize_t
    

if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
