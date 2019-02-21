from __future__ import print_function
import os,sys
import numpy as np
import torch

if '../' not in sys.path:
    sys.path.append('../')
from models.spuresnet import SparseUResNet

#checkpointfile = sys.argv[1]
checkpointfile = "checkpoint.60000th.tar"

# load model
model = SparseUResNet( (512,512), 2, 32, 5, 3 )
print(model)

# load statedict
#checkpoint = torch.load( checkpointfile )
#model.load_state_dict(checkpoint["state_dict"])

# fake input
fakecoords_t = torch.zeros( (10,3), dtype=torch.long )
fakepixval_t = torch.zeros( (10, 1), dtype=torch.float )
fakebatch_t  = torch.zeros( (1), dtype=torch.long )

print( fakecoords_t.shape )
print( fakepixval_t.shape )
print( fakebatch_t.shape  )


# trace the net
traced_script_module = torch.jit.trace( model, ( (fakecoords_t, fakepixval_t, fakebatch_t) ) )
traced_script_module.save( "sparse_uresnet.pt" )


