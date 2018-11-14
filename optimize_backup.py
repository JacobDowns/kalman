import numpy as np
from stats.optimizer import *
import sys

# Flowline
flowline = sys.argv[1]
# Input dictionary
inputs = {}
    
### Center
#############################################################

Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686])

if flowline == 'center1_seasonal':
    inputs['in_dir'] = 'transform_long/center1_seasonal/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform_long/center1_seasonal/opt1/')

if flowline == 'center2_seasonal':
    inputs['in_dir'] = 'transform_long/center2_seasonal/'
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_long/center2_seasonal/opt1/')

    
### South
#############################################################

Ls = [424777.2650658561, 394942.08036138373, 332430.9181651594, 303738.499327732, 296659.0156905292, 284686.5963970118]
    
if flowline == 'south1_seasonal':
    inputs['in_dir'] = 'transform_long/south1_seasonal/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform_long/south1_seasonal/opt1/')

if flowline == 'south2_seasonal':
    inputs['in_dir'] = 'transform_long/south2_seasonal/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err2, max_err = max_err2, out_dir = 'transform_long/south2_seasonal/opt1/')