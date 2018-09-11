import numpy as np
from stats.optimizer import *
import sys

# Flowline
flowline = sys.argv[1]
# Input dictionary
inputs = {}
# Minimum error, first pass
min_err1 = 1000.**2
# Maximum error, first pass
max_err1 = 5000.**2
# Minimum error, second pass
min_err2 = 5000.**2
# Maximum error, second pass
max_err2 = 50000.**2
# Observation skip to reduce computation time
skip = 3

                               
### South
#############################################################

Ls = np.array([424777.2650658561, 394942.08036138373, 332430.91816515941, 303738.49932773202, 296659.0156905292])

if flowline == 'south1':
    inputs['in_dir'] = 'transform/south1/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform/south1/opt1/')

    
### Center
#############################################################

Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725])

if flowline == 'center1':
    inputs['in_dir'] = 'transform/center1/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform/center1/opt1/')

if flowline == 'center2':
    inputs['in_dir'] = 'transform/center2/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform/center2/opt1/')

if flowline == 'center3':
    inputs['in_dir'] = 'transform/center3/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform/center3/opt1/')

if flowline == 'center4':
    inputs['in_dir'] = 'transform/center4/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform/center4/opt1/')

if flowline == 'center5':
    inputs['in_dir'] = 'transform/center5/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform/center5/opt1/')

if flowline == 'center6':
    inputs['in_dir'] = 'transform/center6/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform/center6/opt1/')

if flowline == 'center7':
    inputs['in_dir'] = 'transform/center7/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform/center7/opt1/')

if flowline == 'center8':
    inputs['in_dir'] = 'transform/center8/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err2, max_err = max_err2, out_dir = 'transform/center8/opt2/')

    

### North
#############################################################

Ls = np.array([443746.66897917818, 397822.86008538032, 329757.49741948338, 292301.29712071194, 285478.05793305294])

if flowline == 'north1':
    inputs['in_dir'] = 'transform/north1/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip,  min_err = min_err1, max_err = max_err1, out_dir = 'transform/north1/opt1/')
