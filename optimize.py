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
max_err1 = 2500.**2
# Minimum error, second pass
min_err2 = 5000.**2
# Maximum error, second pass
max_err2 = 50000.**2
# Observation skip to reduce computation time
skip = 5

                               
### South
#############################################################

Ls = np.array([424777.2650658561, 394942.08036138373, 332430.91816515941, 303738.49932773202, 296659.0156905292])

if flowline == 'south_prior1':
    inputs['in_dir'] = 'filter/south_prior1/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'filter/south_prior1/opt/')

if flowline == 'south_prior2':
    inputs['in_dir'] = 'filter/south_prior2/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, sparse_obs = True, min_err = min_err2, max_err = max_err2, out_dir = 'filter/south_prior2/opt/')
    
    
### Center
#############################################################

Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725])

if flowline == 'center_prior1':
    inputs['in_dir'] = 'filter/center_prior1/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'filter/center_prior1/opt/')

if flowline == 'center_prior2':
    inputs['in_dir'] = 'filter/center_prior2/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, sparse_obs = True, min_err = min_err2, max_err = max_err2, out_dir = 'filter/center_prior2/opt/')


### North 
#############################################################

Ls = np.array([443746.66897917818, 397822.86008538032, 329757.49741948338, 292301.29712071194, 285478.05793305294])

if flowline == 'north_prior1':
    inputs['in_dir'] = 'filter/north_prior1/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip,  min_err = min_err1, max_err = max_err1, out_dir = 'filter/north_prior1/opt/')

if flowline == 'north_prior2':
    inputs['in_dir'] = 'filter/north_prior2/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, sparse_obs = True, min_err = min_err2, max_err = max_err2, out_dir = 'filter/north_prior2/opt/')
