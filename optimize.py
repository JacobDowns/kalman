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
max_err2 = 60000.**2
# Observation skip to reduce computation time
skip = 3
    
    
# 1.02
### Center
#############################################################

Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686])

if flowline == 'center1':
    inputs['in_dir'] = 'transform_long/center1/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform_long/center1/opt1/')

if flowline == 'center1_seasonal':
    inputs['in_dir'] = 'transform_long/center1_seasonal/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform_long/center1_seasonal/opt1/')

if flowline == 'center1_opt2':
    inputs['in_dir'] = 'transform_long/center1/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = 10000.**2, out_dir = 'transform_long/center1/opt2/')

if flowline == 'center2':
    inputs['in_dir'] = 'transform_long/center2/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err2, max_err = max_err2, out_dir = 'transform_long/center2/opt1/')

if flowline == 'center2_seasonal':
    inputs['in_dir'] = 'transform_long/center2_seasonal/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err2, max_err = max_err2, out_dir = 'transform_long/center2_seasonal/opt1/')

if flowline == 'center3':
    inputs['in_dir'] = 'transform_long/center3/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err2, max_err = max_err2, out_dir = 'transform_long/center3/opt1/')

if flowline == 'center4_opt2':
    inputs['in_dir'] = 'transform_long/center4/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err2, max_err = max_err2, out_dir = 'transform_long/center4/opt2/')

if flowline == 'center4_opt3':
    inputs['in_dir'] = 'transform_long/center4/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err2, max_err = max_err2, out_dir = 'transform_long/center4/opt3/')

if flowline == 'beta2_0':
    inputs['in_dir'] = 'transform_long/beta2_0/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform_long/beta2_0/opt1/')

if flowline == 'beta2_1':
    inputs['in_dir'] = 'transform_long/beta2_1/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform_long/beta2_1/opt1/')

if flowline == 'P_frac_0':
    inputs['in_dir'] = 'transform_long/P_frac_0/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform_long/P_frac_0/opt1/')

if flowline == 'P_frac_1':
    inputs['in_dir'] = 'transform_long/P_frac_1/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform_long/P_frac_1/opt1/')

if flowline == 'lambda_ice_0':
    inputs['in_dir'] = 'transform_long/lambda_ice_0/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform_long/lambda_ice_0/opt1/')

if flowline == 'lambda_ice_1':
    inputs['in_dir'] = 'transform_long/lambda_ice_1/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform_long/lambda_ice_1/opt1/')

if flowline == 'south1_seasonal':
    inputs['in_dir'] = 'transform_long/south1_seasonal/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err1, max_err = max_err1, out_dir = 'transform_long/south1_seasonal/opt1/')

if flowline == 'south2_seasonal':
    inputs['in_dir'] = 'transform_long/south2_seasonal/'
    opt = Optimizer(inputs)
    opt.optimize(Ls, skip = skip, min_err = min_err2, max_err = max_err2, out_dir = 'transform_long/south2_seasonal/opt1/')
