import numpy as np
from stats.optimizer_sensitivity import *

# Input dictionary
inputs = {}
# Minimum error, second pass
min_err = 10000.**2
# Maximum error, second pass
max_err = 100000.**2
# Observation skip to reduce computation time
skip = 3

Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686])

inputs['in_dir'] = 'transform_long/sensitivity/'
opt = Optimizer(inputs)
opt.optimize(Ls, skip = skip, min_err = min_err, max_err = max_err, out_dir = 'transform_long/sensitivity/opt1/')
