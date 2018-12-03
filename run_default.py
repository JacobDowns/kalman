import numpy as np
from model.transient_runner import *

""" 
Perform a model run with a delta P = 0.
"""

### Model inputs
#######################################################

# Input dictionary
inputs = {}
# Input file name
inputs['in_file'] = 'paleo_inputs/center_steady_seasonal.h5'
out_dir = 'default_runs/center1/'
# Time step
inputs['dt'] = 1./3.
# Number of time steps (this is dumb)
inputs['N'] = 4400*3#35058
# Tuned basal traction
inputs['beta2'] = 1.6e-3


### Perform the model run
#######################################################
tr = TransientRunner(inputs)
ages, Ls, Hs, Ps = tr.run()

np.savetxt(out_dir + 'age.txt', ages)
np.savetxt(out_dir + 'L.txt', Ls)
np.savetxt(out_dir + 'H.txt', Hs)
np.savetxt(out_dir + 'P.txt', Ps)
