import numpy as np
import sys
from model.transient_runner import *
from scipy.interpolate import interp1d

""" 
Perform a model run with a delta P of 0.
"""

### Model inputs
#######################################################

"""
# Input dictionary
inputs = {}
# Input file name
inputs['in_file'] = 'paleo_inputs/north_steady_seasonal.h5'
out_dir = 'paleo_runs/north_seasonal/'
# Time step
inputs['dt'] = 1./3.
# Number of model time steps
inputs['N'] = 4300*3
# Tuned basal traction
inputs['beta2'] = 1.6e-3
"""

# Input dictionary
inputs = {}
# Input file name
inputs['in_file'] = 'paleo_inputs/north_steady.h5'
out_dir = 'paleo_runs/north_seasonal/'
# Time step
inputs['dt'] = 1./3.
# Number of model time steps
inputs['N'] = 4300*3
# Tuned basal traction
inputs['beta2'] = 1.6e-3


### Perform the model run
#######################################################
tr = TransientRunner(inputs)
ages, Ls, Hs, Ps = tr.run()

quit()


np.savetxt(out_dir + 'age.txt', ages)
np.savetxt(out_dir + 'L.txt', Ls)
np.savetxt(out_dir + 'H.txt', Hs)
np.savetxt(out_dir + 'P.txt', Ps)
