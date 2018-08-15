import numpy as np
import sys
from model.paleo_runner import *
from scipy.interpolate import interp1d

""" 
Perform a model run with an optimized delta temp.
"""

### Model inputs
#######################################################

# Input dictionary
inputs = {}
# Input directory
in_dir = sys.argv[1]
# Optimization results directory
opt_dir = sys.argv[2]
# Steady state file name
inputs['in_file'] = in_dir + '/steady.h5'
# Time step
inputs['dt'] = 1./3.
# Number of model time steps
inputs['N'] = 4300*3


### Delta temp. function
#######################################################

# State vector times
sigma_ts = np.loadtxt(in_dir + '/sigma_ts.txt')
delta_temps_opt = np.loadtxt(in_dir + '/' + opt_dir  + '/opt_m.txt')
# Interpolated delta temp. function 
inputs['delta_temp_func'] = interp1d(sigma_ts, delta_temps_opt, kind = 'linear')


### Perform the model run
#######################################################
model_runner = PaleoRunner(inputs)
ages, Ls = model_runner.run()

np.savetxt(in_dir + '/' + opt_dir + '/opt_ages.txt', ages)
np.savetxt(in_dir + '/' + opt_dir + '/opt_Ls.txt', Ls)
