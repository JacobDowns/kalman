import numpy as np
from model.transient_runner import *
import matplotlib.pyplot as plt
import sys

"""
Perform a transient sensitivity run with given name and index.
"""

### Model inputs
#######################################################

# Input dictionary
inputs = {}
# Input file name
inputs['in_file'] = 'paleo_inputs/center_new_steady.h5'
# Time step
inputs['dt'] = 1./3.
# Number of model time steps
inputs['N'] = 4300*3


### Delta temp. function
#######################################################

data = np.loadtxt('paleo_data/jensen_dye3.txt')
# Years before present (2000)
years = data[:,0] - 2000.0
# Temps. in K
temps = data[:,1]
# Interp. delta temp. 
inputs['delta_temp_func'] = interp1d(years, temps - temps[-1], kind = 'linear')


### Precip param. function
#######################################################

# Interp. precip param. 
inputs['precip_param_func'] = interp1d(years, np.ones(len(years)), kind = 'linear')


### Perform model run 
#######################################################

model_runner = TransientRunner(inputs)
ages, Ls, Hs = model_runner.run()

np.savetxt('paleo_runs/run_test/age.txt', ages)
np.savetxt('paleo_runs/run_test/L.txt', Ls)
np.savetxt('paleo_runs/run_test/H.txt', Hs)
