import numpy as np
import sys
from model.transient_runner import *
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
# DJ or Buizert ?
dt_forcing = sys.argv[2]
# Steady state file name
inputs['in_file'] = in_dir + '/steady.h5'
# Time step
inputs['dt'] = 1./6.
# Number of model time steps
inputs['N'] = 4300*3


### Delta temp. function
#######################################################

if dt_forcing == 'dj':
    data = np.loadtxt('paleo_data/jensen_dye3.txt')
    years = data[:,0] - 2000.0
    temps = data[:,1]
else :
    data = np.loadtxt('paleo_data/buizert_dye3.txt')
    years = -data[:,0][::-1]
    temps = data[:,1][::-1]
    
inputs['delta_temp_func'] = interp1d(years, temps - temps[-1], kind = 'linear')


### Perform the model run
#######################################################
tr = TransientRunner(inputs)
ages, Ls, Hs, Ps = tr.run()

np.savetxt(in_dir + 'age.txt', ages)
np.savetxt(in_dir + 'L.txt', Ls)
np.savetxt(in_dir + 'H.txt', Hs)
np.savetxt(in_dir + 'p.txt', Ps)

