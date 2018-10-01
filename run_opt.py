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
# Optimization results directory
opt_dir = sys.argv[2]
# Steady state file name
inputs['in_file'] = in_dir + '/steady.h5'
# Time step
inputs['dt'] = 1./3.
# Number of model time steps
inputs['N'] = 11580*3


### Delta temp. function
#######################################################
data = np.loadtxt('paleo_data/buizert_dye3.txt')
years = -data[:,0][::-1]
temps = data[:,1][::-1]
inputs['delta_temp_func'] = interp1d(years, temps - temps[-1], kind = 'linear')


### Precip param. file
#######################################################

# State vector times
sigma_ts = np.loadtxt(in_dir + '/sigma_ts.txt')
precip_param_opt = np.loadtxt(in_dir + '/' + opt_dir  + '/opt_m.txt')
v = np.loadtxt(in_dir + '/' + opt_dir  + '/v.txt')
# Interpolated delta temp. function 
inputs['precip_param_func'] = interp1d(sigma_ts, precip_param_opt, kind = 'linear')


plt.plot(sigma_ts, precip_param_opt)
plt.plot(sigma_ts, precip_param_opt + 2.*np.sqrt(v))
plt.plot(sigma_ts, precip_param_opt - 2.*np.sqrt(v))
plt.show()
#quit()

### Perform the model run
#######################################################
tr = TransientRunner(inputs)
ages, Ls, Hs, Ps = tr.run()

np.savetxt(in_dir + '/' + opt_dir + '/opt_age.txt', ages)
np.savetxt(in_dir + '/' + opt_dir + '/opt_L.txt', Ls)
np.savetxt(in_dir + '/' + opt_dir + '/opt_H.txt', Hs)
np.savetxt(in_dir + '/' + opt_dir + '/opt_p.txt', Ps)

