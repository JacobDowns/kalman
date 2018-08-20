import numpy as np
import sys
from model.paleo_runner import *
import matplotlib.pyplot as plt

# Integer index
index = int(sys.argv[1])
# Number of runs
runs = 9

### Model inputs
#######################################################

# Input dictionary
inputs = {}
# Steady state file name
inputs['in_file'] = 'paleo_inputs/center_paleo_steady_11_6.h5'
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


### Perform model
#######################################################

# Number of sensitivity experiments
num_experiments = 324

# Run several delta temp. sigma points through the forward model
for i in range(index*runs, min(num_experiments, index*runs + runs)):

    print i

    # Load the sensitivity parameters
    params = np.loadtxt('sensitivity/params_' + str(i) + '.txt')

    inputs['pdd_var'] = params[0]
    inputs['lambda_snow'] = params[1]
    inputs['lambda_ice'] = params[2]
    inputs['lambda_precip'] = params[3]
    inputs['P_frac'] = params[4]
    inputs['beta2'] = params[5]

    model_runner = PaleoRunner(inputs)
    ages, Ls = model_runner.run()

    # Save the results 
    np.savetxt('sensitivity/ages_' + str(i) + '.txt', ages)
    np.savetxt('sensitivity/Ls_' + str(i) + '.txt', Ls)
