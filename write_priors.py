from stats.prior_writer import PriorWriter
import numpy as np
import sys
from pyamg.gallery import poisson

inputs = {}

# Directory to write prior
inputs['out_dir'] = sys.argv[1]
# Sigma point times
N = 44
sigma_ts = np.linspace(-11554., 1., N)
inputs['sigma_ts'] = sigma_ts
# Set prior
#inputs['x'] = 0.0*np.ones(N)
#chi = np.linspace(0., 1., N)
#inputs['x'] = 0.5*(1. - chi)
inputs['x'] = np.loadtxt('transform_start/center1/opt1/opt_m.txt')
# Prior precision matrix
delta = 20e3
Q = delta*np.asarray(poisson((N,)).todense())
# Prior covariance
inputs['Px'] = np.linalg.inv(Q)
# The first weight for tuning
inputs['w_0'] = 0.5
                        
    
pw = PriorWriter(inputs)
