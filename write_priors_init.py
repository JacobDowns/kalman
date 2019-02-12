from stats.prior_writer import PriorWriter
import numpy as np
import sys
from pyamg.gallery import poisson

inputs = {}

# Directory to write prior
inputs['out_dir'] = sys.argv[1]
# Sigma point times
N = 10
sigma_ts = np.linspace(-12554., -11554.+1., N)
inputs['sigma_ts'] = sigma_ts
# Set prior
inputs['x'] = 0.0*np.ones(N)
#inputs['x'] = np.loadtxt('transform_init/south3/opt1/opt_m.txt')
# Prior precision matrix
delta = 1e3
Q = delta*np.asarray(poisson((N,)).todense())
# Prior covariance
inputs['Pxx'] = np.linalg.inv(Q)
# The first weight for tuning
inputs['w0'] = 0.5
                            
pw = PriorWriter(inputs)
