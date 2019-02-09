from stats.prior_writer_sensitivity import PriorWriter
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
inputs['x'] = np.loadtxt('transform_final/center3/opt1/opt_m.txt')
# Prior precision matrix
delta = 7.5e3
Q = delta*np.asarray(poisson((N,)).todense())
# Prior covariance
inputs['Pxx'] = np.linalg.inv(Q)
    
pw = PriorWriter(inputs)
