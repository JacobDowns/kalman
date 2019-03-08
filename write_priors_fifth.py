from stats.prior_writer import PriorWriter
import numpy as np
import sys
from pyamg.gallery import poisson
import matplotlib.pyplot as plt

inputs = {}

# Directory to write prior
inputs['out_dir'] = sys.argv[1]
# Sigma point times
N = 44
sigma_ts = np.linspace(-11554., 0., N)
inputs['sigma_ts'] = sigma_ts
# Set prior
#inputs['x'] = 0.2*np.ones(N)
chi = np.linspace(0., 1., N)
#inputs['x'] = (0.33*(1.-chi)*chi+0.2)[0:17]
inputs['x'] = np.loadtxt('transform_dj_seasonal/center2/opt1/opt_m.txt')
# Prior precision matrix
delta = 5e3
Q = delta*np.asarray(poisson((N,)).todense())
# Prior covariance
inputs['Pxx'] = np.linalg.inv(Q)
inputs['w0'] = 3.

pw = PriorWriter(inputs)
