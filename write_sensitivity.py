from stats.prior_writer import PriorWriter
import numpy as np

""" 
Write parameter sets for sensitivity tests.
"""

ratio = 1./4.
    
### Water pressure fracion test
############################################################################
params = [0.825, 0.875]
np.savetxt('sensitivity/P_frac/params.txt', params)
    
### Basal traction test
############################################################################
params = [1e-3 - ratio*1e-3, 1e-3 + ratio*1e-3]
np.savetxt('sensitivity/beta2/params.txt', params)
