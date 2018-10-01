from stats.prior_writer import PriorWriter
import numpy as np

""" 
Write parameter sets for sensitivity tests.
"""

ratio = 1./4.
    
### Water pressure fracion test
############################################################################
params = [0.8, 0.9]
np.savetxt('sensitivity/P_frac/params.txt', params)
    
### Basal traction test
############################################################################
params = [1e-3 - ratio*1e-3, 1e-3 + ratio*1e-3]
np.savetxt('sensitivity/beta2/params.txt', params)

### Ice melt rate test
############################################################################
params = [0.008 - ratio*0.008, 0.008 + ratio*0.008]
np.savetxt('sensitivity/lambda_ice/params.txt', params)

### Snow melt rate test
############################################################################
params = [0.005 - ratio*0.005, 0.005 + ratio*0.005]
np.savetxt('sensitivity/lambda_snow/params.txt', params)
