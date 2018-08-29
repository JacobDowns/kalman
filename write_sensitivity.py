from stats.prior_writer import PriorWriter
import numpy as np

""" 
Write parameter sets for sensitivity tests.
"""

ratio = 1./5.

### Ice hardness
############################################################################
spy = 60**2*24*365
# -5 C
b0 = (9e-25 * spy)**(-1./3.)
# -10 C
b1 = (3.5e-25 * spy)**(-1./3.)
# -15 C
b2 = (2.1e-25 * spy)**(-1./3.)
params0 = [b0, 5.5, 0.005, 0.008, 0.07, 0.75, 1e-3]
params1 = [b1, 5.5, 0.005, 0.008, 0.07, 0.75, 1e-3]
params2 = [b2, 5.5, 0.005, 0.008, 0.07, 0.75, 1e-3]
param_sets = [params0, params1, params2]
np.savetxt('sensitivity/test_hardness/param_sets.txt', param_sets)


### PDD Variance test
############################################################################
params0 = [b2, 5. , 0.005, 0.008, 0.07, 0.75, 1e-3]
params1 = [b2, 5.5, 0.005, 0.008, 0.07, 0.75, 1e-3]
param_sets = [params0, params1]
np.savetxt('sensitivity/test_pdd_var/param_sets.txt', param_sets)

    
### Lambda snow test
############################################################################
params0 = [b2, 5.5, 0.005 - ratio*0.005, 0.008, 0.07, 0.75, 1e-3]
params1 = [b2, 5.5, 0.005              , 0.008, 0.07, 0.75, 1e-3]
params2 = [b2, 5.5, 0.005 + ratio*0.005, 0.008, 0.07, 0.75, 1e-3]
param_sets = [params0, params1, params2]
np.savetxt('sensitivity/test_lambda_snow/param_sets.txt', param_sets)

    
### Lambda ice test
############################################################################
params0 = [b2, 5.5, 0.005, 0.008 - ratio*0.008, 0.07, 0.75, 1e-3]
params1 = [b2, 5.5, 0.005, 0.008              , 0.07, 0.75, 1e-3]
params2 = [b2, 5.5, 0.005, 0.008 + ratio*0.008, 0.07, 0.75, 1e-3]
param_sets = [params0, params1, params2]
np.savetxt('sensitivity/test_lambda_ice/param_sets.txt', param_sets)


### Lambda precip test
############################################################################
params0 = [b2, 5.5, 0.005, 0.008, 0.07, 0.75, 1e-3]
params1 = [b2, 5.5, 0.005, 0.008, 0.09, 0.75, 1e-3]
param_sets = [params0, params1]
np.savetxt('sensitivity/test_lambda_precip/param_sets.txt', param_sets)

    
### Water pressure test
############################################################################
params0 = [b2, 5.5, 0.005, 0.008, 0.07, 0.75 - ratio*0.75, 1e-3]
params1 = [b2, 5.5, 0.005, 0.008, 0.07, 0.75             , 1e-3]
params2 = [b2, 5.5, 0.005, 0.008, 0.07, 0.75 + ratio*0.75, 1e-3]
param_sets = [params0, params1, params2]
np.savetxt('sensitivity/test_p_frac/param_sets.txt', param_sets)
    

### Basal traction test
############################################################################
params0 = [b2, 5.5, 0.005, 0.008, 0.07, 0.75, 1e-3 - ratio*1e-3]
params1 = [b2, 5.5, 0.005, 0.008, 0.07, 0.75, 1e-3] 
params2 = [b2, 5.5, 0.005, 0.008, 0.07, 0.75, 1e-3 + ratio*1e-3]
param_sets = [params0, params1, params2]
np.savetxt('sensitivity/test_beta2/param_sets.txt', param_sets)
