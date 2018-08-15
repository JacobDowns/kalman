from stats.prior_writer import PriorWriter
import numpy as np

ratio = 1./4.

# PDD variance
pdd_vars = [5., 5.5]
# Ablation rate snow
lambda_ss = [0.005 - ratio*0.005, 0.005, 0.005 + ratio*0.005] 
# Ablation rate ice
lambda_is = [0.008 - ratio*0.008, 0.008, 0.008 + ratio*0.008]
# Precip. param.
lambda_ps = [0.07, 0.09]
# Water pressure
P_fracs = [0.6, 0.75, 0.9]
# Basal traction
beta2s = [1e-3 - ratio*1e-3, 1e-3, 1e-3 + ratio*1e-3]

params = [pdd_vars, lambda_ss, lambda_is, lambda_ps, P_fracs, beta2s]

import itertools
i = 0
things = []
for element in itertools.product(*params):
    np.savetxt('sensitivity/params_' + str(i) + '.txt', np.array(element))
    things.append(np.array(element))
    i += 1
