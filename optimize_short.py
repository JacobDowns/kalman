import numpy as np
from stats.optimizer import *
import sys

# Flowline
flowline = sys.argv[1]
# Input dictionary
inputs = {}
    
### Center
#############################################################

if flowline == 'center1': 
    inputs['in_dir'] = 'transform_short/center1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')[0:172]
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')[0:172]
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_c.txt')[0:172,0:172]
    opt = Optimizer(inputs)
    #print(inputs['y_ages'][0:173])
    #quit()
    opt.optimize(out_dir = 'transform_short/center1/opt1/')
