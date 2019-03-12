import numpy as np
from stats.optimizer import *
import sys

# Flowline
flowline = sys.argv[1]
# Input dictionary
inputs = {}
    
### Center
#############################################################

if flowline == 'center': 
    inputs['in_dir'] = 'transform_fifth/center/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    #print(inputs['y_ages'][0:173])
    #quit()
    opt.optimize(out_dir = 'transform_fifth/center/opt1/')
