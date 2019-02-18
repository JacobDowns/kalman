import numpy as np
from stats.optimizer_sensitivity import *
import sys

inputs = {}
inputs['in_dir'] = 'transform_dj/center_sensitivity/'
inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_c.txt')
opt = Optimizer(inputs)
opt.optimize(out_dir = 'transform_dj/center_sensitivity/opt1/')
