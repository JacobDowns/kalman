import sys
sys.path.append('../')
from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import os.path
import matplotlib

matplotlib.rcParams.update({'font.size': 20})
#fig = plt.figure(figsize=(12,14))

# Plot a steady state profile
def plot_steady(name, index, bed = False, surface = True, label = ''):
    dt = 1./3.
    model_inputs = PaleoInputs('../sensitivity/' +  name + '/steady_' + str(index) + '.h5', {})
    model = ForwardIceModel(model_inputs, "out", "paleo")

    xs = model_inputs.mesh.coordinates() * model_inputs.L_init
    L_init = model_inputs.L_init

    surface = project(model.S)
    bed = project(model.B)

    
    if bed:
        plt.plot(xs, surface.compute_vertex_values(), linewidth = 3.5, label = str(index))
    if surface:
        plt.plot(xs, bed.compute_vertex_values(), 'k', linewidth = 3.5)

plot_steady('test_p_frac', 0, bed = True)
plot_steady('test_p_frac', 1)
plot_steady('test_p_frac', 2)
        
plot_steady('test_pdd_var', 0)
plot_steady('test_pdd_var', 1)

plot_steady('test_lambda_precip', 0)
plot_steady('test_lambda_precip', 1)

plot_steady('test_p_frac', 0)
plot_steady('test_p_frac', 1)
plot_steady('test_p_frac', 2)

plot_steady('test_lambda_ice', 0)
plot_steady('test_lambda_ice', 1)
plot_steady('test_lambda_ice', 2)

plot_steady('test_lambda_snow', 0)
plot_steady('test_lambda_snow', 1)
plot_steady('test_lambda_snow', 2)

plot_steady('test_lambda_ice', 0)
plot_steady('test_lambda_ice', 1)
plot_steady('test_lambda_ice', 2)

plot_steady('test_beta2', 0)
plot_steady('test_beta2', 1)
plot_steady('test_beta2', 2)

plt.legend()
plt.show()
