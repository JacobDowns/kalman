from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import os.path

# Directory to read stuff from 
in_dir = 'filter/prior1/'
# Integer index
index = int(sys.argv[1])
print "index", index
# index = 0
runs = 43
# Load sigma points
sigma_points = np.loadtxt(in_dir + 'prior_sigma_points.txt')
num_sigma_points = sigma_points.shape[0]

# Run several delta temp. sigma points through the forward model
for i in range(index*runs, min(num_sigma_points, index*runs + runs)):
    print i

    if not os.path.isfile(in_dir + 'sigma_point_' + str(i) + '.txt'):
        delta_temps = np.repeat(sigma_points[i], 15)
        dt = 1./3.
        model_inputs = PaleoInputs('paleo_inputs/is_paleo_11_6_steady.hdf5', dt = dt)
        model = ForwardIceModel(model_inputs, "out", "paleo")

        N = 4295*3
        Ls = []
        ages = []

        for j in range(N):
            print model.t + dt
            age = -11.6e3 + model.t + dt
            ages.append(age)

            L = model.step(delta_temps[j], accept = True)
            Ls.append(L)

        np.savetxt(in_dir + 'sigma_point_' + str(i) + '.txt', delta_temps)
        np.savetxt(in_dir + 'ages_' + str(i) + '.txt', np.array(ages))
        np.savetxt(in_dir + 'Y_' + str(i) + '.txt', np.array(Ls))
