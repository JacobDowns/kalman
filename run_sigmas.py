from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import os.path

# Directory to read stuff from 
in_dir = 'filter/prior5/'
# Integer index
index = int(sys.argv[1])
print "index", index
# index = 0
runs = 22
# Load sigma points
sigma_points = np.loadtxt(in_dir + 'X.txt')
num_sigma_points = sigma_points.shape[0]
# Load sigma times
sigma_ts = np.loadtxt(in_dir + 'sigma_ts.txt')

# Run several delta temp. sigma points through the forward model
for i in range(index*runs, min(num_sigma_points, index*runs + runs)):
    print i

    if not os.path.isfile(in_dir + 'Y_' + str(i) + '.txt'):
        # Interpolated delta temp
        model_ts = -11.6e3 + np.linspace(0., 4300., 4300*3)
        Y_i = sigma_points[i]
        delta_temp_interp = interp1d(sigma_ts, Y_i, kind = 'linear')
        delta_temps = delta_temp_interp(model_ts)

        dt = 1./3.
        model_inputs = PaleoInputs('paleo_inputs/is_paleo_11_6_steady.hdf5', dt = dt)
        model = ForwardIceModel(model_inputs, "out", "paleo")

        N = 4300*3
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
