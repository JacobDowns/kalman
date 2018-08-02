from model.inputs.paleo_inputs1 import *
from model.forward_model.forward_ice_model import *
import numpy as np
import sys

class SigmaRunner(object):

    def __init__(self, input_dict):
        
        # Directory to read stuff from 
        in_dir = input_dict['in_dir']
        # Integer index
        index = input_dict['index']
        # Number of runs
        runs = input_dict['runs']
        # Load sigma points
        sigma_points = np.loadtxt(in_dir + 'X.txt')
        num_sigma_points = sigma_points.shape[0]
        # Load sigma times
        sigma_ts = np.loadtxt(in_dir + 'sigma_ts.txt')
        # Model input file
        input_file = 'paleo_inputs/center_paleo_steady_11_6.hdf5'
        # Time step
        if 'dt' in input_dict:
            dt = input_dict['dt']
        else :
            dt = 1./3.

        # Run several delta temp. sigma points through the forward model
        for i in range(index*runs, min(num_sigma_points, index*runs + runs)):
            print i

            if not os.path.isfile(in_dir + 'Y_' + str(i) + '.txt'):
                # Interpolated delta temp
                model_ts = -11.6e3 + np.linspace(0., 4300., 4300*3)
                Y_i = sigma_points[i]
                delta_temp_interp = interp1d(sigma_ts, Y_i, kind = 'linear')
                delta_temps = delta_temp_interp(model_ts)

                model_inputs = PaleoInputs(input_file, dt)
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
