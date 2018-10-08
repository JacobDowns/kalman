import numpy as np
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints
import matplotlib.pyplot as plt

class PriorWriter(object):

    def __init__(self, input_dict):

        # Output directory
        out_dir = input_dict['out_dir']

        #### Setup state vector mean
        ###########################################################################

        # Length of delta P 
        if 'N1' in input_dict:
            delta_P_len = input_dict['delta_P_len']
        else :
            delta_P_len = 45
        self.x = np.zeros(N1 + 3)
        
        
        ### Delta P mean
        ###########################################################################

        # Delta temp. grid years
        dt_years = -11.6e3 + np.linspace(0., 11590, N1)
        np.savetxt(out_dir + 'sigma_ts.txt', dt_years)

        
        ### Param. means
        ###########################################################################

        # Mean and variance for each parameter
        param_priors = {
             'beta2' : [1.6e-3, .2e-3**2],
             'P_frac' : [0.85, 0.025**2],
             'b' : [(3.5e-25*60**2*24*365)**(-1./3.), (0.75e-25*60**2*24*365)**(-1./3.)]
             'lambda_precip' : [0.07, 0.005**2],
             'lambda_ice' : [0.008, 0.0005**2],
             'lambda_snow' : [0.005, 0.0005**2] 
            }

        
        ### Define prior mean
        ###########################################################################
        
        if 'x' in input_dict:
            # Load a custom prior
            x = np.loadtxt(input_dict['x'])
        else :
            chi = np.linspace(0., 1., len(dt_years))
            x = 0.45*np.ones(len(dt_years)) - 0.45*(chi)**8


        ### Define prior covariance 
        ##########################################################################

        # Delta controls smoothness
        delta = 5000.
        if 'delta' in input_dict:
            delta = input_dict['delta']
            
        # Covariance matrix
        P = np.zeros((N, N))
        P[range(N), range(N)] = 2.
        P[range(1,N), range(N-1)] = -1.
        P[range(N-1), range(1,N)] = -1.
        P[N-1, N-1] = 2.
        P = delta*P
        P = np.linalg.inv(P)
        # Save prior mean and covariance
        np.savetxt(out_dir + 'prior_m.txt', x)
        np.savetxt(out_dir + 'prior_P.txt', P)
        

        ### Plot samples from prior
        ##########################################################################

        samples = np.random.multivariate_normal(x, P, 100)
        for i in range(samples.shape[0]):
            plt.plot(samples[i])
        plt.show()

        
        ### Compute sigma points
        ##########################################################################
        # Generate Julier sigma points
        points = JulierSigmaPoints(N, kappa=N)
        sigma_points = points.sigma_points(x, P)

        # Save the mean and covariance weights, as well as the sigma points
        np.savetxt(out_dir + 'm_weights.txt', points.weights()[0])
        np.savetxt(out_dir + 'c_weights.txt', points.weights()[1])
        np.savetxt(out_dir + 'X.txt', sigma_points)

        for i in range(len(sigma_points)):
            print i
            plt.plot(sigma_points[i])

        plt.show()
