import numpy as np
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints
import matplotlib.pyplot as plt

class PriorWriter(object):

    def __init__(self, input_dict):

        # Output directory
        out_dir = input_dict['out_dir']

        ### Delta P mean
        ###########################################################################
        
        # Load delta P mean
        x1 = np.loadtxt(input_dict['x'])
        # Length of delta P
        N1 = len(x1)
        # Delta temp. grid years
        dt_years = -11.6e3 + np.linspace(0., 11590, N1)
        np.savetxt(out_dir + 'sigma_ts.txt', dt_years)

        
        ### Sensitivity param. mean and covariance
        ###########################################################################

        # Parameter names
        param_names = np.array(['beta2', 'A', 'lambda_precip', 'lambda_ice', 'lambda_snow', 'P_frac'])
        # Parameter mean values
        x2 = np.array([1.6e-3, 3.5e-25, 0.07, 0.008, 0.005, 0.85])
        # Inverse parameter variances 
        param_inv_vars = 1. / np.array([.25e-3**2, (1.5e-25)**2, 0.02**2, 0.001**2, 0.001**2, .025**2])
        # Number of params.
        N2 = 6
        np.savetxt(out_dir + 'sensitivity_params.txt', param_names, fmt="%s")
            

        ### Prior mean
        ###########################################################################

        N = N1 + N2
        x = np.zeros(N)
        x[0:N1] = x1
        x[N1:] = x2
        

        ### Prior covariance 
        ##########################################################################

        # Inverse covariance
        delta = input_dict['delta']
        Q = np.zeros((N, N))
        Q[range(N1), range(N1)] = 2.
        Q[range(1,N1), range(N1-1)] = -1.
        Q[range(N1-1), range(1,N1)] = -1.
        Q = delta*Q
        Q[range(N1,N), range(N1,N)] = param_inv_vars

        # Covariance
        P = np.linalg.inv(Q)
        
        # Save prior mean and covariance
        np.savetxt(out_dir + 'prior_m.txt', x)
        np.savetxt(out_dir + 'prior_P.txt', P)

       
        ### Plot samples from prior
        ##########################################################################

        samples = np.random.multivariate_normal(x, P, 20)
        for i in range(samples.shape[0]):
            plt.plot(samples[i][0:N1])
            #print(samples[i][N1:])
        #plt.show()
        
        
        ### Compute sigma points
        ##########################################################################

        # Generate Julier sigma points
        points = JulierSigmaPoints(N, kappa=3-N)
        sigma_points = points.sigma_points(x, P)
      
        # Save the mean and covariance weights, as well as the sigma points
        np.savetxt(out_dir + 'm_weights.txt', points.Wm)
        np.savetxt(out_dir + 'c_weights.txt', points.Wc)
        np.savetxt(out_dir + 'X.txt', sigma_points)

        for i in range(len(sigma_points)):
            plt.plot(sigma_points[i][0:N1])
            #print(sigma_points[i][N1:])

        """
        print(sigma_points[:,-1])
        print(sigma_points[:,-1].min())
        print(sigma_points[:,-1].max())

        j = -1
        for i in range(len(points.Wm)):
            if sigma_points[i, j] == sigma_points[:, j].min():
                print(i)"""
      
        #print(sigma_points[:, -6:].min(axis = 0))
        #print(sigma_points[:, -6:].max(axis = 0))

        sigma_points[:, -6:].argmin(axis = 0)
        sigma_points[:, -6:].argmax(axis = 0)

        print(sigma,points(
        #print(np.median(sigma_points[:, -6:], axis = 0))
        
        #plt.show()
