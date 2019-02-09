import numpy as np
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints
import matplotlib.pyplot as plt

class PriorWriter(object):

    def __init__(self, inputs):

        # Output directory
        out_dir = inputs['out_dir']
        # Times 
        sigma_ts = inputs['sigma_ts']
        np.savetxt(out_dir + 'sigma_ts.txt', sigma_ts)
        # Prior mean
        x = inputs['x']
        # Length of state vector
        n = len(x)
        # Covariance 
        Pxx = inputs['Pxx']

        
        ### Sensitivity param. mean and covariance
        ###########################################################################

        # Parameter names
        param_names = np.array(['beta2', 'lambda_precip', 'lambda_ice', 'lambda_snow', 'P_frac'])
        # Parameter mean values
        x_p = np.array([1.6e-3, 0.07, 0.008, 0.005, 0.85])
        # Inverse parameter variances 
        param_vars = np.diag(np.array([.25e-3**2, 0.02**2, 0.001**2, 0.001**2, .025**2]))
        # Number of params.
        n_p = len(param_names)
        np.savetxt(out_dir + 'sensitivity_params.txt', param_names, fmt="%s")
            

        ### Prior mean
        ###########################################################################

        n_a = n + n_p
        x_a = np.zeros(n_a)
        x_a[0:n] = x
        x_a[n:] = x_p
        

        ### Prior covariance 
        ##########################################################################

        # Inverse covariance
        P_a = np.zeros((n_a, n_a))
        P_a[0:n, 0:n] = Pxx
        P_a[n:, n:] = param_vars
        
        # Save prior mean and covariance
        np.savetxt(out_dir + 'prior_m.txt', x_a)
        np.savetxt(out_dir + 'prior_P.txt', P_a)

       
        ### Plot samples from prior
        ##########################################################################

        samples = np.random.multivariate_normal(x_a, P_a, 20)
        for i in range(samples.shape[0]):
            plt.plot(samples[i][0:n])
        plt.show()
        
        
        ### Compute sigma points
        ##########################################################################

        # Generate Julier sigma points
        points = JulierSigmaPoints(n_a, kappa=-n)
        sigma_points = points.sigma_points(x_a, P_a)
      
        # Save the mean and covariance weights, as well as the sigma points
        np.savetxt(out_dir + 'm_weights.txt', points.Wm)
        np.savetxt(out_dir + 'c_weights.txt', points.Wc)
        np.savetxt(out_dir + 'X.txt', sigma_points)

        for i in range(len(sigma_points)):
            plt.plot(sigma_points[i][0:n])
            print(sigma_points[i][n:])

        plt.show()

     
