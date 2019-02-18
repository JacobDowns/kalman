import numpy as np
from scipy.interpolate import interp1d
from stats.kalman_update_sensitivity import *
import matplotlib.pyplot as plt

class Optimizer(object):

    def __init__(self, input_dict):

        ### Load stuff we need for the unscented transform 
        #############################################################

        # Input dictionary
        self.input_dict = input_dict
        # Input directory 
        self.in_dir = input_dict['in_dir']
        # Model time steps 
        self.model_ages = np.loadtxt(self.in_dir + 'age_0.txt')
        # Sigma points
        self.X = np.loadtxt(self.in_dir + 'X.txt')
        # Sigma points run through 
        self.Y = np.loadtxt(self.in_dir + 'Y.txt')
        # Prior mean
        self.m = np.loadtxt(self.in_dir + 'prior_m.txt')
        # Prior covariance 
        self.P = np.loadtxt(self.in_dir + 'prior_P.txt')
        # Load mean weights
        self.m_weights = np.loadtxt(self.in_dir + 'm_weights.txt')
        # Load covariance weights
        self.c_weights = np.loadtxt(self.in_dir + 'c_weights.txt')
        # Load sigma times
        self.sigma_ts = np.loadtxt(self.in_dir + 'sigma_ts.txt')
        self.N1 = len(self.sigma_ts)
        # Load sensitivity parameter names
        self.sensitivity_params = np.loadtxt(self.in_dir + 'sensitivity_params.txt', dtype = str)
        self.N2 = len(self.sensitivity_params)

    
        ### Use a specially built measurement mean and covariance matrix
        #############################################################
        
        # Measurement ages
        self.meas_ages = input_dict['y_ages']        
        # Measurement mean
        self.meas_y = input_dict['y']
        dt = int(self.meas_ages[1] - self.meas_ages[0])
        # Measurement covariance
        self.meas_Py = input_dict['Py']
        # Measurement indexes
        self.meas_indexes = list(range(0, len(self.model_ages), dt*3))
        # Restrict transformed sigma points
        self.Y = self.Y[:,self.meas_indexes]

        plt.plot(np.sqrt(self.meas_Py[range(len(self.meas_Py)), range(len(self.meas_Py))]))
        plt.show()
        
        
    # Do the Kalman update to incorporate the measurement and correct the prior mean
    def optimize(self, out_dir = None):

        ### Do the Kalman update
        #############################################################

        # The vector to condition on
        y_full  = np.zeros(len(self.meas_y) + 6)
        y_full[0:6] = self.m[-6:]
        y_full[6:] = self.meas_y
        ku = KalmanUpdate(self.m, self.P, self.X, self.Y, self.m_weights, self.c_weights)
        m_p, P_p, mu, K = ku.update(y_full, self.meas_Py)
        
        # Variance
        v = np.diag(P_p)

        if out_dir:
            np.savetxt(out_dir + 'mu.txt', mu)
            np.savetxt(out_dir + 'opt_m.txt', m_p)
            np.savetxt(out_dir + 'opt_P.txt', P_p)
            #np.savetxt(out_dir + 'y.txt', y)
            #np.savetxt(out_dir + 'R.txt', R)
            np.savetxt(out_dir + 'v.txt', v)

            plt.plot(m_p)
            plt.plot(m_p + 2.0*np.sqrt(v))
            plt.plot(m_p - 2.0*np.sqrt(v))
            plt.show()

        return m_p, P_p, mu, K
