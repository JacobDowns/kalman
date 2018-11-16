import numpy as np
from scipy.interpolate import interp1d
from stats.kalman_update import *
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
        # Transformed sigma points
        self.Y = np.loadtxt(self.in_dir + 'Y.txt')
        # Prior mean
        self.m = np.loadtxt(self.in_dir + 'prior_m.txt')
        # Prior covariance 
        self.P = np.loadtxt(self.in_dir + 'prior_P.txt')
        # Load mean weights
        self.m_weights = np.loadtxt(self.in_dir + 'm_weights.txt')
        # Load covariance weights
        self.c_weights = np.loadtxt(self.in_dir + 'c_weights.txt')
        

        ### Use a specially build measurement mean and covariance matrix
        #############################################################
        
        # Measurement ages
        self.meas_ages = np.loadtxt('paleo_inputs/y_ages.txt')
        # Measurement mean
        self.meas_y = input_dict['y']
        # Measurement covariance
        self.meas_Py = input_dict['Py']
        # Measurement indexes
        self.meas_indexes = range(0, len(self.model_ages), 25*3)
        # Restrict transformed sigma points
        self.Y = self.Y[:,self.meas_indexes]

        plt.plot(np.sqrt(self.meas_Py[range(len(self.meas_Py)), range(len(self.meas_Py))]))
        plt.show()


    # Do the Kalman update to incorporate the measurement and correct the prior mean
    def optimize(self, out_dir = None):

        ### Do the Kalman update
        #############################################################
        ku = KalmanUpdate(self.m, self.P, self.X, self.Y, self.m_weights, self.c_weights)
        m_p, P_p, mu, K = ku.update(self.meas_y, self.meas_Py)

        # Variance
        v = P_p[range(len(P_p)), range(len(P_p))]

        if out_dir:
            np.savetxt(out_dir + 'mu.txt', mu)
            np.savetxt(out_dir + 'opt_m.txt', m_p)
            np.savetxt(out_dir + 'opt_P.txt', P_p)
            np.savetxt(out_dir + 'y.txt', self.meas_y)
            np.savetxt(out_dir + 'Py.txt', self.meas_Py)
            np.savetxt(out_dir + 'v.txt', v)

            plt.plot(m_p)
            plt.plot(m_p + 2.0*np.sqrt(v))
            plt.plot(m_p - 2.0*np.sqrt(v))
            plt.show()

        return m_p, P_p, mu, K, self.meas_y, self.meas_Py
