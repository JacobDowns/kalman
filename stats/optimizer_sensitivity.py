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
        # Observed ages 
        self.obs_ages = np.array([-11.6e3, -10.2e3, -9.2e3, -8.2e3, -7.3e3, 0.])
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


    # Do the Kalman update to incorporate the measurement and correct the prior mean
    def optimize(self, obs_Ls, sparse_obs = False, skip = 5, min_err = 5000.**2, max_err = 50000.**2, out_dir = None):

        ### Generate observation vector
        #############################################################

        obs_indexes = range(len(self.model_ages))[::skip]
        if sparse_obs:
            # Include only the real observations (don't linearly interpolate)
            obs_indexes = [abs(self.model_ages - obs_age).argmin() for obs_age in self.obs_ages]
            
        # Interpolate the observations 
        L_interp = interp1d(self.obs_ages, obs_Ls, kind = 'linear')
        y = L_interp(np.round(self.model_ages[obs_indexes], 0))

        
        ### Generate measurement covariance matrix
        #############################################################

        # Assumed observation covariance
        R = np.zeros((len(y), len(y)))

        # Set the error through time
        dif = (self.obs_ages[1:] - self.obs_ages[:-1])
        dif /= dif.max()
        dif *= max_err
        error_ts = np.array([-11.6e3, -10.9e3, -10.2e3, -9.7e3, -9.2e3, -8.7e3, -8.2e3, -7.75e3, -7.3e3, -3650.0, 0.])
        #error_vs = np.array([min_err,  max_err,  min_err,   max_err,  min_err,   max_err,  min_err,   max_err,  min_err, max_err, min_err])
        error_vs = np.array([min_err,  dif[0],  min_err,   dif[1],  min_err,   dif[2],  min_err,   dif[3],  min_err, dif[4], min_err])
        error_interp = interp1d(error_ts, error_vs, kind = 'linear', bounds_error = False)
        errors = error_interp(np.round(self.model_ages[obs_indexes], 0))
        R[range(R.shape[0]), range(R.shape[0])] = errors
        

        ### Do the Kalman update
        #############################################################

        # The vector to condition on
        y_full  = np.zeros(len(y) + 6)
        y_full[0:6] = self.m[-6:]
        y_full[6:] = y
        ku = KalmanUpdate(self.m, self.P, self.X, self.Y[:,obs_indexes], self.m_weights, self.c_weights, obs_indexes)
        m_p, P_p, mu, K = ku.update(y_full, R)

        quit()
        

        # Variance
        v = P_p[range(len(P_p)), range(len(P_p))]

        if out_dir:
            np.savetxt(out_dir + 'mu.txt', mu)
            np.savetxt(out_dir + 'opt_m.txt', m_p)
            np.savetxt(out_dir + 'opt_P.txt', P_p)
            np.savetxt(out_dir + 'y.txt', y)
            #np.savetxt(out_dir + 'R.txt', R)
            np.savetxt(out_dir + 'v.txt', v)

            plt.plot(m_p)
            plt.plot(m_p + 2.0*np.sqrt(v))
            plt.plot(m_p - 2.0*np.sqrt(v))
            plt.show()

        return m_p, P_p, mu, K, y, R