import numpy as np
from scipy.interpolate import interp1d
from kalman_update import *


class Optimizer(object):

    def __init__(self, input_dict):

        ### Load stuff we need for the unscented transform 
        #############################################################

        # Input dictionary
        self.input_dict = input_dict
        # Input directory 
        self.in_dir = input_dict['in_dir']
        # Observed ages 
        self.obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
        # Model time steps 
        self.model_ages = np.loadtxt(self.in_dir + 'ages_0.txt')
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


    # Do the Kalman update to incorporate the measurement and correct the prior mean
    def optimize(self, obs_Ls, sparse_obs = False, skip = 5, min_err = 5000.**2, max_err = 50000.**2, out_dir = None):

        ### Generate observation vector
        #############################################################

        obs_indexes = range(len(self.model_ages))[::skip]
        if sparse_obs:
            # Include only the real observations (don't linearly interpolate)
            obs_indexes = [abs(self.model_ages - obs_age).argmin() for obs_age in obs_ages]

        # Interpolate the observations 
        L_interp = interp1d(self.obs_ages, obs_Ls, kind = 'linear')
        y = L_interp(self.model_ages[obs_indexes])


        ### Generate measurement covariance matrix
        #############################################################

        # Assumed observation covariance
        R = np.zeros((len(y), len(y)))

        # Set the error through time
        error_ts = np.array([-11.6, -10.9, -10.2, -9.7, -9.2, -8.7, -8.2, -7.75, -7.3])*1e3
        error_vs = np.array([min_err,  max_err,  min_err,   max_err,  min_err,   max_err,  min_err,   max_err,  min_err])
        error_interp = interp1d(error_ts, error_vs, kind = 'linear')
        errors = error_interp(self.model_ages[obs_indexes])
        R[range(R.shape[0]), range(R.shape[0])] = errors


        ### Do the Kalman update
        #############################################################
        ku = KalmanUpdate(self.m, self.P, self.X, self.Y[:,obs_indexes], self.m_weights, self.c_weights, obs_indexes)
        m_p, P_p, mu, K = ku.update(y, R)

        if out_dir:
            np.savetxt(out_dir + 'mu.txt', mu)
            np.savetxt(out_dir + 'opt_m.txt', m_p)
            np.savetxt(out_dir + 'opt_P.txt', P_p)
            np.savetxt(out_dir + 'y.txt', y)
            np.savetxt(out_dir + 'R.txt', R)

        return m_p, P_p, mu, K, y, R

    
inputs = {}
inputs['in_dir']  = 'filter/north_prior2/'
opt = Optimizer(inputs)
opt.optimize(np.array([424777.2650658561, 394942.08036138373, 332430.91816515941, 303738.49932773202, 296659.0156905292]), out_dir = 'filter/north_prior2/opt/')


                               

