import numpy as np
from scipy.interpolate import interp1d
from kalman_update import *


class Optimizer(object):

    def __init__(self, input_dict):
        
        # Input directory 
        in_dir = input_dict['in_dir']
        # Output directory
        out_dir = input_dict['out_dir']
        # Observed ages 
        obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
        # Observed lengths
        obs_Ls = input_dict['obs_Ls']
        # Interpolate the observations 
        L_interp = interp1d(obs_ages, obs_Ls, kind = 'linear')
        # Model time steps 
        model_ages = np.loadtxt(in_dir + 'ages_0.txt')

        
        ### Observations 
        #############################################################
        
        if 'sparse_obs' in input_dict and input_dict['sparse_obs']:
            # Include only the real observations (don't linearly interpolate)
            obs_indexes = [abs(model_ages - obs_age).argmin() for obs_age in obs_ages]
        elif 'skip' in input_dict:
            # Skip some of the observations to reduce computation time
            skip = input_dict['skip']
            obs_indexes = range(len(model_ages))[::skip]
        else:
            skip = 5
            obs_indexes = range(len(model_ages))[::skip]

        self.y = L_interp(model_ages[obs_indexes])


        ### Observation error
        #############################################################

        # Assumed observation covariance
        self.R = np.zeros((len(self.y), len(self.y)))
        
        # Minimum variance at observation points
        if 'min_err' in input_dict:
            min_err = input_dict['min_err']
        else:
            min_err = 5000.**2

        # Maximum variance between observation points
        if 'max_err' in input_dict:
            max_err = input_dict['max_err']
        else:
            max_err = 50000.**2

        # Set the error through time
        error_ts = np.array([-11.6, -10.9, -10.2, -9.7, -9.2, -8.7, -8.2, -7.75, -7.3])*1e3
        error_vs = np.array([min_err,  max_err,  min_err,   max_err,  min_err,   max_err,  min_err,   max_err,  min_err])
        error_interp = interp1d(error_ts, error_vs, kind = 'linear')
        errors = error_interp(model_ages[obs_indexes])
        self.R[range(self.R.shape[0]), range(self.R.shape[0])] = errors


        ### Load stuff we need for the unscented transform 
        #############################################################

        # Sigma points
        self.X = np.loadtxt(in_dir + 'X.txt')
        # Sigma points run through 
        self.Y = np.loadtxt(in_dir + 'Y.txt')
        self.Y = Y[:, obs_indexes]
        # Prior mean
        self.m = np.loadtxt(in_dir + 'prior_m.txt')
        # Prior covariance 
        self.P = np.loadtxt(in_dir + 'prior_P.txt')
        # Load mean weights
        self.m_weights = np.loadtxt(in_dir + 'm_weights.txt')
        # Load covariance weights
        self.c_weights = np.loadtxt(in_dir + 'c_weights.txt')


        ### Do Kalman update
        #############################################################

        # Object for doing the Kalman update computations
        self.ku = KalmanUpdate(self.m, self.P, self.X, self.Y, self.m_weights, self.c_weights, obs_indexes)
 

        plt.plot(m, 'ko')
        plt.plot(m_p, 'ro')
        plt.show()

        np.savetxt(in_dir + 'mu_error.txt', mu)
        np.savetxt(in_dir + 'opt_m_error.txt', m_p)
        np.savetxt(in_dir + 'opt_P_error.txt', P_p)

        np.savetxt('opt_m.txt', m_p)


    # Do the Kalman update to incorporate the measurement and correct the prior mean
    def update():
        m_p, P_p, mu, K = self.ku.update(self.y, self.R)
        # Return optimized mean, covariance, as well as 
        return m_p, P_p, mu, K

    
    # Do the Kalman update to incorporate the measurement and correct the prior mean. Also saves the
    # optimized mean and covariance
    def update_save():
        m_p, P_p, mu, K = self.ku.update(self.y, self.R)
        np.savetxt(out_dir + 'mu.txt', mu)
        np.savetxt(out_dir + 'opt_m.txt', m_p)
        np.savetxt(out_dir + 'opt_P.txt', P_p)
        np.savetxt(out_dir + 'y.txt', self.y)
        np.savetxt(out_dir + 'R.txt', self.R)
        

"""                               
# Observed lengths
obs_Ls = np.array([424777.2650658561, 394942.08036138373, 332430.91816515941, 303738.49932773202, 296659.0156905292])


# Model time steps 
model_ages = np.loadtxt(in_dir + 'ages_0.txt')
# To reduce computation time, we only  use observations at periodic intervals for the kalman update
skip = 5
obs_indexes = range(len(model_ages))[::skip]
# Observation
y = L_interp(model_ages[obs_indexes])
# Assumed observation covariance
#R = 500.**2 * np.identity(len(y))
R = np.zeros((len(y), len(y)))
error_ts = np.array([-11.6, -10.9, -10.2, -9.7, -9.2, -8.7, -8.2, -7.75, -7.3])*1e3
#min_err = 1000.**2
#max_err = 2500.**2 #1000.**2
min_err = 5000.**2
max_err = 50000.**2
error_vs = np.array([min_err,  max_err,  min_err,   max_err,  min_err,   max_err,  min_err,   max_err,  min_err])
error_interp = interp1d(error_ts, error_vs, kind = 'linear')
errors = error_interp(model_ages[obs_indexes])
R[range(R.shape[0]), range(R.shape[0])] = errors


### Load stuff we need for the unscented transform 
#############################################################

# Sigma points
X = np.loadtxt(in_dir + 'X.txt')
# Sigma points run through 
Y = np.loadtxt(in_dir + 'Y.txt')
Y = Y[:, obs_indexes]
# Prior mean
m = np.loadtxt(in_dir + 'prior_m.txt')
# Prior covariance 
P = np.loadtxt(in_dir + 'prior_P.txt')
# Load mean weights
m_weights = np.loadtxt(in_dir + 'm_weights.txt')
# Load covariance weights
c_weights = np.loadtxt(in_dir + 'c_weights.txt')


### Do Kalman update
#############################################################

# Object for doing the Kalman update computations
ku = KalmanUpdate(m, P, X, Y, m_weights, c_weights, obs_indexes)

m_p, P_p, mu, K = ku.update(y, R)

plt.plot(m, 'ko')
plt.plot(m_p, 'ro')
plt.show()

np.savetxt(in_dir + 'mu_error.txt', mu)
np.savetxt(in_dir + 'opt_m_error.txt', m_p)
np.savetxt(in_dir + 'opt_P_error.txt', P_p)

np.savetxt('opt_m.txt', m_p)

plt.subplot(3,1,2)
plt.plot(model_ages, np.repeat(m_p, 30))
plt.plot(model_ages, np.repeat(m, 30))

plt.subplot(3,1,3)
v = P[range(len(P)), range(len(P))]
plt.plot(v)
plt.show()"""

inputs = {}
inputs['in_dir'] = 'filter/north_prior2/'
inputs['obs_Ls'] = np.array([424777.2650658561, 394942.08036138373, 332430.91816515941, 303738.49932773202, 296659.0156905292])
inputs['sparse_obs'] = True
inputs['out_dir'] = 'filter/north_prior2/opt/'
opt = Optimizer(inputs)

                               

