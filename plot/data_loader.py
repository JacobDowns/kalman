import numpy as np

"""
Load optimized data.
"""

class DataLoader(object):

    def __init__(self, base_dir, opt_dir):

        # Sigma point ages
        self.sigma_ages = np.loadtxt(base_dir + 'sigma_ts.txt') / 1e3
        
        in_dir = base_dir + opt_dir
        
        # Measurement ages
        self.ages = np.loadtxt(in_dir + 'opt_age.txt') / 1e3
        # Measurement mean
        self.mu = np.loadtxt(in_dir + 'mu.txt')
        # Glacier length
        self.L = None
        try :
            self.L = np.loadtxt(in_dir + 'opt_L.txt') / 1e3
        except :
            pass

        # Mean
        self.deltap = np.loadtxt(in_dir + 'opt_m.txt')
        
        # Precipitation
        self.precip = None
        try:
            self.precip = (np.loadtxt(in_dir + 'opt_p.txt') / (self.L*1e3))
        except:
            pass
        
        # Covariance 
        self.P = np.loadtxt(in_dir + 'opt_P.txt')
               
        self.v = np.diag(self.P)
        # Confidence bands
        self.deltap_l = self.deltap - 2.0*np.sqrt(self.v)
        self.deltap_u = self.deltap + 2.0*np.sqrt(self.v) 
