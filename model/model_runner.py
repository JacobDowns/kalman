from model.inputs.paleo_inputs1 import *
from model.forward_model.forward_ice_model import *

#import matplotlib.pyplot as plt
#import numpy as np
#import sys
#import os.path

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

