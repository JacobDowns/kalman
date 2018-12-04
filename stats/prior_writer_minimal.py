import numpy as np
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints
import matplotlib.pyplot as plt

class PriorWriter(object):

    def __init__(self, input_dict):

        # Output directory
        out_dir = input_dict['out_dir']
        # State vector length 
        if 'N' in input_dict:
            N = input_dict['N']
        else :
            N = 45

        # Delta temp. grid years
        start_age = -11.62e3 + 66.
        dt_years = start_age + np.linspace(0., abs(start_age), N)
        np.savetxt(out_dir + 'sigma_ts.txt', dt_years)

        ### Define prior mean
        ###########################################################################
        
        if 'x' in input_dict:
            # Load a custom prior
            x = np.loadtxt(input_dict['x'])
        else :
            chi = np.linspace(0., 1., len(dt_years))
            x = .45*np.ones(len(dt_years)) - 0.45*chi**2


        ### Define prior covariance 
        ##########################################################################

        # Delta controls smoothness
        delta = 5000.
        if 'delta' in input_dict:
            delta = input_dict['delta']
            
        # Precision matrix
        Q = np.zeros((N, N))
        Q[range(N), range(N)] = 2.
        Q[range(1,N), range(N-1)] = -1.
        Q[range(N-1), range(1,N)] = -1.
        Q[N-1, N-1] = 2.
        Q = delta*Q
        # Covariance 
        P = np.linalg.inv(Q)

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

        
        # Compute a minimal set of sigma points
        ##############################################

        # Compute X matrix where each row is a sigma point
        w_0 = 0.5
        alpha = np.sqrt((1. - w_0) / N)
        C = np.linalg.cholesky(np.diag(np.ones(N), 0) - (alpha**2)*np.ones((N, N)))
        W = np.diag(np.diag(np.linalg.multi_dot([w_0*(alpha**2)*np.linalg.inv(C), np.ones((N,N)), np.linalg.inv(C.T)])), 0)
        W_sqrt = np.linalg.cholesky(W)
        P_sqrt = np.linalg.cholesky(P)
        X = np.zeros((N, N+1))
        X[:,0] = np.dot(-P_sqrt, (alpha / np.sqrt(w_0))*np.ones(N))
        X[:,1:] = np.linalg.multi_dot([P_sqrt, C, np.linalg.inv(W_sqrt)])
        X = X.T
        X += x

        # Array of weights
        weights = np.zeros(N+1)
        weights[0] = w_0
        weights[1:] = np.diag(W, 0)


        x_n = np.zeros(N)
        P_n = np.zeros((N,N))
        # Plot the sigma points and check the mean and covariance
        for i in range(N+1):
            w_i = weights[i]
            x_i = X[i]
            x_n += w_i*x_i
            P_n += w_i*np.outer(x_i - x, x_i - x)
            plt.plot(x_i)


        plt.show()
        print(abs(x_n - x).max())
        print(abs(P_n - P).max())
        
        # Save the mean and covariance weights, as well as the sigma points
        np.savetxt(out_dir + 'm_weights.txt', weights)
        np.savetxt(out_dir + 'c_weights.txt', weights)
        np.savetxt(out_dir + 'X.txt', X)

        
