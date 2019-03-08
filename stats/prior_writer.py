import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from stats.sigma_points import SigmaPoints

class PriorWriter(object):

    def __init__(self, inputs):

        # Output directory
        out_dir = inputs['out_dir']
        # Times 
        sigma_ts = inputs['sigma_ts'] 
        # Prior mean
        x = inputs['x']
        # Length of state vector
        N = len(x)
        # Covariance 
        Pxx = inputs['Pxx']

        
        ### Plot samples from prior
        ##########################################################################

        """
        samples = np.random.multivariate_normal(x, Pxx, 100)

        for i in range(100):
            plt.plot(samples[i])
        plt.show()
        """
        
        ### Compute a minimal set of sigma points
        ##############################################

        print(x.shape)
        print(Pxx.shape)
        points = SigmaPoints(x, Pxx)
        #quit()
        #X, wm, wc = points.get_minimal_set(inputs['w0'])
        X, wm, wc = points.get_fifth_order_set(inputs['w0'])
        # points.__get_random_set__(N + 1)


        branches = [[ np.array([v]) for v in np.unique(X[:,0]) ]]


        branches = []
    
        for i in range(1,45):
            x = len(np.unique(X[:, 0:i], axis = 0))
            print(x)
            branches.append(x)

        branches = np.array(branches)
        print()
        print(branches.sum() / (X.shape[0]*X.shape[1]))
        quit()
    
        #print(X)
        #quit()
        x_n = np.zeros(N)
        P_n = np.zeros((N,N))
        # Plot the sigma points and check the mean and covariance
        for i in range(len(X)):
            wm_i = wm[i]
            wc_i = wc[i]
            x_i = X[i]
            x_n += wm_i*x_i
            P_n += wc_i*np.outer(x_i - x, x_i - x)
            plt.plot(x_i)

        plt.show()


        ### Write the prior info to a file
        ####################################################
        np.savetxt(out_dir + 'sigma_ts.txt', sigma_ts)
        np.savetxt(out_dir + 'prior_m.txt', x)
        np.savetxt(out_dir + 'prior_P.txt', Pxx)
        np.savetxt(out_dir + 'm_weights.txt', wm)
        np.savetxt(out_dir + 'c_weights.txt', wc)
        np.savetxt(out_dir + 'X.txt', X)

        
