import numpy as np
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints

class PriorWriter(object):

    def __init__(self, input_dict):

        # Load output directory
        out_dir = input_dict['out_dir']
        if 'N' in input_dict:
            N = input_dict['N']
        else :
            N = 87
        # Delta temp. grid years
        dt_years = -11.6e3 + np.linspace(0., 4300, N)
        np.savetxt(out_dir + 'sigma_ts.txt', dt_years)
        

        ### Define prior mean
        ###########################################################################
        
        # Prior defined by an interpolation object
        if 'delta_temp_file' in input_dict:
            x = np.loadtxt(input_dict['delta_temp_file'])
        else :
            # Load Jensen dye3 temp.
            data = np.loadtxt('paleo_data/jensen_dye3.txt')
            # Years before present (2000)
            years = data[:,0] - 2000.0
            # Temps. in K
            temps = data[:,1]
            # Delta temps. 
            delta_temp_interp = interp1d(years, temps - temps[-1], kind = 'linear')
            x = delta_temp_interp(dt_years)


        ### Define prior covariance 
        ##########################################################################
        # Delta controls smoothness
        delta = 1500.
        # Covariance matrix
        P = np.zeros((N, N))
        P[range(N), range(N)] = 2.
        P[range(1,N), range(N-1)] = -1.
        P[range(N-1), range(1,N)] = -1.
        P[N-1, N-1] = 1.
        delta = 1500.
        P = delta*P
        P = np.linalg.inv(P)
        # Save prior mean and covariance
        np.savetxt(out_dir + 'prior_m.txt', x)
        np.savetxt(out_dir + 'prior_P.txt', P)

        
        ### Compute sigma points
        ##########################################################################
        # Generate Juleir sigma points
        points = JulierSigmaPoints(N, kappa=20*len(years))
        sigma_points = points.sigma_points(x, P)
        # Save the mean and covariance weights, as well as the sigma points
        np.savetxt(out_dir + 'm_weights.txt', points.weights()[0])
        np.savetxt(out_dir + 'c_weights.txt', points.weights()[1])
        np.savetxt(out_dir + 'X.txt', sigma_points)
