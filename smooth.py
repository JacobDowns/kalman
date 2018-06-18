import numpy as np
from sigma_points_scalar import *
import matplotlib.pyplot as plt

# Times
ts = np.linspace(0., 2*np.pi, 100)
# Un-smoothed means
mus = np.sin(ts)
# Un-smoothed variances
Ps = 0.1*np.ones(len(mus))

mwer_sigma = SigmaPointsScalar(alpha = 0.1, beta = 2., kappa = 2.)
Q = 0.01


def F(x):
    return x

# Smoothed mean at last time
m_s_k_1 = mus[-1]
# Smoothed variance at last time
P_s_k_1 = Ps[-1]

ms_smoothed = [m_s_k_1]
Ps_smoothed = [P_s_k_1]

for i in range(len(mus) - 2, -1, -1):
    print i
    
    # Unsmoothed mean 
    m_k = mus[i]
    # Unsmoothed variance 
    P_k = Ps[i]

    ### Compute predicted mean, predicted covariance and cross-covariance
    ########################################################################
    x_k = mwer_sigma.sigma_points(m_k, P_k)
    x_hat_k_1 = F(x_k) 
    # Predicted mean
    m_minus_k_1 = np.dot(mwer_sigma.mean_weights, x_hat_k_1)
    # Predicted variance
    P_minus_k_1 = np.dot(mwer_sigma.variance_weights, (x_hat_k_1 - m_minus_k_1)**2) + Q
    # Predicted cross-variance
    D_k_1 = np.dot(mwer_sigma.variance_weights, (x_k - m_k) * (x_hat_k_1 - m_minus_k_1))

    ### Compute smoother gain, smoothed mean, and covariance
    ########################################################################
    # Smoother gain
    G_k = D_k_1 * (1. / P_minus_k_1)
    # Smoothed mean
    m_s_k = m_k + G_k * (m_s_k_1 - m_minus_k_1)
    P_s_k = P_k + G_k * (P_s_k_1 - P_minus_k_1) * G_k


    ### Save these for the next step
    ########################################################################
    m_s_k_1 = m_s_k
    P_s_k_1 = P_s_k

    ms_smoothed.append(m_s_k_1)
    Ps_smoothed.append(P_s_k_1)


ms_smoothed = np.array(ms_smoothed)[::-1]
Ps_smoothed = np.array(Ps_smoothed)[::-1]


plt.plot(ts, mus, 'ko-')
plt.plot(ts, ms_smoothed, 'ro-')
plt.show()

    
    
    
