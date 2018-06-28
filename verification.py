import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints

"""
This script just verifies some calculations for the unscented transform.
"""

### Mean and covariance of prior
##########################################################################
N = 100 
x = np.ones(N)
P = np.zeros((N, N))
P[range(N), range(N)] = 2.
P[range(1,N), range(N-1)] = -1.
P[range(N-1), range(1,N)] = -1.
P[N-1, N-1] = 1.
P = 300.*P
P = np.linalg.inv(P)


### Compute sigma points
##########################################################################
points = JulierSigmaPoints(N, kappa=3 - len(x))
# Rows are sigma points
X = points.sigma_points(x, P)


### Run the sigma points through the model
##########################################################################

# Model matrix
A = np.zeros((N, N)) + 1.
A = np.tril(A)

# Sigma points passed through observation model 
Y = np.zeros((X.shape[0], N))

for i in range(X.shape[0]):
    Y[i] = np.dot(A, X[i])
    plt.plot(Y[i])

    
### Check mean
##########################################################################
    
# Unscented mean
mu = np.dot(points.weights()[0], Y)
# Analytic mean
mu_a = np.dot(A, x)

print "mu dif"
print np.abs(mu_a - mu).max()
print 


### Check measurement variance
##########################################################################

# Analytic covariance of y
S_a = np.dot(A, np.dot(P, A.T))
# Unscented covariance of y
S = np.zeros((Y.shape[1], Y.shape[1]))
for i in range(len(points.weights()[1])):
    S += points.weights()[1][i]*np.outer(Y[i] - mu, Y[i] - mu)

print "S dif"
print np.abs(S_a - S).max()
print


### Check cross covariance 
##########################################################################


# Analytic cross covariance
C_a = np.dot(P, A.T)
# Unscented cross covariance
C = np.zeros((X.shape[1], Y.shape[1]))
for i in range(len(points.weights()[1])):
    C += points.weights()[1][i]*np.outer(X[i] - x, Y[i] - mu)

print "C dif"
print np.abs(C_a - C).max()
print
