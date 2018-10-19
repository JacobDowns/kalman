from pylab import *

P = loadtxt('opt_P.txt')
P = abs(P)
print abs(P).min(), abs(P).max()

P[P < 1e-8] = 0.


imshow(P, cmap = 'jet')
colorbar()


show() 
