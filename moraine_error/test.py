import matplotlib.pyplot as plt
import numpy as np


Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686])

ts = -np.array([11.62, 10.35, 9.09, 8.13, 7.30, 0.])
vs = np.array([0.0,  0.19,  0.20, 0.29, 0.28, 0.])




plt.plot(ts - vs, Ls, 'k--')
plt.plot(ts, Ls, 'k')
plt.plot(ts + vs, Ls, 'k--') 
plt.show()
