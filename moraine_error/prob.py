import matplotlib.pyplot as plt
import numpy as np


Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686]) / 1000.

# Mean 
ts = -np.array([11.6, 10.3, 9.0, 8.1, 7.3, 0.])

tsf = np.linspace(ts.min(), ts.max(), 100)
Lsf = np.linspace(Ls.min(), Ls.max(), 100)


tt, ll = np.meshgrid(ts, y, sparse=False, indexing='xy')

# Define the line segments 
