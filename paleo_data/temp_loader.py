import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

"""
Load Buizert and Dahl-Jensen temperature anomalies. 
"""

class TempLoader(object):

    def __init__(self):

        # Load and interpolate Buizert data
        ##################################################
        self.b_years = np.loadtxt('buizert_ages.txt')
        self.b_dts = np.loadtxt('buizert_dts.txt')
        self.b_avg = self.b_dts.mean(axis = 1)
        self.b_interp = [interp1d(self.b_years, self.b_dts[:,i]) for i in range(12)]
        self.b_avg_interp = interp1d(self.b_years, self.b_avg)

        
        # Load Dahl-Jensen GRIP data
        ##################################################
        self.d_years = np.loadtxt('dj_ages.txt')
        self.d_dts = np.loadtxt('dj_dts.txt')
        self.d_interp = [interp1d(self.d_years, self.d_dts[:,i].T) for i in range(12)]
        self.d_avg = self.d_dts.mean(axis = 1)
        self.d_avg_interp = interp1d(self.d_years, self.d_avg)

#dt_seasonal = np.zeros((len(l.d_years), 12))
ts = np.linspace(-15e3, 0., 1000)
l = TempLoader()
dj_seasonal = np.zeros((len(ts), 12))


plt.plot(l.d_interp[0](ts))
plt.plot(l.b_interp[0](ts))
plt.show()

quit()


plt.subplot(2,1,1)
for i in range(12):
    #plt.plot(ts, l.d_interp[i](ts))
    #offset = l.b_interp[i](ts) - l.b_avg_interp(ts)
    plt.plot(ts, l.b_interp[i](ts))
    #plt.plot(ts, l.b_interp[i](ts))
    
plt.plot(ts, l.b_avg_interp(ts), 'k', lw = 3)


plt.subplot(2,1,2)
for i in range(12):
    #plt.plot(ts, l.d_interp[i](ts))
    offset = l.b_interp[i](ts) - l.b_avg_interp(ts)
    dj_seasonal[:,i] = l.d_avg_interp(ts) + offset
    plt.plot(ts, dj_seasonal[:,i])

plt.plot(ts, l.d_avg_interp(ts))
np.savetxt('dj_dts_seasonal.txt', dj_seasonal)
#plt.plot(l.d_avg, 'k', lw = 3)

plt.show()


"""
for i in range(12):
    offset = l.b_interp[i](ts) - l.b_avg_interp(ts)
    #plt.plot(offset)
    dj_seasonal[:,i] = l.d_avg_interp(ts) + offset
    #plt.plot(dj_seasonal[:,i])
    #plt.plot(l.b_interp[i](ts))
    #plt.plot(ts, l.b_interp[i](ts), 'k')
    #plt.plot(l.d_dts[:,i])
    plt.plot(ts, l.d_avg_interp(ts))
    plt.plot(ts, dj_seasonal[:,i])
    #dj_seasonal[:,i] = 
    
plt.show()
np.savetxt('dj_dts_seasonal.txt', dj_seasonal)
"""
