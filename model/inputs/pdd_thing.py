from pdd_calculator import *
import numpy as np
import matplotlib.pyplot as plt


# Temperature from -20 to 10 degrees C
temps = np.linspace(-20., 10.)
# The PDD calculator
pdd_calc = PDDCalculator(5.5)
# PDD's at the given temperature
pdds = pdd_calc.get_pdd(temps)
# Plot temp. v. PDD's
plt.plot(temps, pdds)
plt.xlabel('Temp (C)')
plt.ylabel("PDD's")
plt.xlim([temps.min(), temps.max()])
plt.show()
