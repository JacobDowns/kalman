import sys
sys.path.append('/home/jake/kalman')
from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import os.path
import matplotlib
import seaborn as sns

"""
plot_moraine('11_6_limit.p', 'r', label = '11.6 Moraine')
plot_moraine('11_6_moraine.p', 'r')
plot_moraine('10_2_limit.p', 'g', label = '10.2 Moraine')
plot_moraine('10_2_moraine.p', 'g')
plot_moraine('9_2_limit.p', 'b', label = '9.2 Moraine')
plot_moraine('9_2_moraine.p', 'b')
plot_moraine('8_2_limit.p', 'maroon', label = '8.2 Moraine')
plot_moraine('8_2_moraine.p', 'maroon')
plot_moraine('7_3_limit.p', 'c', label = '7.3 Moraine')
plot_moraine('7_3_moraine.p', 'c')"""

matplotlib.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(10,8))
current_palette = sns.color_palette()
#sns.palplot(current_palette)
#plt.show()
#quit()

### Center
################################################################################
model_inputs = PaleoInputs('paleo_inputs/center_steady_seasonal.h5', {})
model = ForwardIceModel(model_inputs, "out", "paleo")

xs = model_inputs.mesh.coordinates() * model_inputs.L_init
L_init = model_inputs.L_init

surface = project(model.S)
bed = project(model.B)

ax = fig.add_subplot(211)
plt.title('(a)')

obs_Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686])
plt.plot([obs_Ls[0], obs_Ls[0]], [-1000., 5000.], 'k', lw = 3, dashes = (2,1), label = '11.6')
plt.plot([obs_Ls[1], obs_Ls[1]], [-1000., 5000.], 'k', lw = 3, dashes = (2,1), label = '10.2')
plt.plot([obs_Ls[2], obs_Ls[2]], [-1000., 5000.], 'k', lw = 3, dashes = (2,1), label = '9.2')
plt.plot([obs_Ls[3], obs_Ls[3]], [-1000., 5000.], 'k', lw = 3, dashes = (2,1), label = '8.2')
plt.plot([obs_Ls[4], obs_Ls[4]], [-1000., 5000.], 'k', lw = 3, dashes = (2,1), label = '7.3')
plt.plot([obs_Ls[5], obs_Ls[5]], [-1000., 5000.], color = current_palette[3], lw = 3, dashes = (2,1), label = '0')

plt.plot(xs, surface.compute_vertex_values(), color = 'k', linewidth = 4.7)
plt.plot(xs, surface.compute_vertex_values(), color = current_palette[0], linewidth = 3.5)
plt.plot(xs, bed.compute_vertex_values(), 'k', linewidth = 3.5)

plt.ylim([-300., 3700.])
plt.xlim([0., 430e3 + 2.5e3])
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.ylabel('Elevation (m)')

### South
################################################################################
model_inputs = PaleoInputs('paleo_inputs/south_steady_seasonal.h5', {})
model = ForwardIceModel(model_inputs, "out", "paleo")

xs = model_inputs.mesh.coordinates() * model_inputs.L_init
L_init = model_inputs.L_init

surface = project(model.S)
bed = project(model.B)

ax = fig.add_subplot(212)
plt.title('(b)')

obs_Ls = [424777.2650658561, 394942.08036138373, 332430.9181651594, 303738.499327732, 296659.0156905292, 284686.5963970118]
plt.plot([obs_Ls[0], obs_Ls[0]], [-1000., 5000.], 'k', lw = 3, dashes = (2,1), label = '11.6')
plt.plot([obs_Ls[1], obs_Ls[1]], [-1000., 5000.], 'k', lw = 3, dashes = (2,1), label = '10.2')
plt.plot([obs_Ls[2], obs_Ls[2]], [-1000., 5000.], 'k', lw = 3, dashes = (2,1), label = '9.2')
plt.plot([obs_Ls[3], obs_Ls[3]], [-1000., 5000.], 'k', lw = 3, dashes = (2,1), label = '8.2')
plt.plot([obs_Ls[4], obs_Ls[4]], [-1000., 5000.], 'k', lw = 3, dashes = (2,1), label = '7.3')
plt.plot([obs_Ls[5], obs_Ls[5]], [-1000., 5000.], color = current_palette[3], lw = 3, dashes = (2,1), label = '0')

plt.plot(xs, surface.compute_vertex_values(), color = 'k', linewidth = 4.7)
plt.plot(xs, surface.compute_vertex_values(), color = current_palette[0], linewidth = 3.5)
plt.plot(xs, bed.compute_vertex_values(), 'k', linewidth = 3.5)

plt.ylim([-300., 3700.])
plt.xlim([0., 430e3 + 2.5e3])
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.ylabel('Elevation (m)')
plt.xlabel(r'Along Flow Length (km)')
plt.tight_layout()
plt.savefig('geo_final.png', dpi=500)
