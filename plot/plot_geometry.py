import sys
sys.path.append('../')
from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import os.path
import matplotlib

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

matplotlib.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(12,14))

### North
################################################################################
dt = 1./3.
model_inputs = PaleoInputs('../paleo_inputs/north_steady.h5', {})
model = ForwardIceModel(model_inputs, "out", "paleo")

xs = model_inputs.mesh.coordinates() * model_inputs.L_init
L_init = model_inputs.L_init

surface = project(model.S)
bed = project(model.B)

ax = fig.add_subplot(311)
plt.title('(a)')

obs_Ls = np.array([443746.66897917818, 397822.86008538032, 329757.49741948338, 292301.29712071194, 285478.05793305294])
colors = ['r', 'g', 'b', 'maroon', 'c']
plt.plot([obs_Ls[0], obs_Ls[0]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '11.6')
plt.plot([obs_Ls[1], obs_Ls[1]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '10.2')
plt.plot([obs_Ls[2], obs_Ls[2]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '9.2')
plt.plot([obs_Ls[3], obs_Ls[3]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '8.2')
plt.plot([obs_Ls[4], obs_Ls[4]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '7.3')

plt.plot(xs, surface.compute_vertex_values(), 'b', linewidth = 3.5)
plt.plot(xs, bed.compute_vertex_values(), 'k', linewidth = 3.5)

plt.ylim([-300., 2950.])
plt.xlim([0., 443746. + 2.5e3])

ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])

### Center
################################################################################
model_inputs = PaleoInputs('../paleo_inputs/center_steady.h5', {})
model = ForwardIceModel(model_inputs, "out", "paleo")

xs = model_inputs.mesh.coordinates() * model_inputs.L_init
L_init = model_inputs.L_init

surface = project(model.S)
bed = project(model.B)

ax = fig.add_subplot(312)
plt.title('(b)')

obs_Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725])
plt.plot([obs_Ls[0], obs_Ls[0]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '11.6')
plt.plot([obs_Ls[1], obs_Ls[1]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '10.2')
plt.plot([obs_Ls[2], obs_Ls[2]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '9.2')
plt.plot([obs_Ls[3], obs_Ls[3]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '8.2')
plt.plot([obs_Ls[4], obs_Ls[4]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '7.3')

plt.plot(xs, surface.compute_vertex_values(), 'b', linewidth = 3.5)
plt.plot(xs, bed.compute_vertex_values(), 'k', linewidth = 3.5)

plt.ylim([-300., 2950.])
plt.xlim([0., 443746. + 2.5e3])
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.ylabel('Elevation (m)')

### South
################################################################################
model_inputs = PaleoInputs('../paleo_inputs/south_steady.h5', {})
model = ForwardIceModel(model_inputs, "out", "paleo")

xs = model_inputs.mesh.coordinates() * model_inputs.L_init
L_init = model_inputs.L_init

surface = project(model.S)
bed = project(model.B)

ax = fig.add_subplot(313)
plt.title('(c)')

obs_Ls = np.array([424777.2650658561, 394942.08036138373, 332430.91816515941, 303738.49932773202, 296659.0156905292])
plt.plot([obs_Ls[0], obs_Ls[0]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '11.6')
plt.plot([obs_Ls[1], obs_Ls[1]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '10.2')
plt.plot([obs_Ls[2], obs_Ls[2]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '9.2')
plt.plot([obs_Ls[3], obs_Ls[3]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '8.2')
plt.plot([obs_Ls[4], obs_Ls[4]], [-1000., 5000.], 'k', lw = 3, linestyle = ':', label = '7.3')

plt.plot(xs, surface.compute_vertex_values(), 'b', linewidth = 3.5)
plt.plot(xs, bed.compute_vertex_values(), 'k', linewidth = 3.5)



plt.ylim([-300., 2950.])
plt.xlim([0., 443746. + 2.5e3])
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.xlabel(r'Length (km)')
plt.tight_layout()
plt.savefig('geo.png', dpi=700)
