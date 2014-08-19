import numpy as np
import matplotlib.pylab as plt

# This is the starting point in time.
t = 0

dt = 0.0000001

electron_momentum = 1e6 # eV
electron_mass = 511 # eV
c = 3e8 # m/s

electron_energy = np.sqrt(electron_momentum**2 + electron_mass**2)

x0 = 0.0

v0 = c*(electron_momentum/electron_energy)

v = v0
x = x0

################################################################################
# Loop over in time, taking one dt step each time.
################################################################################
while t < 0.000001:

    # Figure out how far it has moved in time dt
    # This is dependent on the velocity, v.
    dx = dt*v

    # Move forward that dx step
    x = x + dx

    print x

    t += dt



