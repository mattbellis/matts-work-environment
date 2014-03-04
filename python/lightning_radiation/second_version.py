import numpy as np
import matplotlib.pylab as plt

import scipy.constants as constants

dt = 0.00000001

electron_mass = 511 # eV
c = constants.speed_of_light # Speed of light
h = constants.h # Speed of light

prob_of_brem = 0.1 # 10%

photon_energies = []

nelectrons = 10000

################################################################################
# Do this for many electrons.
################################################################################
for i in range(0,nelectrons):
    # This is the starting point in time.
    t = 0

    electron_momentum = 1e6 # eV

    electron_energy = np.sqrt(electron_momentum**2 + electron_mass**2)

    x0 = 0.0

    v0 = c*(electron_momentum/electron_energy)

    v = v0
    x = x0
    ################################################################################
    # Loop over in time, taking one dt step each time.
    ################################################################################
    while t < 0.000001 and electron_energy>electron_mass:

        # Figure out how far it has moved in time dt
        # This is dependent on the velocity, v.
        dx = dt*v

        # Move forward that dx step
        x = x + dx

        #print x

        # Calculate whether or not, I've radiated a photon.
        test_number = np.random.random()
        if test_number < prob_of_brem:
            # We've radiated a brem photon!
            # Now figure out, what the energy of the photon is!
            photon_energy = electron_energy*np.random.random()

            photon_energies.append(photon_energy)
            #photon_energies.append((h*c)/(1.6e-19*photon_energy))

            #print "photon_energy:",photon_energy

            electron_energy -= photon_energy
            electron_momentum = np.sqrt(electron_energy**2 - electron_mass**2)
            v = c*(electron_momentum/electron_energy)

        t += dt



plt.figure()
plt.hist(photon_energies,bins=50)
plt.show()
