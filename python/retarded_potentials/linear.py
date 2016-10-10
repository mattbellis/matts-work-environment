import gr_tools as gr
import numpy as np
import matplotlib.pylab as plt

positions = np.array([1.,2.,3.,4.,5.,6.])
velocities = np.array([1.,1.,1.,1.,1.,1.])
accelerations = None

dt = 0.01

t = 0 # Time
tmax = 1

all_positions = []
all_times = []
while t < tmax:

    forces = gr.calc_forces(positions)
    #print forces
    accelerations = gr.acc_from_force(forces)

    velocities += accelerations*dt

    positions += dt*velocities
    all_positions.append(positions.copy())
    all_times.append(t)

    t += dt

all_positions = np.array(all_positions)
all_positions = all_positions.transpose()

print all_positions
print accelerations

for p in all_positions:
    plt.plot(all_times, p)
plt.show()



