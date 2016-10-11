import gr_tools as gr
import numpy as np
import matplotlib.pylab as plt

#positions = np.array([1.,2.,3.,4.,5.,6.])
#velocities = np.array([1.,1.,1.,1.,1.,1.])
positions = np.arange(1.0,20.0,1.0)
velocities = np.ones(len(positions))
accelerations = None

dt = 0.01

t = 0 # Time
tmax = 100

all_positions = []
all_velocities = []
all_times = []
while t < tmax:

    #forces = gr.calc_forces(positions)
    forces = gr.calc_retarded_forces(positions,velocities)
    #print forces
    accelerations = gr.acc_from_force(forces)

    velocities += accelerations*dt

    positions += dt*velocities

    all_positions.append(positions.copy())
    all_velocities.append(velocities.copy())
    all_times.append(t)

    t += dt

all_positions = np.array(all_positions)
all_positions = all_positions.transpose()

all_velocities = np.array(all_velocities)
all_velocities = all_velocities.transpose()

print all_positions
print accelerations

plt.figure()
plt.subplot(1,2,1)
for p in all_positions:
    plt.plot(all_times, p)

plt.subplot(1,2,2)
for p in all_velocities:
    plt.plot(all_times, p)
plt.show()



