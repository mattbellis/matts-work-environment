import numpy as np
import matplotlib.pylab as plt
import pylhef

import sys

lhefile = pylhef.read(sys.argv[1])

pts = []

for event in lhefile.events:
    particles = event.particles

    for particle in particles:
        if np.abs(particle.id)>=11 and np.abs(particle.id)<=18:
            first,last = particle.first_mother,particle.last_mother
            if first==last and np.abs(particles[first].id)==6:
                p = particle.p
                Ee = p[0] # Energy
                #Et = particles[first].p[0]
                Mt = particles[first-1].mass
                #Mt = 173.
                print(particle.id,Ee,Mt)
                #pt = np.sqrt(p[1]*p[1] + p[2]*p[2])
                #pts.append(pt)
                pts.append(2*Ee/Mt)


plt.figure()
plt.hist(pts,bins=50)
plt.show()
