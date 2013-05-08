import numpy as np
import matplotlib.pylab as plt

nx = 100
ny = 100

gbx = np.array([])
gby = np.array([])

grid_size = 20

for i in range(0,nx+1):
    for j in range(0,ny+1):
        if i%grid_size==0:
            gbx = np.append(gbx,i)
            gby = np.append(gby,j)

        if j%grid_size==0:
            gbx = np.append(gbx,i)
            gby = np.append(gby,j)


################################################################################
# Start diffusing the particles.
################################################################################
nparticles = 10

px = np.zeros(10)
py = (ny/2)*np.ones(10)

plt.plot(gbx,gby,'bo',markersize=1)
plt.plot(px,py,'ro',markersize=10)

for t in xrange(10):
    probs = [0.0, 0.0, 0.0, 0.0]
    for k in xrange(nparticles):
        x = gbx[k]
        y = gby[k]

        for ig in xrange(len(gbx)):
            if x+1==gbx[ig] and y==gbx[ig]:
                probs[0] = 0.2
                break



plt.show()
