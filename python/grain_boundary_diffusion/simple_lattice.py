import numpy as np

import matplotlib
matplotlib.use('Agg')

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
nparticles = 200

px = np.zeros(nparticles)
# Start all of the atoms at one point
#py = (ny/2)*np.ones(nparticles)
# Randomly arrange them on the y-axis
starty = np.arange(0,ny+1)
np.random.shuffle(starty)
py = None
if nparticles<ny:
    py = starty[0:nparticles]
else:
    py = np.ones(nparticles)
    for i in range(0,nparticles):
        py[i] = np.random.randint(0,nx+1)

plt.plot(gbx,gby,'bo',markersize=1)
plt.plot(px,py,'ro',markersize=10)

imgcount = 0
fig_img = plt.figure(figsize=(13,5),dpi=100)
ax0 = fig_img.add_subplot(1,2,1)
ax1 = fig_img.add_subplot(1,2,2)


for t in xrange(25000):
    for k in xrange(nparticles):
        probs = np.array([0.02, 0.02, 0.02, 0.02])
        x = px[k]
        y = py[k]

        for ig in xrange(len(gbx)):
            if x+1==gbx[ig] and y==gby[ig] and x+1<=nx:
                probs[0] = 0.2
            if x==gbx[ig] and y-1==gby[ig] and y-1>=0:
                probs[1] = 0.2
            if x-1==gbx[ig] and y==gby[ig] and x-1>=0:
                probs[2] = 0.2
            if x==gbx[ig] and y+1==gby[ig] and y+1>=0:
                probs[3] = 0.2

            # Boundary conditions
            if x==nx:
                probs[0]=0
            if x==0:
                probs[2]=0
            if y==ny:
                probs[3]=0
            if y==0:
                probs[1]=0

        #print probs
        #norm = sum(probs)

        #probs /= norm
        for i in range(1,len(probs)):
            probs[i] += probs[i-1]
        #print probs

        # Take a step
        step_dir = np.random.random()
        istep_dir = -1
        for i in range(0,len(probs)):
            if step_dir<probs[i]:
                istep_dir = i
                break

        if istep_dir==0:
            px[k]+=1
        elif istep_dir==1:
            py[k]-=1
        elif istep_dir==2:
            px[k]-=1
        elif istep_dir==3:
            py[k]+=1


    if t%10==0:
        print "t:",t
        ax0.plot(gbx,gby,'bo',markersize=1)
        ax0.plot(px,py,'go',markersize=5)
        ax1.hist(px,bins=50,range=(0,nx))
        ax1.set_ylim(0,nparticles)
        name = "Plots/img_200atoms%03d.png" % (imgcount)
        fig_img.savefig(name)
        imgcount += 1
        ax0.clear()
        ax1.clear()

plt.plot(gbx,gby,'bo',markersize=1)
plt.plot(px,py,'go',markersize=5)

#plt.show()
