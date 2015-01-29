import numpy as np

import matplotlib
#matplotlib.use('Agg')

import lichen.lichen as lch

import matplotlib.pylab as plt

import sys

################################################################################
# Define the grid spacing and boundary locations
################################################################################
nx = 1000
ny = 1000

gbx = np.array([]) # grain-boundary x
gby = np.array([]) # grain-boundary y

grid_size = 40

for i in range(0,nx+1):
    for j in range(0,ny+1):
        if i%grid_size==0:
            gbx = np.append(gbx,i)
            gby = np.append(gby,j)

        if j%grid_size==0:
            gbx = np.append(gbx,i)
            gby = np.append(gby,j)


imgcount = 0
fig_img = plt.figure(figsize=(13,5),dpi=100)
ax0 = fig_img.add_subplot(1,2,1)
ax1 = fig_img.add_subplot(1,2,2)

for infile in sys.argv[1:]:

    print infile
    file = open(infile,'r')
    vals = np.array(file.read().split()).astype(float)
    nentries = len(vals)
    x = vals[0:nentries/2]
    y = vals[nentries/2:]

    ax0.plot(gbx,gby,'yo',markersize=0.5,alpha=0.1)
    ax0.plot(x,y,'o',markersize=1.5)
    lch.hist_err(x,range=(0,300),axes=ax1)
    ax1.set_ylim(0,50)

    name = "Plots_run2/img_%dx%dgrid_%dspacing_%datoms%06d.png" % (nx,ny,grid_size,1000,imgcount)
    
    fig_img.savefig(name)
    imgcount += 1
    ax0.clear()
    ax1.clear()

    #plt.show()

exit()


################################################################################
# Step forward in time
################################################################################
for t in xrange(25000):
    for k in xrange(nparticles):
        probs = np.array([0.02, 0.02, 0.02, 0.02])
        x = px[k]
        y = py[k]

        #'''
        xp1 = gbx==x+1
        xm1 = gbx==x-1
        xeq = gbx==x

        yp1 = gby==y+1
        ym1 = gby==y-1
        yeq = gby==y

        #print x,y,xp1,xm1,xeq,yp1,ym1,yeq

        if len(gbx[xp1*yeq])==1 and x+1<=nx:
            probs[0] = 0.2
        if len(gbx[xeq*ym1])==1 and y-1>=0:
            probs[1] = 0.2
        if len(gbx[xm1*yeq])==1 and x-1>=0:
            probs[2] = 0.2
        if len(gbx[xeq*yp1])==1 and y+1>=0:
            probs[3] = 0.2
        #'''


        '''
        for ig in xrange(len(gbx)):
            if x+1==gbx[ig] and y==gby[ig] and x+1<=nx:
                probs[0] = 0.2
            if x==gbx[ig] and y-1==gby[ig] and y-1>=0:
                probs[1] = 0.2
            if x-1==gbx[ig] and y==gby[ig] and x-1>=0:
                probs[2] = 0.2
            if x==gbx[ig] and y+1==gby[ig] and y+1>=0:
                probs[3] = 0.2
        '''

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
       # print probs

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


    # Save an image every 10 time steps
    if t%10==0:
        print "t:",t
        ax0.plot(gbx,gby,'bo',markersize=1)
        ax0.plot(px,py,'go',markersize=5)
        ax1.hist(px,bins=50,range=(0,nx))
        ax1.set_ylim(0,nparticles/4.0)
        #name = "Plots/img_slicing_200atoms%04d.png" % (imgcount)
        name = "Plots/img_slicing_%dx%dgrid_%dspacing_%datoms%04d.png" % (nx,ny,grid_size,nparticles,imgcount)
        #name = "Plots/test%04d.png" % (imgcount)
        fig_img.savefig(name)
        imgcount += 1
        ax0.clear()
        ax1.clear()

plt.plot(gbx,gby,'bo',markersize=1)
plt.plot(px,py,'go',markersize=5)

#plt.show()
