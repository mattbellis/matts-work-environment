import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from cogent_utilities import *


pi = np.pi

################################################################################
# Propagate point
################################################################################
def propagate(x,y,z,px,py,pz,r_extent,z_extent,stepsize=0.01):

    half_z = z_extent/2.0

    r,theta,z = cartesian_to_cylindrical(x,y,z)

    #print "-------"
    while np.abs(z)<half_z and r<r_extent:
        x += px*stepsize
        y += py*stepsize
        z += pz*stepsize
        #print x,y,z

        r,theta,z = cartesian_to_cylindrical(x,y,z)
        #print r,theta,z 
    
    return x,y,z

################################################################################
# Pick a random direction
################################################################################
def random_dir(npts):
    
    theta = np.arccos((np.random.random(npts)*2)-1.0)
    phi = 2*np.pi*np.random.random(npts)

    return theta,phi

################################################################################
# Convert cartesian to cylindrical
################################################################################
def cartesian_to_cylindrical(x,y,z):
    
    r = np.sqrt(x*x + y*y)
    theta = np.arcsin(y/r)
    z = z

    return r,theta,z

################################################################################
# Convert cylindrical coordinates to x,y,z
################################################################################
def cylindrical_to_cartesian(r,theta,z):
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    return x,y,z

################################################################################
# Generate a random point somewhere in the volume.
################################################################################
#def point_in_detector(npts, r_lo, r_hi, z_lo, z_hi):
def point_in_detector(npts, r_extent, z_extent, surface_depth=0.0):

    z = np.array([])
    r = np.array([])
    theta = np.array([])

    half_z = z_extent/2.0


    i = 0
    z_lo = 0.0
    r_lo = 0.0
    if surface_depth!=0.0:
        z_lo = half_z-surface_depth

    while i<npts:
        zpt = np.random.random()*half_z
        if zpt>z_lo:
            r_lo = 0.0

            rpt = (np.sqrt(np.random.random())*r_extent) - r_lo
            thetapt = (2.0*np.pi*np.random.random())

            if np.random.random()>0.5:
                zpt = -zpt

            z = np.append(z,zpt)
            r = np.append(r,rpt)
            theta = np.append(theta,thetapt)

            i+=1

        else:
            r_lo = r_extent-surface_depth

            rpt = (np.sqrt(np.random.random())*r_extent)
            
            if rpt>r_extent-surface_depth:
                thetapt = (2.0*np.pi*np.random.random())

                if np.random.random()>0.5:
                    zpt = -zpt

                z = np.append(z,zpt)
                r = np.append(r,rpt)
                theta = np.append(theta,thetapt)

                i+=1

    print len(theta)
    print len(r)
    print len(z)

    return r,theta,z



################################################################################
################################################################################
# Cylindrical coordinates for CoGeNT detector
# Full extent in theta (2pi)
cogent_r = 2.21 # radius in cm
cogent_z = 5.07 # length (z) in cm
surface_depth = 0.1 # 0.1 mm (0.01 cm)

#cogent_r = 2.01 # radius in cm
#cogent_z = 4.67 # length (z) in cm
#surface_depth = 0.0 # 0.1 mm (0.01 cm)

volume_cogent = (np.pi*(cogent_r**2))*cogent_z

volume_good = (np.pi*((cogent_r-surface_depth)**2))*(cogent_z-(2*surface_depth))

volume_surface = volume_cogent-volume_good

fractional_volume = volume_surface/volume_cogent

print "Volume CoGeNT: %f" % (volume_cogent)
print "Volume good: %f" % (volume_good)
print "Volume surface: %f" % (volume_surface)
print "fractional volume surface: %f" % (fractional_volume)

npts = 10000
#r,theta,z = point_in_detector(npts,cogent_r,cogent_z)
r,theta,z = point_in_detector(npts,cogent_r,cogent_z,surface_depth)
x,y,z = cylindrical_to_cartesian(r,theta,z)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, 'b.',markersize=1)

################################################################################
# Read in the CoGeNT data
################################################################################
infile = open('../data/before_fire_HG.dat')
content = np.array(infile.read().split()).astype('float')
print content
ndata = len(content)/2
print ndata
index = np.arange(0,ndata*2,2)
print index
times = content[index]
index = np.arange(1,ndata*2+1,2)
print index
amplitudes = content[index]
print times
energies = amp_to_energy(amplitudes,0)
print energies

plt.figure()
plt.hist(energies,bins=500)


################################################################################
# Propagate the points.
################################################################################
nsurface = ndata*fractional_volume
print "nsurface: ",nsurface 
#nsurface = 2000

# Generate the starting points
r,theta,z = point_in_detector(nsurface,cogent_r,cogent_z,surface_depth)
#r,theta,z = point_in_detector(nsurface,cogent_r,cogent_z,0.0)
x,y,z = cylindrical_to_cartesian(r,theta,z)

ptheta,pphi = random_dir(nsurface) 
px = np.sin(ptheta)*np.cos(pphi)
py = np.sin(ptheta)*np.sin(pphi)
pz = np.cos(ptheta)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(px,py,pz,'b.',markersize=1)

pathlen = np.array([])
for x0,y0,z0,ppx,ppy,ppz in zip(x,y,z,px,py,pz):
    x1,y1,z1 = propagate(x0,y0,z0,ppx,ppy,ppz,r_extent=cogent_r,z_extent=cogent_z,stepsize=0.001)
    #print cartesian_to_cylindrical(x1,y1,z1)
    pl = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
    pathlen = np.append(pathlen,pl)

plt.figure()
plt.hist(pathlen,bins=1500)

energy_sample = energies[energies>4.5]
indices = np.random.random_integers(0,len(energy_sample)-1,nsurface)
energy_subsample = energy_sample[indices]

plt.figure()
print pathlen
plt.plot(energy_subsample,pathlen,'.',markersize=1)
#plt.plot(energy_subsample,energy_subsample*pathlen,'.',markersize=1)

#exit()

plt.show()
