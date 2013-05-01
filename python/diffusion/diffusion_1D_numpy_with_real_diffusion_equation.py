import matplotlib.pylab as plt
import numpy as np


################################################################################
# Spatial dimensions
################################################################################
lo = 0
hi = 2775e-6
npts = 500

xpos = np.linspace(lo,hi,npts)

cmax0 = 6.13
cmin0 = 0.62

frac_interface = hi/1520e-6

point_of_intial_interface = int(npts/frac_interface)
xvals = np.linspace(lo,hi,npts)

concentration = np.zeros(npts)
concentration[:point_of_intial_interface] = cmax0
concentration[point_of_intial_interface:] = cmin0

c01 = np.zeros(npts)
c10 = np.zeros(npts)

plt.figure()
plt.plot(xpos,concentration,'o')
plt.ylim(0,7.0)

################################################################################
# Move in time
################################################################################

dt   = 0.01
dx   = (hi-lo)/float(npts)
t0   = 0
tmax = (60*17) + 31

invdx2 = 1.0/(dx**2)

#D = 3.17e-10
D = 3.17364e-10
#D = 3.185e-10

print "dx: ",dx
print "invdx2: ",invdx2
print "dt: ",dt
print "D*dt*invdx2: ",D*(dt*invdx2)

t = t0

#tag = "default"
#tag = "D=3.17"
tag = "D=3.17364"
#tag = "D=3.185"

#fig_img = plt.figure()
#ax = fig_img.add_subplot(1,1,1)

imgcount = 0
it = 0

Dnum = dt*invdx2
Dnum *= D

print "Dnum: ",Dnum


while t<tmax:

    if t%10000==0:
        print t

    c01 = np.roll(concentration, 1)
    c10 = np.roll(concentration,-1)

    # All the points except the end points.
    concentration_temp = concentration + Dnum*(c01-concentration + c10-concentration)

    # The end points
    concentration_temp[0]  = concentration[0]  + (Dnum*(concentration[1] -concentration[0]))
    concentration_temp[-1] = concentration[-1] + (Dnum*(concentration[-2]-concentration[-1]))

    # Copy it over. 
    concentration = concentration_temp.copy()

    t += dt

    '''
    if it%1000==0:
        print "here"
        ax.plot(xpos,concentration,'o')
        ax.set_ylim(0,7.0)
        name = "Plots/img%03d.png" % (imgcount)
        fig_img.savefig(name)
        imgcount += 1
        ax.clear()
    it += 1
    '''

print "max: ",max(concentration)
print "min: ",min(concentration)

################################################################################
# Write out the output
################################################################################
name = "output_%s.dat" % (tag)
outfile = open(name,'w+')

output = ""
for x,y in zip(xpos,concentration):
    output += "%f %f\n" % (x,y)
outfile.write(output)
outfile.close()


################################################################################
# Plot the result
################################################################################
plt.figure()
plt.plot(xpos,concentration,'o')
plt.ylim(0,7.0)
plt.show()

