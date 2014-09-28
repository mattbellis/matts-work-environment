import sys
import matplotlib.pylab as plt
import numpy as np

from plot_diffusion_data import read_in_a_microprobe_data_file,read_in_an_isotope_data_file


################################################################################
# Read in a data file
################################################################################
# Read in the microprobe data
xmp,ymp = read_in_a_microprobe_data_file(sys.argv[1])
xis,yis,yerris,cis = read_in_an_isotope_data_file(sys.argv[2])


################################################################################
# Beta
################################################################################
beta = 0.25

################################################################################
# Spatial dimensions
################################################################################
#lo = -1155.0e-6
#hi = 1530.0e-6
lo = xmp[0]
hi = xmp[-1]
interface_x = 0
npts = 100
dx   = (hi-lo)/float(npts)

xpos = np.linspace(lo,hi,npts) 

#cmax0 = 0.009 # fraction
#cmin0 = 0.991 # fraction

# For the ``new" files?
cmax0 = 0.0085 # fraction
cmin0 = 1.013 # fraction

frac_interface = (interface_x-lo)/(hi-lo)
print "frac_interface: ",frac_interface

point_of_intial_interface = int(npts*frac_interface)
print "point_of_intial_interface: ", point_of_intial_interface
xvals = np.linspace(lo,hi,npts)

c56 = np.zeros(npts)
c56[:point_of_intial_interface] = cmax0
c56[point_of_intial_interface:] = cmin0

c01 = np.zeros(npts)
c10 = np.zeros(npts)

fig0 = plt.figure(figsize=(14,4))
fig0.add_subplot(1,3,1)
plt.plot(xpos,c56,'o')
plt.ylim(0,1.100)

c54 = c56.copy()

c = c54.copy()

################################################################################
# Move in time
################################################################################
dt   = 10.0
t0   = 0
hours = 120 # For FNDA_1
tmax = (3600*hours) # seconds?

invdx2 = 1.0/(dx**2)

#D = 3.17e-10
#D = 3.17364e-10
#D = 3.185e-10

print "dx: ",dx
print "invdx2: ",invdx2
print "dt: ",dt

t = t0

tag = "default"

#imgcount = 0
#it = 0


################################################################################
while t<tmax:

    if t%10000==0:
        print "%d of %d" % (t,tmax)

    # FNDA1
    #exp(-30.268 + 5.00 xFe - 13.39 xFe^2 + 6.30 xFe^3)
    D56 = np.exp(-30.268 + (5.00*c56) - 13.39*(c56**2) + 6.30*(c56**3))
    # FNDA2
    #exp(-28.838 + 4.92 xFe - 12.91 xFe^2 + 6.17 xFe^3)
    #D56 = np.exp(-28.838 + (4.92*c56) - 12.91*(c56**2) + 6.17*(c56**3))

    # Condition for finite element approach to be stable. 
    if len( (D56*dt*invdx2)[D56*(dt*invdx2)>0.5])>0:
        print "D56*dt*invdx2: ",D56*(dt*invdx2)

    D54 = D56*((56.0/54.0)**beta)

    i = 0
    for D,concentration in zip([D56,D54],[c56, c54]):
        c01 = np.roll(concentration, 1) # This shifts things ``forward", so that this is the previous position.
        c10 = np.roll(concentration,-1) # This shifts things ``backward", so that this is the subsequent position.

        D01 = np.roll(D, 1)
        D10 = np.roll(D,-1)

        # All the points except the end points.
        term1 =  (D10-D01)*(c10-c01)
        term2 =  D*(c10-(2*concentration)+c01)
        concentration_temp = concentration + dt*(term1/(4*dx*dx) + term2/(dx*dx))

        # The end points # NOT RIGHT!!!!! BUT OK FOR NOW
        concentration_temp[0]  = concentration[0]  + (D[0]*(concentration[1] -concentration[0]))
        concentration_temp[-1] = concentration[-1] + (D[-2]*(concentration[-2]-concentration[-1]))

        # Copy it over. 
        if i==0:
            c56 = concentration_temp.copy()
        elif i==1:
            c54 = concentration_temp.copy()

        i += 1


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

print "max: ",max(c56)
print "min: ",min(c56)

################################################################################
# Write out the output
################################################################################
name = "output_%s.dat" % (tag)
outfile = open(name,'w+')

output = ""
for x,y in zip(xpos,c56):
    output += "%f %f\n" % (x,y)
outfile.write(output)
outfile.close()


################################################################################
# Plot the result
################################################################################
#xis -= 0.00126
xis -= 0.00045 # For the new stuff

fig0.add_subplot(1,3,2)
plt.plot(xpos,c56,'bo',label='Fe56 simulation')
plt.plot([interface_x,interface_x],[0,110.0])
plt.ylim(0,1.10)
plt.plot(xmp,ymp,'ro',label='microprobe data')
plt.plot(xis[::-1],cis,'co',label='data from isotope file')
plt.legend()

fig0.add_subplot(1,3,3)
plt.plot(xpos,c54,'bo',label='Fe54 simulation')
plt.plot([interface_x,interface_x],[0,110.0])
plt.ylim(0,1.10)
plt.plot(xmp,ymp,'ro',label='microprobe data')
plt.legend()


plt.figure()
plt.plot(xpos,(c56/c54 - 1.0)*1000.0,'o')
plt.errorbar(xis[::-1],yis,yerr=yerris,markersize=10,fmt='o')
plt.ylim(-20,20)

plt.figure()
plt.plot(xis[::-1],cis,'o')

plt.show()

