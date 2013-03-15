import matplotlib.pylab as plt
import numpy as np

masses = [88.0, 87.0, 86.0, 84.0]

################################################################################
# Calculate the diffusion coefficients
################################################################################
def calc_Ds(beta):

    #D = np.array([3.17e-10, 0.0, 0.0, 0.0])
    D = np.array([3.5e-10, 0.0, 0.0, 0.0])

    for i in range(1,4):
        # (M1/M2)^beta = (D2/D1)
        # D2 = (M1/M2)^beta * D1
        D[i] = ((masses[0]/masses[i])**beta) * D[0]

    '''
    print "beta",beta
    for m,d in zip(masses,D):
        print m,d
    '''

    return D


################################################################################
# Spatial dimensions
################################################################################
lo = 0
hi = 2775e-6
npts = 500

xpos = np.linspace(lo,hi,npts)

################################################################################
# Figure out where the boundary is wrt our individual
# spatial points.
################################################################################
frac_interface = hi/1520e-6

point_of_intial_interface = int(npts/frac_interface)


################################################################################
# Initial isotope information
################################################################################
cmax0 = 6.13
cmin0 = 0.62

pct_isotopes0 = np.array([83.05, 6.89, 9.537, 0.52])
frac_isotopes0 = pct_isotopes0/100.0


concentration = np.zeros(npts)
concentration[:point_of_intial_interface] = cmax0
concentration[point_of_intial_interface:] = cmin0

# Declare and initialize the concentrations
c = []
c01 = []
c10 = []

bulk_c = np.zeros(npts)
bulk_c01 = np.zeros(npts)
bulk_c10 = np.zeros(npts)

for i in range(4):

    c.append(np.zeros(npts))
    c01.append(np.zeros(npts))
    c10.append(np.zeros(npts))

    c[i][:point_of_intial_interface] = cmax0*frac_isotopes0[i]
    c[i][point_of_intial_interface:] = cmin0*frac_isotopes0[i]

    c01.append(np.zeros(npts))
    c10.append(np.zeros(npts))

fig0 = plt.figure()
axes0 = []
for i in range(0,4):
    axes0.append(fig0.add_subplot(2,2,i+1))
    plt.plot(xpos,c[i],'o')
    plt.ylim(0,7.0)

bulk_c = c[0] + c[1] + c[2] + c[3]
fig_bulk0 = plt.figure()
axes_bulk0 = fig_bulk0.add_subplot(1,1,1)
axes_bulk0.plot(xpos,bulk_c,'ro')
axes_bulk0.set_ylim(0,7.0)
################################################################################
# Move in time
################################################################################

dt   = 0.01 # This needs to be small (0.01) or it doesn't work.
            # We got discontinuities at the boundary at the first step.
            # The concentrations ``crossed over into one another".
dx   = (hi-lo)/float(npts)
t0   = 0
tmax = (60*17) + 31

invdx2 = 1.0/(dx**2)

#D = 3.17e-10
#D = 3.17364e-10
#D = 3.185e-10

beta = 0.5
D = calc_Ds(beta)

print "dx: ",dx
print "invdx2: ",invdx2
print "dt: ",dt
print "D*dt*invdx2: ",D*(dt*invdx2)

t = t0

for m,f,d in zip(masses,frac_isotopes0,D):
    print m,f,d

#tag = "default"
#tag = "D=3.17"
#tag = "D=3.17364"
#tag = "D=3.185"
tag = "beta=%2.1f" % (beta)

#fig_img = plt.figure()
#ax = fig_img.add_subplot(1,1,1)

imgcount = 0
it = 0

Dnum = dt*invdx2
Dnum *= D

print "Dnum: ",Dnum

################################################################################
# Start the time steps
################################################################################
while t<tmax:

    if t%100==0:
        print t

    #print "bulk: ----------- "
    bulk_c = c[0] + c[1] + c[2] + c[3]
    bulk_c01 = np.roll(bulk_c, 1)
    bulk_c10 = np.roll(bulk_c,-1)

    #print bulk_c[265:280]

    for i in range(4):

        frac = c[i]/bulk_c
        #print frac[265:280]
        #print frac

        '''
        if i==0:
            print i
            print frac[265:280]
            print c[i][265:280]
            print Dnum[i]
        '''
        # All the points except the end points.
        concentration_temp = c[i] + frac*Dnum[i]*(bulk_c01-bulk_c + bulk_c10-bulk_c)

        # The end points
        concentration_temp[0]  = c[i][0]  + frac[0]*Dnum[i]*((bulk_c[1]  -bulk_c[0]))
        concentration_temp[-1] = c[i][-1] + frac[-1]*Dnum[i]*((bulk_c[-2]-bulk_c[-1]))

        # Copy it over. 
        c[i] = concentration_temp.copy()

        '''
        if i==0:
            print c[i][265:280]
        '''

    t += dt

    '''
    if t>t0 + 100*dt:
        exit()
    '''

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

for i,con in enumerate(c):
    print i
    print "max: ",max(con)
    print "min: ",min(con)

################################################################################
# Write out the output
################################################################################
'''
name = "output_%s.dat" % (tag)
outfile = open(name,'w+')

output = ""
for x,y in zip(xpos,concentration):
    output += "%f %f\n" % (x,y)
outfile.write(output)
outfile.close()
'''


################################################################################
# Plot the result
################################################################################
fig1 = plt.figure()
axes1 = []
for i in range(0,4):
    axes1.append(fig1.add_subplot(2,2,i+1))
    plt.plot(xpos,c[i],'o')
    #plt.ylim(0,7.0)

bulk_c = c[0] + c[1] + c[2] + c[3]
fig_bulk1 = plt.figure()
axes_bulk1 = fig_bulk1.add_subplot(1,1,1)
axes_bulk1.plot(xpos,bulk_c,'ro')
axes_bulk1.set_ylim(0,7.0)
plt.show()

