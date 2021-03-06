import matplotlib.pylab as plt
import numpy as np

masses = [88.0, 87.0, 86.0, 84.0]

################################################################################
# Calculate the diffusion coefficients
################################################################################
def calc_Ds(beta, D_for_Sr88=3.17e-10):

    D = np.array([D_for_Sr88, 0.0, 0.0, 0.0])

    for i in range(1,4):
        # (M1/M2)^beta = (D2/D1)
        # D2 = (M1/M2)^beta * D1
        D[i] = ((masses[0]/masses[i])**beta) * D[0]
        #D[i] = 1.01*D[i-1]

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
#hi = 3200e-6
npts = 1000

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
#pct_isotopes0 = np.array([82.58, 7.0, 9.86, 0.56])
frac_isotopes0 = pct_isotopes0/100.0
print "frac_isotopes0:"
print frac_isotopes0

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

frac10 = np.zeros(npts)
frac01 = np.zeros(npts)

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
    plt.xlabel(r"Position ($\mu$ m)")
    plt.ylabel(r'Concentration (fractional)')
    plt.title(r'Initial concentrations')

bulk_c = c[0] + c[1] + c[2] + c[3]
fig_bulk0 = plt.figure()
axes_bulk0 = fig_bulk0.add_subplot(1,1,1)
axes_bulk0.plot(xpos,bulk_c,'ro')
axes_bulk0.set_ylim(0,7.0)
################################################################################
# Move in time
################################################################################

dt   = 0.01 # This needs to be small (0.01) or it doesn't work.
dx   = (hi-lo)/float(npts)
t0   = 0
tmax = (60*17) + 31.0
#tmax = 100

invdx2 = 1.0/(dx**2)

beta = 0.5
D_for_Sr88 = 3.17364e-10
#D_for_Sr88 = 3.18e-10
D = calc_Ds(beta,D_for_Sr88)

print "dx: ",dx
print "invdx2: ",invdx2
print "dt: ",dt
print "D*dt*invdx2: ",D*(dt*invdx2)

t = t0

for m,f,d in zip(masses,frac_isotopes0,D):
    print m,f,d

tag = "beta=%2.2f_D_Sr88=%5.4e" % (beta,D_for_Sr88)
print tag

Dnum = dt*invdx2
print "Dnum: ",Dnum
Dnum *= D
print "Dnum should be less than 1/2 for stability."
# http://www.me.ucsb.edu/~moehlis/APC591/tutorials/tutorial5/node3.html

print "Dnum: ",Dnum

print "dt*invdx2: ",dt*invdx2

imgcount = 0
it = 0

################################################################################
# Start the time steps
################################################################################
print "tmax: ",tmax
while t<tmax:

    if (t%100)<=dt:
        print t

    #print "bulk: ----------- "
    bulk_c = c[0] + c[1] + c[2] + c[3]
    bulk_c01 = np.roll(bulk_c, 1)
    bulk_c10 = np.roll(bulk_c,-1)

    # Loop over all 4 isotopes and propogate them individually.
    for i in range(4):

        frac = c[i]/bulk_c

        # Roll over the others to speed up the calculation
        c01[i] = np.roll(c[i], 1) # The i-1 val
        c10[i] = np.roll(c[i],-1) # The i+1 val

        ########################################################################
        # Using the bulk concentration to decide how much moves and how many
        # move.
        ########################################################################
        #concentration_temp = c[i] + frac*Dnum[i]*(bulk_c01-bulk_c + bulk_c10-bulk_c)
        # The end points
        #concentration_temp[0]  = c[i][0]  + frac[0]*Dnum[i]*((bulk_c[1]  -bulk_c[0]))
        #concentration_temp[-1] = c[i][-1] + frac[-1]*Dnum[i]*((bulk_c[-2]-bulk_c[-1]))

        ########################################################################
        # Using the bulk concentration to decide how much moves, but the relative
        # fraction of those events to calculate how many move.
        ########################################################################
        '''
        frac01 = np.roll(frac, 1) # The i-1 val
        frac10 = np.roll(frac,-1) # The i+1 val

        addition01 = (bulk_c01-bulk_c)
        #addition01 *= (frac + frac01)/2.0
        addition01[addition01<0] *= frac[addition01<0]
        addition01[addition01>0] *= frac01[addition01>0]

        addition10 = (bulk_c10-bulk_c)
        #addition10 *= (frac + frac10)/2.0
        addition10[addition10<0] *= frac[addition10<0]
        addition10[addition10>0] *= frac10[addition10>0]

        concentration_temp = c[i] + Dnum[i]*(addition01 + addition10)

        # The end points
        a01 = bulk_c[1] -bulk_c[0]
        #concentration_temp[0]  = c[i][0]  + ((frac[0]+frac[1])/2.0)*Dnum[i]*a01
        #if a01<0:
            #concentration_temp[0]  = c[i][0]  + frac[0]*Dnum[i]*a01
        #elif a01>0:
            #concentration_temp[0]  = c[i][0]  + frac[1]*Dnum[i]*a01

        a10 = (bulk_c[-2]-bulk_c[-1])
        #concentration_temp[-1]  = c[i][-1]  + ((frac[-2]+frac[-1])/2.0)*Dnum[i]*a10
        #if a10<0:
            #concentration_temp[-1] = c[i][-1] + frac[-1]*Dnum[i]*a10
        #elif a10>0:
            #concentration_temp[-1] = c[i][-1] + frac[-2]*Dnum[i]*a10
        '''
        ########################################################################
        # Using the species' concentration
        ########################################################################
        concentration_temp = c[i] + Dnum[i]*(c01[i]-c[i] + c10[i]-c[i])
        # The end points
        concentration_temp[0]  = c[i][0]  + Dnum[i]*((c[i][1] - c[i][0]))
        concentration_temp[-1] = c[i][-1] + Dnum[i]*((c[i][-2]-c[i][-1]))

        # Copy over the temporary array
        c[i] = concentration_temp.copy()

    t += dt

    # Uncomment this section to make images
    '''
    if it%1000==0:
        print "here"
        ax.plot(xpos,bulk_c,'o')
        ax.set_ylim(0,7.0)
        name = "Plots/img%03d.png" % (imgcount)
        fig_img.savefig(name)
        imgcount += 1
        ax.clear()
    it += 1
    '''

################################################################################
# Print out the max and min values for the plots
################################################################################
print "End points -----------"
bulk_c = c[0] + c[1] + c[2] + c[3]
print "max: ",max(bulk_c)
print "min: ",min(bulk_c)
for i,con in enumerate(c):
    print i
    print "max: ",max(con)
    print "min: ",min(con)



################################################################################
# Plot the result
################################################################################
fig1 = plt.figure()
axes1 = []
#print c[0]
for i in range(0,4):
    axes1.append(fig1.add_subplot(2,2,i+1))
    plt.plot(xpos,c[i],'o')
    #plt.ylim(0,7.0)

# Plot the bulk concentration
fig_bulk1 = plt.figure()
axes_bulk1 = fig_bulk1.add_subplot(1,1,1)
axes_bulk1.plot(xpos,bulk_c,'ro')
axes_bulk1.set_ylim(0,7.0)

################################################################################
# Calculate and plot the deltas
################################################################################
fig_deltas = []
axes_deltas = []
vals = [None,None,None,None]
for i in range(0,1):
    fig_deltas.append(plt.figure())
    axes_deltas.append([])
    for j in range(0,4):
        axes_deltas[i].append(fig_deltas[i].add_subplot(2,2,j+1))
        if i!=j:
            vals[i] = ((c[j]/frac_isotopes0[j]) / (c[i]/frac_isotopes0[i])  - 1.0)*1000.0
            axes_deltas[i][j].plot(xpos,vals[i],'o')
            #plt.ylim(0,7.0)



################################################################################
# Write out the data.
################################################################################

name = "output_%s.dat" % (tag)
outfile = open(name,'w+')

output = ""
#for xpt,y0,y1,y2,y3,y4,y5,y6,y7 in zip(xpos,bulk_c,c[0],c[1],c[2],c[3],vals[0],vals[1],vals[2]):
    #output += "%f %f %f %f %f %f %f %f %f\n" % (xpt,y0,y1,y2,y3,y4,y5,y6,y7)
for xpt,y0,y1,y2,y3,y4,y5 in zip(xpos,bulk_c,c[0],c[1],c[2],c[3],vals[0]):
    output += "%f %f %f %f %f %f %f\n" % (xpt,y0,y1,y2,y3,y4,y5)
outfile.write(output)
outfile.close()

plt.show()

