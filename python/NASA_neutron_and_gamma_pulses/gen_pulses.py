import numpy as np
import matplotlib.pylab as plt
import scipy.stats as stats
from scipy.stats import lognorm

# Stuff
nmax = 1
for ncount in xrange(nmax):
    npts_in_peak = 250
    npts = np.random.poisson(npts_in_peak)
    print npts

    tstep = 2e-9

    tmin = -2.71162e-007
    tmax = 7.30838e-007
    trange = tmax-tmin

    sigma = 1.000
    mu = 1.000
    xmax = 10
    x = np.linspace(0,xmax, npts)
    y = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

# Shift the time range
    x /= xmax
    x *= npts
    x *= tstep


    plt.figure(figsize=(16,8))
    plt.subplot(2,2,1)
    plt.plot(x,y,'ro',lw=5, alpha=0.6, label='lognorm pdf',markersize=1)
    plt.legend()

# Add noise 
    noise_sigma = 0.010
    noise = np.random.normal(0,noise_sigma,npts)
    y += noise

    plt.subplot(2,2,2)
    plt.plot(x,y,'bo',lw=5, alpha=0.6, label='lognorm pdf with noise',markersize=1)
    plt.legend()

# Add 0 noise before and after
# Before
    npts_before = 150
    nptsb = np.random.poisson(npts_before)
    noiseb = np.random.normal(0,noise_sigma,nptsb)
    xb = np.arange(-tstep*nptsb,0,tstep)
    #print len(xb)
    #print nptsb
    #print "%.12e %.12e" % (xb[-1],x[0])
    #print xb

# After
    npts_after = 150
    nptsa = np.random.poisson(npts_after)
    noisea = np.random.normal(0,noise_sigma,nptsa)
    xa = np.arange(max(x)+tstep,max(x)+tstep*nptsa+tstep,tstep)
    #print len(xa)
    #print nptsa
    #print "%.12e %.12e" % (xb[-1],x[0])
    #print xb

    xtot = np.zeros(npts+nptsb+nptsa)
    ytot = np.zeros(npts+nptsb+nptsa)

    xtot[0:nptsb] = xb
    ytot[0:nptsb] = noiseb

    xtot[nptsb:nptsb+npts] = x
    ytot[nptsb:nptsb+npts] = y

    xtot[nptsb+npts:nptsb+npts+nptsa] = xa
    ytot[nptsb+npts:nptsb+npts+nptsa] = noisea

    ytot *= -0.70
    ytot -= 0.010

    plt.subplot(2,2,3)
    plt.plot(xtot,ytot,'ko',lw=5, alpha=0.6, label='With noise before and after',markersize=1)
    plt.legend()


# Digitize it
    nsamples = 256.
    ytot *= nsamples
    ytot = ytot.astype(int)
    #print ytot
    #print type(ytot)
    ytot = ytot.astype(float)/nsamples

    plt.subplot(2,2,4)
    plt.plot(xtot,ytot,'ko',lw=1, alpha=0.6, label='With noise',markersize=1)
    plt.legend()

    #print len(ytot)

    #plt.show()

# Write out the output
    output = "LECROYWR204M,51184,Waveform\n"
    output += "Segments,1,SegmentSize,502\n"
    output += "Segment,TrigTime,TimeSinceSegment1\n"
    output += "#1,22-Jan-2015 11:35:11,0\n"
    output += "Time,Ampl\n"

    for a,b in zip(xtot,ytot):
        output += "%.5e,%.8f\n" % (a,b)

    #print output

    outfilename = "neutrons_%03d.csv" % (ncount)
    outfile = open(outfilename,'w+')
    outfile.write(output) 
    outfile.close()




