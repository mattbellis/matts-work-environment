import numpy as np

rho = 5.323 # Ge, g/cm^3

diam = 6.05 # cm
l = 3.1 # cm

r = diam/2.0

pi = np.pi

volume_tot = pi*r*r*l

for i in range(0,10):

    r0 = r - .10 - i*0.0085
    l0 = l - .20 - 2*i*0.0085

    volume_dead = volume_tot - pi*r0*r0*l0

    r1 = r - .20 - i*0.0085
    l1 = l - .40 - 2*i*0.0085

    #print r1

    volume_fiducial = pi*r1*r1*l1

    volume_transition = pi*r0*r0*l0 - volume_fiducial

    print "--------------"
    #print volume_tot
    #print volume_dead
    #print volume_fiducial
    #print volume_transition

    #print 

    #print volume_tot*rho
    #print volume_dead*rho
    #print volume_fiducial*rho
    print volume_transition*rho

    print

