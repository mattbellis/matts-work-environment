import numpy as np
from numpy import sin,cos,arccos
from astropy.cosmology import FlatLambdaCDM


def radecz2xyz(ra,dec,comdist):

    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    # Convert spherical to Cartesian Coordinates
    x=comdist*np.sin(dec)*np.cos(ra)
    y=comdist*np.sin(dec)*np.sin(ra)
    z=comdist*np.cos(dec)

    return x,y,z


def angular_sep(ra,dec):

    r1 = np.deg2rad(ra[0])
    r2 = np.deg2rad(ra[1])

    d1 = np.deg2rad(dec[0])
    d2 = np.deg2rad(dec[1])

    #cos(A) = sin(d1)sin(d2) + cos(d1)cos(d2)cos(ra1-ra2)

    cosA = sin(d1)*sin(d2) + cos(d1)*cos(d2)*cos(r1-r2)


    A = np.arccos(cosA)
    #print "cosA: ",cosA
    #print "   A: ",A

    return A

'''
gals = [
        [218.226380,28.818504,0.51],
        [221.685380,28.339808,0.497408],
        ] 
'''

#'''
gals = [[144.461360,6.627704,0.684696],
        [218.226380,28.818504,0.676657],
        [212.589070,22.507323,0.534445],
        [251.936440,16.800100,0.580227],
        [237.138020,22.639834,0.520722],
        [209.363730,36.513512,0.526835],
        [221.685380,28.339808,0.497408],
        [139.714610,8.968519,0.461584],
        [151.888920,40.852866,0.515396],
        [151.904610,13.562258,0.453322]] 
#'''


cosmo=FlatLambdaCDM(H0=70,Om0=0.3)
#comdist=cosmo.comoving_distance(redshift).value
#cosmo.kpc_comoving_per_arcmin(redshift).value


for g1 in gals:
    for g2 in gals:
        asep = angular_sep([g1[0],g2[0]],[g1[1],g2[1]])
        asep_in_min = np.rad2deg(asep)*60.
        redshift1 = g1[2]
        redshift2 = g2[2]

        avgredshift = (redshift1+redshift2)/2.

        x = cosmo.kpc_comoving_per_arcmin(redshift1).value 
        #x = cosmo.kpc_comoving_per_arcmin(avgredshift).value 
        x *= asep_in_min/1000.0 # Convert to Mpc

        d1 = cosmo.comoving_distance(redshift1).value
        d2 = cosmo.comoving_distance(redshift2).value

        y = d2-d1

        s = np.sqrt(x*x + y*y)

        print "------------"
        print g1
        print g2
        print "asep: ",asep,np.rad2deg(asep)
        print "x,y: ",x,y
        print "d1,d2: ",d1,d2
        print s

        x1,y1,z1 = radecz2xyz(g1[0],g1[1],d1)
        x2,y2,z2 = radecz2xyz(g2[0],g2[1],d2)

        print x1,y1,z1
        print x2,y2,z2

        sother = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )

        print sother



