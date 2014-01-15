import os, math, sys
import numpy as np
#from ROOT import *

### I want to take ra, dec, z and convert into x,y,z in cartesian coords. This is a simplified version of the conversion code in our ccogs github repository:  https://github.com/djbard/ccogs/blob/master/angular_correlation/utility_scripts/convertRADECZ_XYZ.py

# From Debbie Bard 

"""
usage:
import convertRaDecZtoxyz
x,y,z = convertRaDecZtoxyz.convert(ra, dec, z)
print x, y, z

"""


################################################################################
def getRedshiftMpcFit():
    ### first, I'm gonna plot my cosmo-dependent redshift/distance relation
    redshift_model = [.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    Mpc_model = [209.0, 413.4, 613.5, 808.8, 999.4, 1185.3, 1366.5, 1542.9, 1714.6, 1881.7]

    #gr_model = TGraph(len(redshift_model), np.array(redshift_model), np.array(Mpc_model))
    #gr_model.Fit("pol2")
    #fitresults = gr_model.GetFunction('pol2')

    #p0 = fitresults.GetParameter(0)
    #p1 = fitresults.GetParameter(1)
    #p2 = fitresults.GetParameter(2)
    #print "polynomial fit params: p0 =", p0, ", p1 =", p1, ", p2 =", p2

    params,cov = np.polyfit(redshift_model,Mpc_model,2,cov=True)
    print params
    print cov
    p0 = params[2]
    p1 = params[1]
    p2 = params[0]

    print "polynomial fit params: p0 =", p0, ", p1 =", p1, ", p2 =", p2
    
    return p0, p1, p2





###############################################
### convert redshift z into Mpc
def getMpcFromRedshift(redshift, p0, p1, p2):
    Mpc = p0 + p1*redshift + p2*redshift*redshift
    return Mpc


###############################################
### Convert ra/dec/Mpc coords into x/y/z coords.
def convertRaDecMpcToXYZ(ra, dec, Mpc):
    x, y, z, = 0, 0, 0
    rad = math.pi/180.0
    
    x = Mpc*np.sin(rad*(-1.0*dec+90))*np.cos(rad*(ra))
    y = Mpc*np.sin(rad*(-1.0*dec+90))*np.sin(rad*(ra))
    z = Mpc*np.cos(rad*(-1.0*dec+90))
    
    return x, y, z


#######################################
# Conversion code
#######################################
def convert(ra, dec, z):
    
    ### Fit 2nd order polynomial to Redshift-Mpc relation (based on standard cosmology) and get fit function params.
    p0, p1, p2 = getRedshiftMpcFit()

    ### convert redshift to Mpc
    zMpc = getMpcFromRedshift(z, p0, p1, p2)
    
    
    ### convert ra,dec, Mpc to x,y,z in Mpc
    x, y, zMpc = convertRaDecMpcToXYZ(ra, dec, zMpc)
    
    print 'converted ra=', ra, 'dec=', dec, 'and z=', z, '\n to x=', x, 'y=', y, 'z=', z, 'in Mpc. ' 
    
    return x, y, zMpc


################################################################################
def main():

    infile = open(sys.argv[1])
    content = np.array(infile.read().split()).astype('float')
    ncolumns = 3
    nentries = len(content)
    index = np.arange(0,nentries,ncolumns)
    ra = content[index]
    dec = content[index+1]
    z = content[index+2]

    #print convert(ra[0],dec[0],z[0])
    #print convert(ra,dec,z)
    print convert(0,0,1.2)


################################################################################
if __name__=="__main__":
    main()
