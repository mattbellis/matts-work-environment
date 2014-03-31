import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from datetime import datetime,timedelta

import scipy.integrate as integrate

from cogent_utilities import *
import cogent_pdfs 
import fitting_utilities as fu

import lichen.lichen as lch
import lichen.pdfs as pdfs

################################################################################
################################################################################

xlim = [0.5,3.2]
tlim = [1.0,917.0]

x = np.linspace(xlim[0],xlim[1],10)
t = np.linspace(tlim[0],tlim[1],10)

print x
print t

################################################################################
# Flat
################################################################################
print "----------"
print "Flat PDF"
print "----------"

flatx = pdfs.poly(x,[],xlim[0],xlim[1])
flaty = pdfs.poly(t,[],tlim[0],tlim[1])

print flatx
print flaty
print flatx*flaty

################################################################################
# WIMP: SHM
################################################################################
print "----------"
print "WIMP: SHM"
print "----------"

num_wimps = integrate.dblquad(cogent_pdfs.wimp,xlim[0],xlim[1],lambda x: tlim[0],lambda x:tlim[1],args=(72.0,10.0,2e-41,lambda x:1.0,'shm'),epsabs=0.10)[0]

wimp = cogent_pdfs.wimp(t,x,72,10.0,2e-41,model='shm')

print "num_wimps: ",num_wimps
print wimp
print wimp/num_wimps

print "-----------"
print "ratio"
print "-----------"

print (wimp/num_wimps)/(flatx*flaty)
