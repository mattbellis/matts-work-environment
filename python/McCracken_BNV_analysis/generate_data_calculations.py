#!/usr/bin/env python

import sys
from math import *

import numpy as np
import matplotlib.pyplot as plt

from analysis_utilities import *

################################################################################
infile = open(sys.argv[1])
outfile = open(sys.argv[2],"w+")

p = [0.0, 0.0, 0.0, 0.0, 0.0]
n = 0
masses = []
masses_inv = []
masses_mm = []
flight_length = []
vtx = [0.0,0.0,0.0,0.0]
count = 0
beam = None
nvtx = 0
output = ""
v3 = [0.0, 0.0, 0.0]

'''
masses = [[mass_k,mass_p,mass_pi],
          [mass_p,mass_k,mass_pi]]
'''

masses = [[mass_k,mass_e,mass_k],
          [mass_e,mass_k,mass_k]]


for line in infile:
    vals = line.split()
    if len(vals)==2:

        beam_e = float(vals[1])
        beam = np.array([beam_e, 0.0, 0.0, beam_e])

    elif len(vals)==5:
        #print vals

        x = [float(vals[0]),float(vals[1]),float(vals[2])]
        v3[n] = np.array(x)

        n += 1

        if n>=3:

            ############ Do some calculations #######################
            for j in xrange(2):

                #### Go through the two mass hypothesis.
                for i,mass in enumerate(masses[j]):

                    pmag = magnitude_of_3vec(v3[i])
                    energy = sqrt(mass*mass + pmag*pmag)
                    p[i] = np.insert(v3[i],[0],energy)

                ##### Missing mass
                p4 = beam + target - p[0] - p[1] - p[2]
                output += "%f," % (mass_from_special_relativity(p4))

                # Assuming first + particle is a kaon
                if j==0:
                    p4 = beam + target - p[0]
                    output += "%f," % (mass_from_special_relativity(p4))

                    p4 = p[1]+p[2]
                    output += "%f," % (mass_from_special_relativity(p4))

                # Assuming second + particle is a kaon
                elif j==1:
                    p4 = beam + target - p[1]
                    output += "%f," % (mass_from_special_relativity(p4))

                    p4 = p[0]+p[2]
                    output += "%f," % (mass_from_special_relativity(p4))

            n=0
            count +=1 
            if count%10000==0:
                print count
            if count > 1000000:
                break

    elif len(vals)==3:

        x = [float(vals[0]),float(vals[1]),float(vals[2])]
        v3vtx = np.array(x)

        vtx[nvtx] = v3vtx

        nvtx += 1

        if nvtx>=4:
            ############ Do some calculations #######################
            output += "%f\n" % (magnitude_of_3vec(vtx[0]-vtx[1]))

            nvtx = 0
            outfile.write(output)
            output = ""


outfile.close()
