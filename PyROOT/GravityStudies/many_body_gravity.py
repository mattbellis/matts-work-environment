#!/usr/bin/env python

from numpy import array
import ROOT
from ROOT import *

from color_palette import *

import sys
from optparse import OptionParser

import re

import random as rnd

################################################################################
################################################################################

################################################################################
################################################################################
def main(argv):

    parser = OptionParser()
    parser.add_option("-m", "--max", dest="max", default=1e9, 
            help="Max games to read in.")
    parser.add_option("--tag", dest="tag", default=None, 
            help="Tag for output files.")
    parser.add_option("--batch", dest="batch", default=False, 
            action="store_true", help="Run in batch mode.")

    (options, args) = parser.parse_args()


    ############################################################################
    # Set some physics quantitites
    ############################################################################
    rnd = TRandom3()
    grav_constant = 100.0
    mass = 5.0 
    position_scale = 100.0

    # Save time, radius and mag of momentum
    recorded_values = [array('d'),array('d'),array('d')]

    ############################################################################
    # Build the particles using velocity and position
    ############################################################################
    num_particles = 4
    particles = []
    for i in range(0,num_particles):

        x = position_scale*(rnd.Rndm()-0.5)
        y = position_scale*(rnd.Rndm()-0.5)
        z = position_scale*(rnd.Rndm()-0.5)


        # Place the particles at 10.0 and -10.0 on z-axis with 0 momentum.
        pos = TVector3(x,y,z)
        vel = TVector3(0.0,0.0,0.0)
        prev_pos = TVector3(pos)

        particles.append((pos,vel,prev_pos))

    ############################################################################
    # Calculate the motion of two bodies
    ############################################################################
    t = 0
    dt = 0.10
    num_time_steps = 0

    # Declare these once for use later
    force = TVector3()
    direction = TVector3()

    minimum_distance_allowable = 0.010
    smallest_distance = 1000000.0

    while smallest_distance>minimum_distance_allowable:

        #print t
        if num_time_steps%1000==0:
            print "%f %d" % (t,num_time_steps)

        # Copy the current position to the previous position
        for p in (particles):
            p[2].SetXYZ(p[0].X(),p[0].Y(),p[0].Z())

        #print " ---------- "
        for p in particles:

            smallest_distance = 1000000.0

            # Clear out the force vector 
            force.SetXYZ(0.0,0.0,0.0)

            # Calculate the forces and then move them

            # Loop over all the other particles to calculate the sum of the 
            # forces on our one particle
            for q in particles:

                if p is not q:

                    #print "print pz and qz: %f %f" % (p[2].Z(),q[2].Z())
                    
                    direction.SetXYZ(p[2].X(),p[2].Y(),p[2].Z())
                    direction -= q[2]
                    distance = direction.Mag()
                    #print "distance: %f" % (distance)
                    if distance<smallest_distance:
                        smallest_distance=distance

                    # Take into account the normalization that will have to be done.
                    force_mag =  grav_constant * mass * mass/(distance*distance*distance)
                    # direction will now actually be the full force vector
                    direction *= (-force_mag)

                    force += direction

            # Update the momentum
            # Make force the momentum and then add it
            force *= dt
            #force.Print()
            p[1].SetXYZ(p[1].X()+force.X(), p[1].Y()+force.Y(), p[1].Z()+force.Z())

            # Update the new position
            p[0].SetXYZ(p[0].X()+p[1].X()*dt, p[0].Y()+p[1].Y()*dt, p[0].Z()+p[1].Z()*dt)

            # Save the radius and the magnitude of momentum
            recorded_values[0].append(t)
            #print "recorded values: %f %f" % (p[0].Mag(),p[1].Mag())
            recorded_values[1].append(p[0].Mag())
            recorded_values[2].append(p[1].Mag())


        #print abs(smallest_distance)
        #print dt
        if abs(smallest_distance)<10.0:
            dt = abs(smallest_distance)/10000.0

        t += dt
        num_time_steps += 1
        #'''
        if num_time_steps>10000:
            break
        #'''
                    

    outfile = open("many_bodies.txt","w+")
    print "num_time_steps: %d" % (num_time_steps)
    num_entries = len(recorded_values[0])
    for i in range(0,num_entries):
        output = ""
        output += "%f %f %f\n" % (recorded_values[0][i],recorded_values[1][i],recorded_values[2][i])
        outfile.write(output)

    outfile.close()

################################################################################
################################################################################
if __name__ == '__main__':
    main(sys.argv)



