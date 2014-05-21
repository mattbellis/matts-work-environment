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

    # Style options
    gStyle.SetOptStat(0)
    set_palette("palette",50)

    ############################################################################
    # Declare the canvases
    ############################################################################
    num_can = 1
    can = []
    for i in range(0,num_can):
        name = "can%d" % (i)
        if i<2:
            can.append(TCanvas(name,"",10+10*i,10+10*i,1350,700))
        else:
            can.append(TCanvas(name,"",10+10*i,10+10*i,700,700))
        can[i].SetFillColor(0)
        can[i].Divide(1,1)

    ############################################################################
    # Declare some histograms 
    ############################################################################
    lo = 0.0
    hi = 1.0
    color = 2
    nbins = 100
    h = []
    for i in range(0,5):
        name = "h%d" % (i)
        h.append(TH1F(name,"",nbins, lo, hi))
        h[i].SetFillStyle(1000)
        h[i].SetFillColor(color)
        h[i].SetTitle("")

        h[i].SetNdivisions(8)
        h[i].GetYaxis().SetTitle("# occurances")
        h[i].GetYaxis().SetTitleSize(0.09)
        h[i].GetYaxis().SetTitleFont(42)
        h[i].GetYaxis().SetTitleOffset(0.7)
        h[i].GetYaxis().CenterTitle()

        h[i].GetXaxis().SetTitle("Arbitrary measurements")
        h[i].GetXaxis().SetLabelSize(0.12)
        h[i].GetXaxis().SetTitleSize(0.10)
        h[i].GetXaxis().SetTitleFont(42)
        h[i].GetXaxis().SetTitleOffset(1.0)
        h[i].GetXaxis().CenterTitle()

        h[i].SetMinimum(0)

    h2D = []
    for i in range(0,5):
        name = "h2D%d" % (i)
        h2D.append(TH2F(name,"",nbins, lo, hi, nbins, lo, hi))
        h2D[i].SetFillStyle(1000)
        h2D[i].SetFillColor(color)
        h2D[i].SetTitle("")

        h2D[i].SetNdivisions(8)
        h2D[i].GetYaxis().SetTitleSize(0.09)
        h2D[i].GetYaxis().SetTitleFont(42)
        h2D[i].GetYaxis().SetTitleOffset(0.7)
        h2D[i].GetYaxis().CenterTitle()
        h2D[i].GetYaxis().SetTitle("Visitng team")
        h2D[i].GetXaxis().SetTitle("Home team")
        #h2D[i].GetXaxis().SetLabelSize(0.09)
        h2D[i].GetXaxis().SetTitleSize(0.09)
        h2D[i].GetXaxis().SetTitleFont(42)
        h2D[i].GetXaxis().SetTitleOffset(0.7)
        h2D[i].GetXaxis().CenterTitle()

        h2D[i].SetMinimum(0)

    ############################################################################
    # Set some physics quantitites
    ############################################################################
    grav_constant = 100.0
    mass = 5.0 

    # Save time, radius and mag of momentum
    recorded_values = [array('d'),array('d'),array('d')]

    ############################################################################
    # Build the particles using velocity and position
    ############################################################################
    num_particles = 2.0
    particles = []
    for i in range(0,num_particles):

        # Place the particles at 10.0 and -10.0 on z-axis with 0 momentum.
        z = 100.0
        if i==1:
            z = -100.0
        pos = TVector3(0.0,0.0,z)
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

    minimum_distance_allowable = 1.0
    smallest_distance = 1000000.0

    while smallest_distance>minimum_distance_allowable:

        print t

        # Copy the current position to the previous position
        for p in (particles):
            p[2].SetXYZ(p[0].X(),p[0].Y(),p[0].Z())

        for p in particles:

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
                    print "distance: %f" % (distance)
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
            force.Print()
            p[1].SetXYZ(p[1].X()+force.X(), p[1].Y()+force.Y(), p[1].Z()+force.Z())

            # Update the new position
            p[0].SetXYZ(p[0].X()+p[1].X()*dt, p[0].Y()+p[1].Y()*dt, p[0].Z()+p[1].Z()*dt)

            # Save the radius and the magnitude of momentum
            recorded_values[0].append(t)
            print "recorded values: %f %f" % (p[0].Mag(),p[1].Mag())
            recorded_values[1].append(p[0].Mag())
            recorded_values[2].append(p[1].Mag())


        dt = smallest_distance/1000.0
        t += dt
        num_time_steps += 1
        if num_time_steps>500:
            break
                    

    print "num_time_steps: %d" % (num_time_steps)

    npts = len(recorded_values[0])

    gr = TGraph(npts,recorded_values[1],recorded_values[2])
    
    can[0].cd(1)
    gr.Draw("ap*")
    gPad.Update()
            



    '''
    if options.tag != None:
        name = "Plots/sportsplots%s_%d.eps" % (options.tag,i)
        can[0].SaveAs(name)
    '''

    ################################################################################
    ## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
    if not options.batch:
        rep = ''
        while not rep in [ 'q', 'Q' ]:
            rep = raw_input( 'enter "q" to quit: ' )
            if 1 < len(rep):
                rep = rep[0]

################################################################################
################################################################################
if __name__ == '__main__':
    main(sys.argv)



