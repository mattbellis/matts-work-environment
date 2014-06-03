from pyfeyn.user import *
from math import *
import random

corners = []
corners.append(Point(-4.0,-2.2))
corners.append(Point(-4.0,2.2))
corners.append(Point(4.5,-2.2))
corners.append(Point(4.5,2.2))

#################################################################
#################################################################

def event_topology(outfilename, stage=0):
    #from pyfeyn.user import *

    #trk_color = AQUAMARINE
    #trk_color = YELLOW
    #trk_color = GREY
    trk_color = TAN

    processOptions()
    fd = FeynDiagram()

    ####### Set common borders
    #corners = []
    #corners.append(Point(-4.5,-3.5))
    #corners.append(Point(-4.5,3.5))
    #corners.append(Point(4.5,-3.5))
    #corners.append(Point(4.5,3.5))

    define_border = []
    for item in corners:
        define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))
    ###############

    ##### Beams
    em_in = Point(-3, 0)
    em_out = Point(-0.15, 0)

    ep_in = Point(3, 0)
    ep_out = Point(0.15, 0)

    em = Fermion(em_in, em_out).addLabel(r"\Pem", pos=-0.05, displace=0.01).addArrow(1.05)
    ep = Fermion(ep_in, ep_out).addLabel(r"\Pep", pos=-0.10, displace=0.00).addArrow(1.05)

    frag_angle = []
    track_len_base = 1.5
    if stage==0:
        frag_angle = [10, 30, 50, 70, 86, 100, 120, 150, 170, 178, 185, 200, 220, 240, 255, 280, 300, 320, 345]
        track_len_base = 1.0
    elif stage==1:
        frag_angle = [3, 70, 86, 100, 120, 150, 170, 178, 185, 200, 220, 240, 255, 280, 300, 320, 345]
        track_len_base = 1.0
    elif stage==2:
        frag_angle = [20, 30, 40, 50, 200, 210, 220, 240]
        track_len_base = 1.8

    frags = []
    for a in frag_angle:
        a += random.random()*3.0
        angle = (a/360.0)*2*3.14
        track_len = track_len_base + random.random()/2.0
        xin = 0.0 + 0.20*cos(angle)
        yin = 0.0 + 0.20*sin(angle)
        #xin = Bbarx
        #yin = Bbary
        xout = xin + track_len*cos(angle)
        yout = yin + track_len*sin(angle)
        fin = Point(xin, yin)
        fout = Point(xout, yout)
        frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([trk_color,THICK3]))


    ################################################
    # Do a signal B on the SM processes
    ################################################
    ##### B's
    if stage==1:
        Bangle = 30.0
        angle = (Bangle/360.0)*2*3.14
        Bx = 0.05
        By = 0.05
        B_len = 0.5
        Bx_out = Bx+B_len*cos(angle)
        By_out = By+B_len*sin(angle)
        B_in = Point(Bx,By)
        B_out = Point(Bx_out,By_out)

        #B = Fermion(B_in, B_out).setStyles([CARNATIONPINK,THICK6])
        B = Fermion(B_in, B_out).setStyles([AQUAMARINE,THICK6])

        # B decay products
        lam_decay_angles = [60.0,30.0,10.0,-10.0]
        lam_decay_strings = ["p","Km","piplus"]
        lam_decay = []
        for i,a in enumerate(lam_decay_angles):
            angle = (a/360.0)*2*3.14
            len = 1.3
            xin = Bx_out + 0.00*cos(angle)
            yin = By_out + 0.00*sin(angle)
            xout = xin + len*cos(angle)
            yout = yin + len*sin(angle)
            fin = Point(xin, yin)
            fout = Point(xout, yout)

            lam_decay.append(Fermion(fin, fout).addArrow(1.10).setStyles([AQUAMARINE,THICK4]))


    # Print the file
    outfilename = "%s_%d.pdf" % (outfilename,stage)
    fd.draw(outfilename)
